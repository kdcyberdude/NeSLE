#include "nesle/cuda/kernels.cuh"

#ifdef __CUDACC__

#include "nesle/cuda/batch_console.hpp"
#include "nesle/cuda/batch_render.cuh"
#include "nesle/cuda/batch_step.cuh"

namespace nesle::cuda {
namespace {

__global__ void step_reward_kernel(BatchBuffers buffers, std::uint32_t num_envs) {
    const auto env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env < num_envs) {
        apply_batch_reward_env(buffers, env);
    }
}

// Banked cartridges and mapper 0 are separate kernel instantiations, selected on
// the host in launch_console_step_kernel. Keeping them in one __global__ would
// make ptxas allocate for the worse of the two bodies and cost occupancy on the
// NROM path (measured: 72 registers versus 56 when both live in one kernel).
//
// The frame log's once-a-frame work (batch_frame.cuh) is called out of line and takes
// `buffers` by reference. A __grid_constant__ parameter lets that reference point at the
// launch parameter itself; otherwise the kernel copies it to local memory and loads every
// buffer pointer from there (sm_70+; nothing here writes the struct, only what it points
// at). The cap (96 registers, 5 blocks per SM) keeps the occupancy NROM and banked
// boards had before those calls (94; paged boards had 72): without it the calls'
// register needs raise the whole kernel to ~160.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
#define NESLE_STEP_GRID_CONSTANT
#else
#define NESLE_STEP_GRID_CONSTANT __grid_constant__
#endif
template <BankMode Mode>
__global__ void __launch_bounds__(128, 5)
    console_step_kernel(const NESLE_STEP_GRID_CONSTANT BatchBuffers param,
                        std::uint32_t num_envs,
                        std::uint32_t frameskip,
                        std::uint64_t max_instructions_per_frame,
                        ConsoleStepStats stats) {
    BatchBuffers& buffers = const_cast<BatchBuffers&>(param);
    const auto env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= num_envs) {
        return;
    }

    // Already quarantined by an earlier step and not reset since. The fault stays set
    // until a reset, so without this check the next step_device would resume from the
    // byte after the offending opcode and carry on as though nothing had happened.
    // Report it as done every time, with no reward, rather than executing.
    if (buffers.cpu.fault != nullptr && buffers.cpu.fault[env] != 0) {
        buffers.rewards[env] = 0.0F;
        buffers.done[env] = 1;
        return;
    }

    auto state = load_cpu_state(buffers, env);
    // PPU scalars stay in registers for the whole launch, like CpuState —
    // they are otherwise ~6-10 dependent global round-trips per instruction.
    auto hot = load_ppu_hot_state(buffers, env);
    // Lazy PPU settlement: PPU scalar state only changes at three timing
    // events (sprite-0 hit, vblank, pre-render), so instead of advancing the
    // PPU every instruction we accumulate cycles and settle only when the
    // accumulated span reaches the next event horizon. Between settles the
    // register-resident PPU state the bus reads is exact by construction.
    std::uint32_t pending_ppu_cycles = 0;
    std::uint32_t dots_to_next_event = batch_ppu_cycles_until_next_event_hot<Mode>(buffers, env, hot);
    std::uint64_t total_instructions = 0;
    std::uint32_t total_frames_completed = 0;
    std::uint32_t budget_hits = 0;
    // Set when this env's CPU reaches an opcode the decoder does not implement. The
    // env is then quarantined instead of taking the whole launch down with it.
    bool faulted = false;
    for (std::uint32_t frame = 0; frame < frameskip; ++frame) {
        std::uint64_t instructions = 0;
        std::uint32_t frames_completed = 0;
        while (instructions < max_instructions_per_frame && frames_completed == 0) {
            if (pending_ppu_cycles >= dots_to_next_event) {
                const auto settled =
                    batch_ppu_step_env_hot<Mode>(buffers, env, pending_ppu_cycles, hot);
                frames_completed += settled.frames_completed;
                pending_ppu_cycles = 0;
                dots_to_next_event = batch_ppu_cycles_until_next_event_hot<Mode>(buffers, env, hot);
                if (frames_completed != 0) {
                    break;
                }
            }
            const auto step =
                step_batch_console_instruction_lazy<Mode>(buffers, env, state, hot, pending_ppu_cycles);
            if (step.cpu.illegal) {
                // An opcode the core does not implement. This used to compile to
                // asm("trap;") inside cpu::step, which aborts the entire launch: one
                // env out of 16k reaching a byte the decoder does not know took every
                // other env down with an opaque CUDA error and no hint which one.
                // Quarantine this env instead - record where and what, mark it done so
                // the episode ends cleanly, and let every other env finish the frame.
                if (buffers.cpu.fault != nullptr) {
                    buffers.cpu.fault[env] =
                        (static_cast<std::uint32_t>(step.cpu.pc) << 8) | step.cpu.opcode;
                }
                if (buffers.cpu.fault_pending != nullptr) {
                    // Latched separately: the env is about to be marked done, and
                    // step_device resets done envs in the same call, so a fault kept
                    // only in `fault` would be cleared before anyone could read it.
                    buffers.cpu.fault_pending[env] =
                        (static_cast<std::uint32_t>(step.cpu.pc) << 8) | step.cpu.opcode;
                }
                if (pending_ppu_cycles != 0) {
                    const auto settled =
                        batch_ppu_step_env_hot<Mode>(buffers, env, pending_ppu_cycles, hot);
                    frames_completed += settled.frames_completed;
                    pending_ppu_cycles = 0;
                }
                faulted = true;
                break;
            }
            pending_ppu_cycles += step.ppu_cycles;
            if (stats.opcode_counts != nullptr) {
                atomicAdd(&stats.opcode_counts[step.cpu.opcode], 1ULL);
            }
            if (stats.pc_counts != nullptr) {
                atomicAdd(&stats.pc_counts[step.cpu.pc], 1ULL);
            }
            ++instructions;
            if (step.cpu.opcode == 0x4C && state.pc == step.cpu.pc && frames_completed == 0) {
                // Idle loop: settle what we owe, then fast-forward to the event.
                const auto settled =
                    batch_ppu_step_env_hot<Mode>(buffers, env, pending_ppu_cycles, hot);
                frames_completed += settled.frames_completed;
                pending_ppu_cycles = 0;
                const auto fast_forward =
                    fast_forward_batch_console_idle_loop_hot<Mode>(buffers, env, state, hot);
                frames_completed += fast_forward.frames_completed;
                dots_to_next_event = batch_ppu_cycles_until_next_event_hot<Mode>(buffers, env, hot);
            }
        }
        // Settle the tail so frame accounting is exact at frameskip boundaries.
        if (pending_ppu_cycles != 0) {
            const auto settled = batch_ppu_step_env_hot<Mode>(buffers, env, pending_ppu_cycles, hot);
            frames_completed += settled.frames_completed;
            pending_ppu_cycles = 0;
            dots_to_next_event = batch_ppu_cycles_until_next_event_hot<Mode>(buffers, env, hot);
        }
        total_instructions += instructions;
        total_frames_completed += frames_completed;
        if (frames_completed == 0 && instructions >= max_instructions_per_frame) {
            ++budget_hits;
        }
        if (faulted) {
            // Stop early: the env is quarantined, so the remaining frames of this
            // frameskip would only re-enter the same unimplemented opcode.
            break;
        }
    }
    store_cpu_state(buffers, env, state);
    store_ppu_hot_state(buffers, env, hot);

    if (faulted) {
        // End the episode with no reward rather than running the reward kernel: this
        // env's frame is truncated mid-instruction, so any progress-based term would
        // be measuring the cutoff, not the game. Python sees the reason through
        // CudaBatch.faults().
        buffers.rewards[env] = 0.0F;
        buffers.done[env] = 1;
    } else {
        apply_batch_reward_env(buffers, env);
    }
    if (stats.instructions != nullptr) {
        stats.instructions[env] = total_instructions;
    }
    if (stats.frames_completed != nullptr) {
        stats.frames_completed[env] = total_frames_completed;
    }
    if (stats.budget_hits != nullptr) {
        stats.budget_hits[env] = budget_hits;
    }
}

// One block per env: thread 0 replays the frame's log once to get FCEUmm's state at the
// start of every line, then the threads draw lines in parallel from those states.
__global__ void render_rgb_kernel(BatchBuffers buffers, std::uint32_t num_envs) {
    __shared__ ReplayState states[kFrameHeight];
    const auto env = blockIdx.x;
    if (env >= num_envs) {
        return;
    }
    std::uint32_t scratch[kLogHeaderWords];
    const ReplaySource src = render_source(buffers, env, scratch);
    if (threadIdx.x == 0) {
        render_line_states(src, states);
    }
    __syncthreads();
    std::uint8_t* frame = env_frame_rgb(buffers, env);
    ReplayLine line;
    for (std::uint32_t l = threadIdx.x; l < kFrameHeight; l += blockDim.x) {
        ReplayState s = states[l];
        render_line_rgb(src, s, line, static_cast<int>(l), frame + l * kFrameWidth * kRgbChannels);
    }
}

// One thread per env, serial frame paint: once the env count alone saturates the GPU
// (measured crossover ~ several hundred envs on a GTX 1050 Ti for the earlier
// renderer) this does less total work.
__global__ void render_rgb_kernel_serial(BatchBuffers buffers, std::uint32_t num_envs) {
    const auto env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= num_envs) {
        return;
    }
    render_batch_rgb_frame_env(buffers, env);
}

}  // namespace

void launch_step_kernel(const BatchBuffers& buffers, StepConfig config, cudaStream_t stream) {
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((config.num_envs + kThreads - 1) / kThreads);
    step_reward_kernel<<<blocks, kThreads, 0, stream>>>(buffers, config.num_envs);
}

void launch_console_step_kernel(const BatchBuffers& buffers,
                                StepConfig config,
                                std::uint64_t max_instructions_per_frame,
                                ConsoleStepStats stats,
                                cudaStream_t stream) {
    constexpr int kThreads = 128;
    const int blocks = static_cast<int>((config.num_envs + kThreads - 1) / kThreads);
    // bank_kind is fixed for the lifetime of a batch, so the specialization is
    // chosen once per launch here instead of per bus access inside the kernel.
    if (buffers.cart.bank_kind == kBankingNone) {
        console_step_kernel<kBankModeNrom><<<blocks, kThreads, 0, stream>>>(
            buffers,
            config.num_envs,
            config.frameskip,
            max_instructions_per_frame,
            stats);
    } else if (buffers.cart.bank_kind >= kBankingFirstPaged) {
        console_step_kernel<kBankModePaged><<<blocks, kThreads, 0, stream>>>(
            buffers,
            config.num_envs,
            config.frameskip,
            max_instructions_per_frame,
            stats);
    } else {
        console_step_kernel<kBankModeBanked><<<blocks, kThreads, 0, stream>>>(
            buffers,
            config.num_envs,
            config.frameskip,
            max_instructions_per_frame,
            stats);
    }
}

void launch_render_kernel(const BatchBuffers& buffers, StepConfig config, cudaStream_t stream) {
    constexpr int kThreads = 256;
    // Small batches (recording, eval, pixel obs on few envs): block-per-env,
    // threads cooperate on lines. Large batches: the env count already
    // saturates the GPU and the serial per-env paint does less total work.
    constexpr std::uint32_t kPerPixelMaxEnvs = 512;
    if (config.num_envs <= kPerPixelMaxEnvs) {
        const int blocks = static_cast<int>(config.num_envs);
        render_rgb_kernel<<<blocks, kThreads, 0, stream>>>(buffers, config.num_envs);
    } else {
        const int blocks = static_cast<int>((config.num_envs + kThreads - 1) / kThreads);
        render_rgb_kernel_serial<<<blocks, kThreads, 0, stream>>>(buffers, config.num_envs);
    }
}

namespace {

__global__ void reset_console_envs_kernel(BatchBuffers buffers,
                                          const std::uint8_t* mask,
                                          std::uint32_t num_envs) {
    const auto env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env < num_envs && mask[env] != 0) {
        cold_reset_console_env(buffers, env);
    }
}

__global__ void seed_frame_log_kernel(BatchBuffers buffers, std::uint32_t num_envs) {
    const auto env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env < num_envs) {
        frame_log_reset(buffers, env);
        if (buffers.ppu.sprite0_dot != nullptr) {
            buffers.ppu.sprite0_dot[env] = 0;
        }
    }
}

__global__ void reset_synthetic_envs_kernel(BatchBuffers buffers,
                                            const std::uint8_t* mask,
                                            std::uint32_t num_envs) {
    const auto env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env < num_envs && mask[env] != 0) {
        cold_reset_synthetic_env(buffers, env);
    }
}

__global__ void reset_console_envs_from_snapshot_kernel(BatchBuffers buffers,
                                                        SnapshotTemplate snapshot,
                                                        const std::uint8_t* mask,
                                                        std::uint32_t num_envs) {
    const auto env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env < num_envs && mask[env] != 0) {
        warm_reset_console_env(buffers, env, snapshot);
    }
}

}  // namespace

void launch_reset_envs_kernel(const BatchBuffers& buffers,
                              const std::uint8_t* device_mask,
                              std::uint32_t num_envs,
                              bool console_mode,
                              cudaStream_t stream) {
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((num_envs + kThreads - 1) / kThreads);
    if (console_mode) {
        reset_console_envs_kernel<<<blocks, kThreads, 0, stream>>>(
            buffers, device_mask, num_envs);
    } else {
        reset_synthetic_envs_kernel<<<blocks, kThreads, 0, stream>>>(
            buffers, device_mask, num_envs);
    }
}

void launch_seed_frame_log_kernel(const BatchBuffers& buffers, std::uint32_t num_envs, cudaStream_t stream) {
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((num_envs + kThreads - 1) / kThreads);
    seed_frame_log_kernel<<<blocks, kThreads, 0, stream>>>(buffers, num_envs);
}

void launch_snapshot_reset_envs_kernel(const BatchBuffers& buffers,
                                       const SnapshotTemplate& snapshot,
                                       const std::uint8_t* device_mask,
                                       std::uint32_t num_envs,
                                       cudaStream_t stream) {
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((num_envs + kThreads - 1) / kThreads);
    reset_console_envs_from_snapshot_kernel<<<blocks, kThreads, 0, stream>>>(
        buffers, snapshot, device_mask, num_envs);
}

}  // namespace nesle::cuda

#endif
