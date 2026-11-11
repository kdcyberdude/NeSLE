#pragma once

#include <cstdint>
#include <cstring>

#ifdef __CUDACC__
#define NESLE_CUDA_STATE_HD __host__ __device__
#else
#define NESLE_CUDA_STATE_HD
#endif

// Portable restrict qualifier for the SoA member pointers below. Every array
// behind a live BatchBuffers is its own allocation (distinct cudaMalloc calls
// in the binding, distinct std::vector/std::array storage in the host tests),
// so no two members of one struct instance ever alias - telling the compiler
// so lets it keep loaded values in registers across stores through sibling
// pointers.
//
// Everywhere state is mutated (stepping, resets, snapshot capture/restore,
// rendering into frames_rgb) only one BatchBuffers instance is live and its
// members point at pairwise-distinct allocations.
#if defined(__CUDACC__)
#define NESLE_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define NESLE_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define NESLE_RESTRICT __restrict__
#else
#define NESLE_RESTRICT
#endif

namespace nesle::cuda {

// Copy `n` bytes from `src` to `dst` (non-overlapping). On the CUDA device
// trajectory, uses 16-byte vector chunks when `n` and both pointers are
// 16-byte aligned (all per-env blocks are: each array is its own cudaMalloc
// allocation and the per-env strides - 2048/8192/256/32 - are multiples of
// 16); otherwise falls back to a byte loop. Host builds (including the C++
// unit tests, which compile these headers with a plain host compiler) use
// std::memcpy, which imposes no alignment requirement.
NESLE_CUDA_STATE_HD inline void copy_bytes_fast(std::uint8_t* dst,
                                                const std::uint8_t* src,
                                                std::uint32_t n) {
#if defined(__CUDA_ARCH__)
    if ((n % 16u) == 0u &&
        (reinterpret_cast<std::uintptr_t>(dst) % 16u) == 0u &&
        (reinterpret_cast<std::uintptr_t>(src) % 16u) == 0u) {
        auto* d = reinterpret_cast<uint4*>(dst);
        const auto* s = reinterpret_cast<const uint4*>(src);
        const std::uint32_t chunks = n / 16u;
        for (std::uint32_t i = 0; i < chunks; ++i) {
            d[i] = s[i];
        }
        return;
    }
    for (std::uint32_t i = 0; i < n; ++i) {
        dst[i] = src[i];
    }
#else
    std::memcpy(dst, src, n);
#endif
}

// Zero `n` bytes at `dst`. Same alignment strategy as copy_bytes_fast.
NESLE_CUDA_STATE_HD inline void zero_bytes_fast(std::uint8_t* dst, std::uint32_t n) {
#if defined(__CUDA_ARCH__)
    if ((n % 16u) == 0u && (reinterpret_cast<std::uintptr_t>(dst) % 16u) == 0u) {
        auto* d = reinterpret_cast<uint4*>(dst);
        const std::uint32_t chunks = n / 16u;
        for (std::uint32_t i = 0; i < chunks; ++i) {
            d[i] = make_uint4(0u, 0u, 0u, 0u);
        }
        return;
    }
    for (std::uint32_t i = 0; i < n; ++i) {
        dst[i] = 0;
    }
#else
    std::memset(dst, 0, n);
#endif
}

constexpr int kCpuRamBytes = 2048;
constexpr int kPrgRamBytes = 8 * 1024;
constexpr int kChrRamBytes = 8 * 1024;
constexpr int kPaletteRamBytes = 32;
constexpr int kOamBytes = 256;
constexpr int kNametableRamBytes = 2048;
constexpr int kFrameWidth = 256;
constexpr int kFrameHeight = 240;
constexpr int kRgbChannels = 3;
constexpr std::uint8_t kNametableVertical = 0;
constexpr std::uint8_t kNametableHorizontal = 1;
constexpr std::uint8_t kNametableFourScreen = 2;
// Paged boards keep their run-time mirroring as a page map: bit n is the 1 KB
// CIRAM page (0 = $000, 1 = $400) behind logical nametable n ($2000 + n * $400).
constexpr std::uint8_t kPageMapVertical = 0x0A;    // 0 1 0 1
constexpr std::uint8_t kPageMapHorizontal = 0x0C;  // 0 0 1 1
constexpr std::uint8_t kPageMapSingleLow = 0x00;   // 0 0 0 0
constexpr std::uint8_t kPageMapSingleHigh = 0x0F;  // 1 1 1 1

// Paged boards (see MapperStateSoA): per-env page tables and board registers.
constexpr int kPrgPages = 4;          // 8 KB CPU pages at $8000, $A000, $C000, $E000
constexpr int kChrPages = 8;          // 1 KB PPU pages at $0000-$1FFF
constexpr int kMapperRegBytes = 16;   // board registers, layout per board in mapper.cuh

// How the $8000-$FFFF window is split. Resolved once on the host from the iNES
// header (mapper number + submapper + CHR presence) so the device code never
// has to interpret iNES fields, and every banked board reduces to the same
// "switchable window at the bottom, fixed window on top" shape.
enum BankKind : std::uint8_t {
    // NROM: no register, no indirection, 16/32 KB image fixed at $8000.
    kBankingNone = 0,
    // UxROM family: 16 KB window at $8000-$BFFF, last 16 KB page fixed at $C000.
    kBankingUxrom16k,
    // BNROM: single 32 KB window at $8000-$FFFF.
    kBankingBnrom32k,
    // Paged boards, from here on: the bus reads PRG through a per-env 8 KB page
    // table and CHR through a per-env 1 KB page table, mirroring is a per-env
    // register, and the board may raise IRQs. Everything >= kBankingFirstPaged.
    kBankingCnrom,  // mapper 3: 32 KB PRG fixed, 8 KB CHR bank latch
    kBankingMmc1,   // mapper 1: serial shift register, PRG/CHR modes, mirroring
    kBankingMmc2,   // mapper 9: 8 KB PRG bank, 4 KB CHR banks picked by $FD/$FE latches
    kBankingMmc3,   // mapper 4: 8 KB PRG / 1-2 KB CHR banks, scanline IRQ counter
    kBankingFirstPaged = kBankingCnrom,
};

struct CpuStateSoA {
    std::uint16_t* NESLE_RESTRICT pc;
    std::uint8_t* NESLE_RESTRICT a;
    std::uint8_t* NESLE_RESTRICT x;
    std::uint8_t* NESLE_RESTRICT y;
    std::uint8_t* NESLE_RESTRICT sp;
    std::uint8_t* NESLE_RESTRICT p;
    std::uint64_t* NESLE_RESTRICT cycles;
    std::uint8_t* NESLE_RESTRICT nmi_pending;
    std::uint8_t* NESLE_RESTRICT irq_pending;
    std::uint8_t* NESLE_RESTRICT ram;
    std::uint8_t* NESLE_RESTRICT prg_ram;
    std::uint8_t* NESLE_RESTRICT controller1_shift;
    std::uint8_t* NESLE_RESTRICT controller1_shift_count;
    std::uint8_t* NESLE_RESTRICT controller1_strobe;
    std::uint32_t* NESLE_RESTRICT pending_dma_cycles;
    // 0 while the env is healthy. Once its CPU reaches an opcode the core does not
    // implement, this holds (pc << 8) | opcode and the env is quarantined: the step
    // kernel skips it and marks it done, so one bad environment cannot end the launch
    // the way asm("trap;") did. A reset clears it, because the env is genuinely healthy
    // again once it restarts from the reset vector.
    std::uint32_t* NESLE_RESTRICT fault;
    // Same encoding, but a one-shot report rather than execution state. A reset does not
    // clear this, so the fault survives the automatic reset that step_device performs on
    // a done env - without which a training loop would reset the env inside the same
    // call that faulted it and faults() would always come back empty. Reading it through
    // CudaBatch.faults() clears it, so each fault is reported exactly once.
    std::uint32_t* NESLE_RESTRICT fault_pending;
};

struct PpuStateSoA {
    std::uint8_t* NESLE_RESTRICT ctrl;
    std::uint8_t* NESLE_RESTRICT mask;
    std::uint8_t* NESLE_RESTRICT status;
    std::uint8_t* NESLE_RESTRICT oam_addr;
    std::uint8_t* NESLE_RESTRICT nmi_pending;
    // Merged frame position: scanline * kPpuDotsPerScanline + dot. Stored as
    // one word so the per-launch hot-state load/store is a single 32-bit
    // access instead of two dependent 16-bit ones (PpuHotState composes the
    // same value in registers).
    std::uint32_t* NESLE_RESTRICT frame_dot;
    std::uint64_t* NESLE_RESTRICT frame;
    std::uint16_t* NESLE_RESTRICT v;
    std::uint16_t* NESLE_RESTRICT t;
    std::uint8_t* NESLE_RESTRICT x;
    std::uint8_t* NESLE_RESTRICT w;
    std::uint8_t* NESLE_RESTRICT open_bus;
    std::uint8_t* NESLE_RESTRICT read_buffer;
    std::uint8_t* NESLE_RESTRICT scroll_x;
    std::uint8_t* NESLE_RESTRICT scroll_y;
    std::uint8_t* NESLE_RESTRICT nametable_ram;
    std::uint8_t* NESLE_RESTRICT palette_ram;
    std::uint8_t* NESLE_RESTRICT oam;
    // Frame dot of this frame's next sprite-0 event; 0 = none. With a frame log, bit 31
    // marks the point (start of sprite 0's first line) where the hit is worked out from
    // the log, and without it the hit itself. Without a frame log, paged boards keep the
    // OAM-only hit computed at the pre-render line and the others the coarse line-30 hit.
    std::uint32_t* NESLE_RESTRICT sprite0_dot;
    // Cartridge CHR RAM (8 KB per env). nullptr for carts that carry CHR ROM,
    // which is every ROM that worked before UxROM support existed; the PPU
    // pattern fetch then never touches this array.
    std::uint8_t* NESLE_RESTRICT chr_ram;
    // Presentation: kFrameLogWords per env (batch_frame.cuh), two banks of frame-start
    // PPU state plus the PPU-visible writes made while the frame was drawn. The renderer
    // replays the bank frozen at the last vblank (log_bank bit 0); the step kernel writes
    // the other one. log_count is the writing bank's event count. All nullptr => render
    // falls back to live state with no mid-frame writes (host tests, older callers).
    std::uint32_t* NESLE_RESTRICT frame_log;
    std::uint16_t* NESLE_RESTRICT log_count;
    std::uint8_t* NESLE_RESTRICT log_bank;
    // Nametable RAM and OAM frozen at each vblank, for the renderer.
    std::uint8_t* NESLE_RESTRICT snap_oam;        // kOamBytes per env
    std::uint8_t* NESLE_RESTRICT snap_nametable;  // kNametableRamBytes per env
};

// Read-only description of the cartridge, shared by every environment in the
// batch. Everything a mapper can change at runtime lives in MapperStateSoA
// instead: bank registers are per-env mutable state, while the window geometry
// below is a property of the ROM image and never changes.
struct CartridgeView {
    const std::uint8_t* prg_rom;
    const std::uint8_t* chr_rom;
    std::uint32_t prg_rom_size;        // padded size, the power of two the device holds
    // Where $FFFC/$FFFD live in the image at power-on. NROM keeps the whole image
    // at $8000 and UxROM keeps its fixed last 16 KB at $C000, so on both the
    // vector is the last four bytes of the image. BNROM maps bank 0's 32 KB at
    // $8000 at power-on, so its vector is the last four bytes of that window -
    // the first 32 KB of the image - and not of the image.
    std::uint32_t prg_reset_vector_base;
    std::uint32_t prg_rom_mask;  // prg_rom_size - 1; PRG is padded to 2^n
    std::uint32_t chr_rom_size;
    std::uint16_t mapper;
    std::uint8_t bank_kind;      // BankKind
    std::uint8_t bus_conflicts;  // AND the written byte with the mask ROM
    // Switchable window. It occupies [prg_window_start, prg_window_start +
    // window_bytes); a read inside it maps to
    // ((prg_bank & prg_bank_mask) << prg_window_shift) + (address - prg_window_start).
    // The window sits at the bottom of CPU space ($8000) for every board here,
    // so the unsigned distance makes one compare cover both halves.
    std::uint32_t prg_window_start;
    std::uint32_t prg_window_shift;
    std::uint32_t prg_window_mask;  // window_bytes - 1
    std::uint8_t prg_bank_mask;
    std::uint8_t prg_bank_shift;  // register right-shift before masking
    // Fixed window, the complement of the switchable one: maps to
    // prg_fixed_base + (address & prg_fixed_mask). Unused for BNROM, whose
    // single window covers the whole CPU space.
    std::uint32_t prg_fixed_base;
    std::uint32_t prg_fixed_mask;
    std::uint8_t nametable_arrangement;  // power-on mirroring, from the header
    // Paged boards only: number of 8 KB PRG pages - 1 and 1 KB CHR pages - 1
    // (CHR ROM, or the 8 KB of CHR RAM). Both page counts are powers of two.
    std::uint16_t prg_page_mask;
    std::uint16_t chr_page_mask;
};

// Per-environment mapper registers. One byte per env keeps the batch layout
// flat and lets the step kernel's per-instruction read be a single 8-bit load.
// Absent (nullptr) for NROM, which keeps its zero-indirection read path.
struct MapperStateSoA {
    std::uint8_t* NESLE_RESTRICT prg_bank;  // switchable PRG window
    // Paged boards (bank_kind >= kBankingFirstPaged) only; nullptr otherwise.
    // The page tables are derived from `regs` on every register write, so a
    // PRG/CHR fetch is one table load plus the ROM load, whatever the board.
    std::uint32_t* NESLE_RESTRICT prg_page;  // kPrgPages per env: byte offset into PRG ROM
    std::uint32_t* NESLE_RESTRICT chr_page;  // kChrPages per env: byte offset into CHR ROM/RAM
    std::uint8_t* NESLE_RESTRICT mirroring;  // per-env nametable page map (kPageMap*)
    std::uint8_t* NESLE_RESTRICT regs;       // kMapperRegBytes per env
};

// The PPU fields read/modified on every emulated instruction. The step kernel
// loads them into registers once per launch (like CpuState already is) and
// stores them back at exit - without this they cost ~6-10 dependent global
// round-trips per instruction. Everything else (scroll, v/t/x/w, memories,
// presentation snapshot) stays in global memory: those are touched per
// register-access or per frame, not per instruction. Load/store helpers live
// in batch_ppu.cuh next to the timing constants.
struct PpuHotState {
    std::uint32_t frame_dot = 0;  // scanline * dots-per-scanline + dot
    std::uint64_t frame = 0;
    std::uint8_t status = 0;
    std::uint8_t ctrl = 0;
    std::uint8_t mask = 0;
    std::uint8_t nmi_pending = 0;
    // Frame log (batch_frame.cuh): frame dot the current instruction started at, the
    // writing bank's event count, and the presented bank.
    std::uint32_t now = 0;
    std::uint16_t log_n = 0;
    std::uint8_t log_bank = 0;
};

// The APU's sample channel (DMC), reduced to what changes the game: its
// status bit ($4015 bit 4, "sample bytes left") and the 4 CPU cycles each
// sample byte fetch steals, as FCEUmm's DMCDMA does. Tone channels, the frame
// counter and the IRQ flags are not modelled. Times are absolute CPU cycles
// (CpuState::cycles). See batch_apu.cuh.
struct DmcState {
    std::uint64_t byte_end;   // cycle of the timer's next 8th bit clock, when the
                              // sample buffer empties (the timer always runs)
    std::uint16_t bytes_left; // sample bytes still to fetch ($4015 bit 4)
    std::uint8_t control;     // $4010: bit 6 loop, bits 0-3 rate
    std::uint8_t length;      // $4013: sample length = length * 16 + 1 bytes
    std::uint8_t rate;        // rate index byte_end was scheduled with
    std::uint8_t buffer_full; // a fetched byte waits for the shift register
};

// Both nullptr = no APU state (host tests, older callers): $4015 reads 0 and
// no cycles are stolen, the behaviour before the DMC was modelled.
struct ApuStateSoA {
    // Cycle at or after which the DMC needs service: 0 = fetch due,
    // byte_end while a byte is buffered, ~0 when idle. Kept apart from `dmc`
    // because the step kernel tests it after every instruction.
    std::uint64_t* NESLE_RESTRICT dmc_event;
    DmcState* NESLE_RESTRICT dmc;
};

struct BatchBuffers {
    CpuStateSoA cpu;
    PpuStateSoA ppu;
    MapperStateSoA mapper;
    CartridgeView cart;
    std::uint8_t* NESLE_RESTRICT action_masks;
    std::uint8_t* NESLE_RESTRICT done;
    float* NESLE_RESTRICT rewards;
    int* NESLE_RESTRICT previous_mario_x;
    int* NESLE_RESTRICT previous_mario_time;
    std::uint8_t* NESLE_RESTRICT frames_rgb;
    ApuStateSoA apu;
    // 1: do not resolve MMC2's latches at vblank; the caller resolves them from the frozen frame log before the next
    // instruction runs (rl/ runs that replay in its render kernel, where it is ~4x cheaper). 0 = resolve here.
    std::uint8_t defer_mmc2_resolve = 0;
};

// Read-only "bank" of N snapshot templates used by snapshot-based env resets. Each top-level
// array is num_levels copies of a single snapshot's array data, laid out contiguously.
// `env_to_level[env]` selects which slot each env restores from - letting different envs in
// the same batch start at different levels (curriculum training). For the single-level case
// num_levels is 1 and env_to_level is all zeros.
struct SnapshotTemplate {
    // Per-level array buffers (concatenated, length = num_levels * <kind>_Bytes).
    const std::uint8_t* cpu_ram = nullptr;
    const std::uint8_t* prg_ram = nullptr;
    const std::uint8_t* nametable_ram = nullptr;
    const std::uint8_t* palette_ram = nullptr;
    const std::uint8_t* oam = nullptr;

    // Per-level scalar fields (length = num_levels).
    const std::uint16_t* pc = nullptr;
    const std::uint8_t* a = nullptr;
    const std::uint8_t* x = nullptr;
    const std::uint8_t* y = nullptr;
    const std::uint8_t* sp = nullptr;
    const std::uint8_t* p = nullptr;
    const std::uint64_t* cycles = nullptr;
    const std::uint8_t* ppu_ctrl = nullptr;
    const std::uint8_t* ppu_mask = nullptr;
    const std::uint8_t* ppu_status = nullptr;
    const std::uint8_t* ppu_oam_addr = nullptr;
    const std::uint8_t* ppu_open_bus = nullptr;
    const std::uint8_t* ppu_read_buffer = nullptr;
    const std::uint8_t* ppu_x = nullptr;
    const std::uint8_t* ppu_w = nullptr;
    const std::uint16_t* ppu_v = nullptr;
    const std::uint16_t* ppu_t = nullptr;

    // Per-env level assignment (length = num_envs).
    const std::uint8_t* env_to_level = nullptr;

    // Cartridge section. chr_ram is per-level (length = num_levels * kChrRamBytes)
    // and null on CHR ROM boards. has_cart_latch/cart_latch carry the board's
    // bank register: FCEUX saves it as "LATC" (mappers 2/94) or "REGS"[0]
    // (mapper 34), and without it a restored env runs on the wrong 16/32 KB bank.
    const std::uint8_t* chr_ram = nullptr;
    const std::uint8_t* cart_latch = nullptr;
    // Paged boards (CNROM/MMC1/MMC2/MMC3): kMapperRegBytes of board registers per
    // level in MapperStateSoA::regs layout (fcs::board_registers), and the board's
    // IRQ line. Null on every other board.
    const std::uint8_t* mapper_regs = nullptr;
    const std::uint8_t* irq_line = nullptr;

    std::uint32_t num_levels = 0;
};

NESLE_CUDA_STATE_HD inline const std::uint8_t* env_cpu_ram(const BatchBuffers& buffers,
                                                           std::uint32_t env) {
    return buffers.cpu.ram + static_cast<std::uint64_t>(env) * kCpuRamBytes;
}

NESLE_CUDA_STATE_HD inline std::uint8_t* env_cpu_ram(BatchBuffers& buffers, std::uint32_t env) {
    return buffers.cpu.ram + static_cast<std::uint64_t>(env) * kCpuRamBytes;
}

NESLE_CUDA_STATE_HD inline const std::uint8_t* env_oam(const BatchBuffers& buffers,
                                                       std::uint32_t env) {
    return buffers.ppu.oam + static_cast<std::uint64_t>(env) * kOamBytes;
}

NESLE_CUDA_STATE_HD inline std::uint8_t* env_oam(BatchBuffers& buffers, std::uint32_t env) {
    return buffers.ppu.oam + static_cast<std::uint64_t>(env) * kOamBytes;
}

}  // namespace nesle::cuda

#undef NESLE_CUDA_STATE_HD
#undef NESLE_RESTRICT
