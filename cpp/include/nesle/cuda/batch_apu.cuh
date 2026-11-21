#pragma once

#include <cstdint>

#include "nesle/cuda/state.cuh"

#ifdef __CUDACC__
#define NESLE_CUDA_APU_HD __host__ __device__
#else
#define NESLE_CUDA_APU_HD
#endif

// The APU's sample channel (DMC), as far as it changes how a game runs:
//
// - $4015 bit 4 reads 1 while sample bytes are left to fetch. Tecmo Bowl polls
//   it and only starts a new voice/crowd sample when the last one is over.
// - Every sample byte fetch takes the bus for 4 CPU cycles (FCEUmm's DMCDMA:
//   four X6502_DMR reads). Looping crowd samples at the fastest rate steal
//   ~276 cycles a frame, enough to move which frames lag.
//
// The output timer runs from power-on whether or not a sample plays; a byte
// leaves the buffer on its 8th bit clock and the next fetch follows at once.
// The fetched value only feeds the audio, so it is not read. The tone
// channels, the frame counter and the DMC/frame IRQs are not modelled: none of
// the supported games takes an APU IRQ under FCEUmm, and their length counters
// ($4015 bits 0-3) are only read by games that poll them.

namespace nesle::cuda {

constexpr std::uint64_t kDmcIdle = ~0ULL;
constexpr std::uint32_t kDmcFetchCycles = 4;

// NTSC CPU cycles per output bit for $4010 bits 0-3.
NESLE_CUDA_APU_HD inline std::uint32_t dmc_bit_period(std::uint8_t rate) {
    switch (rate & 0x0F) {
        case 0x0: return 428;
        case 0x1: return 380;
        case 0x2: return 340;
        case 0x3: return 320;
        case 0x4: return 286;
        case 0x5: return 254;
        case 0x6: return 226;
        case 0x7: return 214;
        case 0x8: return 190;
        case 0x9: return 160;
        case 0xA: return 142;
        case 0xB: return 128;
        case 0xC: return 106;
        case 0xD: return 84;
        case 0xE: return 72;
        default: return 54;
    }
}

NESLE_CUDA_APU_HD inline std::uint16_t dmc_sample_bytes(std::uint8_t length) {
    return static_cast<std::uint16_t>(length * 16u + 1u);
}

NESLE_CUDA_APU_HD inline void update_dmc_event(BatchBuffers& buffers, std::uint32_t env) {
    const auto& dmc = buffers.apu.dmc[env];
    buffers.apu.dmc_event[env] = dmc.buffer_full != 0 ? dmc.byte_end : kDmcIdle;
}

// Power-on state at CPU cycle `now`: silent, rate 0, timer one cycle from its
// first bit clock (FCEUmm: DMCacc = 1, DMCBitCount = 0).
NESLE_CUDA_APU_HD inline void reset_dmc_env(BatchBuffers& buffers, std::uint32_t env, std::uint64_t now) {
    if (buffers.apu.dmc == nullptr) {
        return;
    }
    auto& dmc = buffers.apu.dmc[env];
    dmc = DmcState{};
    dmc.byte_end = now + 1u + 7u * dmc_bit_period(0);
    buffers.apu.dmc_event[env] = kDmcIdle;
}

// Brings the DMC up to CPU cycle `now` (the end of the instruction just run).
// Returns the cycles its fetches took from the CPU, or 0 with steal == false:
// in a fast-forwarded JMP-to-self loop the fetch replaces loop cycles instead
// of delaying the next instruction.
NESLE_CUDA_APU_HD inline std::uint32_t service_dmc_env(BatchBuffers& buffers,
                                                       std::uint32_t env,
                                                       std::uint64_t now,
                                                       bool steal) {
    auto& dmc = buffers.apu.dmc[env];
    const std::uint64_t byte_cycles = 8ULL * dmc_bit_period(dmc.rate);
    auto skip_silent_bytes = [&] {
        if (dmc.byte_end <= now) {
            dmc.byte_end += ((now - dmc.byte_end) / byte_cycles + 1u) * byte_cycles;
        }
    };
    std::uint32_t stolen = 0;
    if (dmc.buffer_full == 0) {
        skip_silent_bytes();
    }
    for (;;) {
        if (dmc.buffer_full != 0) {
            if (dmc.byte_end > now) {
                break;
            }
            dmc.buffer_full = 0;  // the byte moves to the shift register
            dmc.byte_end += byte_cycles;
            if (dmc.bytes_left == 0) {
                skip_silent_bytes();
                break;
            }
        }
        if (dmc.bytes_left == 0) {
            break;
        }
        dmc.buffer_full = 1;
        if (--dmc.bytes_left == 0 && (dmc.control & 0x40) != 0) {
            dmc.bytes_left = dmc_sample_bytes(dmc.length);  // loop flag: restart
        }
        if (steal) {
            stolen += kDmcFetchCycles;
            now += kDmcFetchCycles;
        }
    }
    // A $4010 rate change: the bit clock already counting keeps the old period,
    // the clocks after it use the new one (FCEUmm reloads DMCacc per clock).
    const auto rate = static_cast<std::uint8_t>(dmc.control & 0x0F);
    if (rate != dmc.rate) {
        const std::uint64_t old_period = dmc_bit_period(dmc.rate);
        const std::uint64_t later_clocks = (dmc.byte_end - now - 1u) / old_period;
        dmc.byte_end = dmc.byte_end - later_clocks * old_period + later_clocks * dmc_bit_period(rate);
        dmc.rate = rate;
    }
    update_dmc_event(buffers, env);
    return stolen;
}

NESLE_CUDA_APU_HD inline std::uint8_t read_apu_status(const BatchBuffers& buffers, std::uint32_t env) {
    if (buffers.apu.dmc == nullptr) {
        return 0;
    }
    return buffers.apu.dmc[env].bytes_left != 0 ? 0x10 : 0x00;
}

// $4010, $4013 and $4015. The step after this instruction services the DMC
// (event 0), which applies a rate change and fetches a newly started sample's
// first byte - where FCEUmm's next CPU hook does.
NESLE_CUDA_APU_HD inline void write_apu_register(BatchBuffers& buffers,
                                                 std::uint32_t env,
                                                 std::uint16_t address,
                                                 std::uint8_t value) {
    if (buffers.apu.dmc == nullptr) {
        return;
    }
    auto& dmc = buffers.apu.dmc[env];
    if (address == 0x4010) {
        dmc.control = value;
    } else if (address == 0x4013) {
        dmc.length = value;
    } else if ((value & 0x10) == 0) {  // $4015
        dmc.bytes_left = 0;
    } else if (dmc.bytes_left == 0) {
        dmc.bytes_left = dmc_sample_bytes(dmc.length);
    }
    buffers.apu.dmc_event[env] = 0;
}

}  // namespace nesle::cuda

#undef NESLE_CUDA_APU_HD
