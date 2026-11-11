#pragma once

#include <cstdint>

#include "nesle/cuda/batch_frame.cuh"
#include "nesle/cuda/mapper.cuh"
#include "nesle/cuda/state.cuh"

#ifdef __CUDACC__
#define NESLE_CUDA_HD __host__ __device__
#else
#define NESLE_CUDA_HD
#endif

namespace nesle::cuda {

constexpr std::uint16_t kPpuDotsPerScanline = 341;
constexpr std::uint16_t kPpuScanlinesPerFrame = 262;
constexpr std::int16_t kPpuVblankStartScanline = 241;
constexpr std::int16_t kPpuPreRenderScanline = 261;
constexpr std::uint16_t kPpuVblankFlagDot = 1;
constexpr std::int16_t kPpuCoarseSpriteZeroHitScanline = 30;
constexpr std::uint16_t kPpuCoarseSpriteZeroHitDot = 1;

struct BatchPpuStepResult {
    std::uint32_t cycles = 0;
    std::uint32_t frames_completed = 0;
    bool nmi_started = false;
};

NESLE_CUDA_HD inline PpuHotState load_ppu_hot_state(const BatchBuffers& buffers,
                                                    std::uint32_t env) {
    PpuHotState hot;
    hot.frame_dot = buffers.ppu.frame_dot[env];
    hot.frame = buffers.ppu.frame[env];
    hot.status = buffers.ppu.status[env];
    hot.ctrl = buffers.ppu.ctrl[env];
    hot.mask = buffers.ppu.mask[env];
    hot.nmi_pending = buffers.ppu.nmi_pending[env];
    if (buffers.ppu.frame_log != nullptr) {
        hot.log_n = buffers.ppu.log_count[env];
        hot.log_bank = buffers.ppu.log_bank[env];
    }
    return hot;
}

NESLE_CUDA_HD inline void store_ppu_hot_state(BatchBuffers& buffers,
                                              std::uint32_t env,
                                              const PpuHotState& hot) {
    buffers.ppu.frame_dot[env] = hot.frame_dot;
    buffers.ppu.frame[env] = hot.frame;
    buffers.ppu.status[env] = hot.status;
    buffers.ppu.ctrl[env] = hot.ctrl;
    buffers.ppu.mask[env] = hot.mask;
    buffers.ppu.nmi_pending[env] = hot.nmi_pending;
    if (buffers.ppu.frame_log != nullptr) {
        buffers.ppu.log_count[env] = hot.log_n;
        buffers.ppu.log_bank[env] = hot.log_bank;
    }
}

NESLE_CUDA_HD inline bool batch_ppu_nmi_enabled(const BatchBuffers& buffers,
                                                std::uint32_t env) {
    return (buffers.ppu.ctrl[env] & 0x80) != 0;
}

NESLE_CUDA_HD inline bool batch_ppu_rendering_enabled(const BatchBuffers& buffers,
                                                      std::uint32_t env) {
    return (buffers.ppu.mask[env] & 0x18) != 0;
}

NESLE_CUDA_HD inline void batch_ppu_set_vblank_hot(PpuHotState& hot, bool enabled) {
    const auto was_in_vblank = static_cast<std::uint8_t>(hot.status & 0x80);
    if (enabled) {
        hot.status = static_cast<std::uint8_t>(hot.status | 0x80);
        if (was_in_vblank == 0 && (hot.ctrl & 0x80) != 0) {
            hot.nmi_pending = 1;
        }
        return;
    }
    hot.status = static_cast<std::uint8_t>(hot.status & 0x7F);
    hot.nmi_pending = 0;
}

NESLE_CUDA_HD inline std::uint32_t batch_ppu_frame_dot(const BatchBuffers& buffers,
                                                       std::uint32_t env) {
    return buffers.ppu.frame_dot[env];
}

NESLE_CUDA_HD inline void batch_ppu_set_vblank(BatchBuffers& buffers,
                                               std::uint32_t env,
                                               bool enabled) {
    const auto was_in_vblank = static_cast<std::uint8_t>(buffers.ppu.status[env] & 0x80);
    if (enabled) {
        buffers.ppu.status[env] = static_cast<std::uint8_t>(buffers.ppu.status[env] | 0x80);
        if (was_in_vblank == 0 && batch_ppu_nmi_enabled(buffers, env)) {
            buffers.ppu.nmi_pending[env] = 1;
        }
        return;
    }

    buffers.ppu.status[env] = static_cast<std::uint8_t>(buffers.ppu.status[env] & 0x7F);
    buffers.ppu.nmi_pending[env] = 0;
}

NESLE_CUDA_HD inline std::uint32_t batch_ppu_cycles_until_next_timing_event(
    const BatchBuffers& buffers,
    std::uint32_t env) {
    constexpr std::uint32_t kFrameDots =
        static_cast<std::uint32_t>(kPpuDotsPerScanline) *
        static_cast<std::uint32_t>(kPpuScanlinesPerFrame);
    constexpr std::uint32_t kSpriteZeroHitDot =
        static_cast<std::uint32_t>(kPpuCoarseSpriteZeroHitScanline) *
            static_cast<std::uint32_t>(kPpuDotsPerScanline) +
        static_cast<std::uint32_t>(kPpuCoarseSpriteZeroHitDot);
    constexpr std::uint32_t kVblankDot =
        static_cast<std::uint32_t>(kPpuVblankStartScanline) *
            static_cast<std::uint32_t>(kPpuDotsPerScanline) +
        static_cast<std::uint32_t>(kPpuVblankFlagDot);
    constexpr std::uint32_t kPreRenderDot =
        static_cast<std::uint32_t>(kPpuPreRenderScanline) *
            static_cast<std::uint32_t>(kPpuDotsPerScanline) +
        static_cast<std::uint32_t>(kPpuVblankFlagDot);

    const auto current = batch_ppu_frame_dot(buffers, env);
    auto distance = [current](std::uint32_t event_dot) {
        return event_dot > current ? event_dot - current : kFrameDots - current + event_dot;
    };

    auto next = kFrameDots - current;
    const auto vblank = distance(kVblankDot);
    if (vblank < next) {
        next = vblank;
    }
    const auto prerender = distance(kPreRenderDot);
    if (prerender < next) {
        next = prerender;
    }
    if (batch_ppu_rendering_enabled(buffers, env)) {
        const auto sprite_zero = distance(kSpriteZeroHitDot);
        if (sprite_zero < next) {
            next = sprite_zero;
        }
    }
    return next;
}

NESLE_CUDA_HD inline std::uint32_t batch_ppu_cycles_until_next_timing_event_hot(
    const PpuHotState& hot) {
    constexpr std::uint32_t kFrameDots =
        static_cast<std::uint32_t>(kPpuDotsPerScanline) *
        static_cast<std::uint32_t>(kPpuScanlinesPerFrame);
    constexpr std::uint32_t kSpriteZeroHitDot =
        static_cast<std::uint32_t>(kPpuCoarseSpriteZeroHitScanline) *
            static_cast<std::uint32_t>(kPpuDotsPerScanline) +
        static_cast<std::uint32_t>(kPpuCoarseSpriteZeroHitDot);
    constexpr std::uint32_t kVblankDot =
        static_cast<std::uint32_t>(kPpuVblankStartScanline) *
            static_cast<std::uint32_t>(kPpuDotsPerScanline) +
        static_cast<std::uint32_t>(kPpuVblankFlagDot);
    constexpr std::uint32_t kPreRenderDot =
        static_cast<std::uint32_t>(kPpuPreRenderScanline) *
            static_cast<std::uint32_t>(kPpuDotsPerScanline) +
        static_cast<std::uint32_t>(kPpuVblankFlagDot);

    const auto current = hot.frame_dot;
    auto distance = [current](std::uint32_t event_dot) {
        return event_dot > current ? event_dot - current : kFrameDots - current + event_dot;
    };

    auto next = kFrameDots - current;
    const auto vblank = distance(kVblankDot);
    if (vblank < next) {
        next = vblank;
    }
    const auto prerender = distance(kPreRenderDot);
    if (prerender < next) {
        next = prerender;
    }
    if ((hot.mask & 0x18) != 0) {
        const auto sprite_zero = distance(kSpriteZeroHitDot);
        if (sprite_zero < next) {
            next = sprite_zero;
        }
    }
    return next;
}

// ------------------------------------------------- MMC3 scanline counter
//
// FCEUmm clocks the MMC3 counter once per scanline while background or sprites
// are on, unless both pattern tables sit at $1000 with 8x8 sprites
// ((ctrl & 0x38) == 0x18): at dot 266 of each visible line (DoLine: 256 + 6 + 4)
// and at dot 256 of the pre-render line. Each clock is a lazy timing event like
// vblank, so the IRQ is raised at the instruction boundary that crosses it.
constexpr std::uint32_t kMmc3VisibleClockDot = 266;
constexpr std::uint32_t kMmc3PreRenderClockDot = 256;

template <BankMode Mode>
NESLE_CUDA_HD inline bool batch_ppu_has_scanline_counter(const BatchBuffers& buffers) {
    if constexpr (Mode == kBankModePaged || Mode == kBankModeRuntime) {
        return buffers.cart.bank_kind == kBankingMmc3 && buffers.mapper.regs != nullptr;
    } else {
        return false;
    }
}

// Dots from frame position `current` to the next counter clock, strictly after it.
NESLE_CUDA_HD inline std::uint32_t batch_ppu_dots_until_scanline_clock(std::uint32_t current) {
    constexpr std::uint32_t kDots = kPpuDotsPerScanline;
    constexpr std::uint32_t kPreRender = static_cast<std::uint32_t>(kPpuPreRenderScanline);
    const std::uint32_t line = current / kDots;
    const std::uint32_t dot = current % kDots;
    if (line < 240u) {
        if (dot < kMmc3VisibleClockDot) {
            return kMmc3VisibleClockDot - dot;
        }
        if (line + 1u < 240u) {
            return kDots - dot + kMmc3VisibleClockDot;
        }
        return (kPreRender - line) * kDots + kMmc3PreRenderClockDot - dot;
    }
    if (line < kPreRender) {
        return (kPreRender - line) * kDots + kMmc3PreRenderClockDot - dot;
    }
    if (dot < kMmc3PreRenderClockDot) {
        return kMmc3PreRenderClockDot - dot;
    }
    return kDots - dot + kMmc3VisibleClockDot;  // line 0 of the next frame
}

// Clocks the counter for every clock dot in (start_dot, end_dot]. The span
// starts inside the current frame and may run into the next ones.
NESLE_CUDA_HD inline void batch_ppu_clock_scanline_counter(BatchBuffers& buffers,
                                                           std::uint32_t env,
                                                           std::uint32_t start_dot,
                                                           std::uint32_t end_dot,
                                                           const PpuHotState& hot) {
    if ((hot.mask & 0x18) == 0 || (hot.ctrl & 0x38) == 0x18) {
        return;
    }
    constexpr std::uint32_t kDots = kPpuDotsPerScanline;
    constexpr std::uint32_t kLines = kPpuScanlinesPerFrame;
    for (std::uint32_t line_abs = start_dot / kDots; line_abs <= end_dot / kDots; ++line_abs) {
        const std::uint32_t line = line_abs % kLines;
        std::uint32_t clock_dot;
        if (line < 240u) {
            clock_dot = kMmc3VisibleClockDot;
        } else if (line == static_cast<std::uint32_t>(kPpuPreRenderScanline)) {
            clock_dot = kMmc3PreRenderClockDot;
        } else {
            continue;
        }
        const std::uint32_t absolute_dot = line_abs * kDots + clock_dot;
        if (absolute_dot > start_dot && absolute_dot <= end_dot) {
            paged::mmc3_clock(buffers, env);
        }
    }
}

// --------------------------------------------------- sprite-0 hit (paged)
//
// The coarse hit at line 30 is SMB's split line, and NROM/window boards keep
// it unchanged. Paged-board games put sprite 0 elsewhere (Punch-Out!! spins on
// $2002 until it is set), so for them the hit is placed where sprite 0's first
// opaque pixel is drawn: line y + 1 + first opaque row, dot x + first opaque
// column + 1. Background opacity is not checked. Worked out once per frame at
// the pre-render line, after the game's vblank OAM DMA.
template <BankMode Mode>
NESLE_CUDA_HD inline bool batch_ppu_has_sprite0_scan(const BatchBuffers& buffers) {
    if constexpr (Mode == kBankModePaged || Mode == kBankModeRuntime) {
        return cart_is_paged<Mode>(buffers.cart) && buffers.ppu.sprite0_dot != nullptr;
    } else {
        return false;
    }
}

// With a frame log every board gets FCEUmm's sprite-0 hit (sprite 0 over opaque
// background), worked out from the log when sprite 0's first line is due.
NESLE_CUDA_HD inline bool batch_ppu_has_sprite0_replay(const BatchBuffers& buffers) {
    return buffers.ppu.frame_log != nullptr && buffers.ppu.sprite0_dot != nullptr;
}

// sprite0_dot with the replay: the frame dot of the next check (bits 0-16) | kSprite0Evaluate,
// and once a check has run, kSprite0Known | its answer (kSprite0Hit) | the log length it
// saw (bits 20-28). With nothing logged since, the next check gives the same answer.
constexpr std::uint32_t kSprite0Evaluate = 0x80000000u;
constexpr std::uint32_t kSprite0Known = 0x40000000u;
constexpr std::uint32_t kSprite0Hit = 0x20000000u;
constexpr std::uint32_t kSprite0DotMask = 0x1FFFFu;

// With the replay, $2002's vblank flag is set, and its vblank and sprite-0 bits are cleared,
// where FCEUmm does it: at the start of its line 241 and of its pre-render line, 12 dots
// before NeSLE's vblank and pre-render dots (batch_frame.cuh kFceummDotOffset + 1). FCEUmm
// raises NMI 12 dots after setting the flag. Tecmo Bowl's $2002 polls depend on this timing.
constexpr std::uint32_t kReplayVblankFlagDot =
    static_cast<std::uint32_t>(kPpuVblankStartScanline) * kPpuDotsPerScanline + kPpuVblankFlagDot - 12u;
constexpr std::uint32_t kReplayStatusClearDot =
    static_cast<std::uint32_t>(kPpuPreRenderScanline) * kPpuDotsPerScanline + kPpuVblankFlagDot - 12u;

// Between the pre-render line and the frame wrap of a frame FCEUmm shortens by a dot.
NESLE_CUDA_HD inline bool batch_ppu_short_frame(const PpuHotState& hot) {
    return (hot.log_bank & kLogShortFrame) != 0;
}

template <BankMode Mode>
NESLE_CUDA_HD inline std::uint32_t batch_ppu_sprite0_hit_dot(const BatchBuffers& buffers,
                                                             std::uint32_t env,
                                                             std::uint8_t ctrl) {
    const std::uint8_t* oam = buffers.ppu.oam + static_cast<std::uint64_t>(env) * kOamBytes;
    const std::uint32_t y = oam[0];
    const std::uint32_t tile = oam[1];
    const std::uint32_t attr = oam[2];
    const std::uint32_t x = oam[3];
    if (y >= 239u) {
        return 0;
    }
    const std::uint32_t height = (ctrl & 0x20) != 0 ? 16u : 8u;
    for (std::uint32_t r = 0; r < height; ++r) {
        const std::uint32_t row = (attr & 0x80u) != 0 ? height - 1u - r : r;
        std::uint32_t address;
        if (height == 16u) {
            address = ((tile & 1u) << 12) + ((tile & 0xFEu) << 4) + ((row & 8u) << 1) + (row & 7u);
        } else {
            address = ((ctrl & 0x08u) << 9) + (tile << 4) + row;
        }
        const auto bits = static_cast<std::uint32_t>(
            read_chr<Mode>(buffers, env, static_cast<std::uint16_t>(address)) |
            read_chr<Mode>(buffers, env, static_cast<std::uint16_t>(address + 8u)));
        if (bits == 0) {
            continue;
        }
        std::uint32_t column = 0;
        if ((attr & 0x40u) != 0) {
            while (((bits >> column) & 1u) == 0) {
                ++column;  // flipped: bit 0 is the leftmost pixel
            }
        } else {
            while (((bits << column) & 0x80u) == 0) {
                ++column;
            }
        }
        const std::uint32_t line = y + 1u + r;
        const std::uint32_t pixel = x + column;
        if (line >= 240u || pixel >= 255u) {
            return 0;
        }
        return line * kPpuDotsPerScanline + pixel + 1u;
    }
    return 0;
}

// The next lazy-settle horizon for this cartridge: the PPU's own events, plus
// the OAM-based sprite-0 hit and the MMC3 counter clocks on paged boards.
template <BankMode Mode = kBankModeRuntime>
NESLE_CUDA_HD inline std::uint32_t batch_ppu_cycles_until_next_event_hot(const BatchBuffers& buffers,
                                                                         std::uint32_t env,
                                                                         const PpuHotState& hot) {
    if (batch_ppu_has_sprite0_replay(buffers)) {
        // Vblank, pre-render, frame wrap, the sprite-0 event, MMC3 clocks.
        constexpr std::uint32_t kFrameDots =
            static_cast<std::uint32_t>(kPpuDotsPerScanline) * static_cast<std::uint32_t>(kPpuScanlinesPerFrame);
        constexpr std::uint32_t kVblankDot =
            static_cast<std::uint32_t>(kPpuVblankStartScanline) * kPpuDotsPerScanline + kPpuVblankFlagDot;
        constexpr std::uint32_t kPreRenderDot =
            static_cast<std::uint32_t>(kPpuPreRenderScanline) * kPpuDotsPerScanline + kPpuVblankFlagDot;
        const auto current = hot.frame_dot;
        // FCEUmm's short frames: the pre-render line loses its last dot (batch_frame.cuh).
        const std::uint32_t skip = batch_ppu_short_frame(hot) ? 1u : 0u;
        const std::uint32_t frame_dots = kFrameDots - skip;
        auto distance = [current, frame_dots](std::uint32_t event_dot) {
            return event_dot > current ? event_dot - current : frame_dots - current + event_dot;
        };
        auto next = frame_dots - current;
        const auto vblank = distance(kVblankDot);
        next = vblank < next ? vblank : next;
        const auto prerender = distance(kPreRenderDot);
        next = prerender < next ? prerender : next;
        const auto vblank_flag = distance(kReplayVblankFlagDot);
        next = vblank_flag < next ? vblank_flag : next;
        const auto status_clear = distance(kReplayStatusClearDot);
        next = status_clear < next ? status_clear : next;
        const auto sprite0 = buffers.ppu.sprite0_dot[env] & kSprite0DotMask;
        if (sprite0 != 0) {
            const auto d = distance(sprite0);
            next = d < next ? d : next;
        }
        if (batch_ppu_has_scanline_counter<Mode>(buffers)) {
            auto clock = batch_ppu_dots_until_scanline_clock(hot.frame_dot);
            if (current + clock >= kFrameDots) {
                clock -= skip;
            }
            next = clock < next ? clock : next;
        }
        return next;
    }
    auto next = batch_ppu_cycles_until_next_timing_event_hot(hot);
    if (batch_ppu_has_sprite0_scan<Mode>(buffers) && (hot.mask & 0x18) == 0x18) {
        const auto hit = buffers.ppu.sprite0_dot[env];
        if (hit != 0) {
            constexpr std::uint32_t kFrameDots =
                static_cast<std::uint32_t>(kPpuDotsPerScanline) *
                static_cast<std::uint32_t>(kPpuScanlinesPerFrame);
            const auto distance =
                hit > hot.frame_dot ? hit - hot.frame_dot : kFrameDots - hot.frame_dot + hit;
            if (distance < next) {
                next = distance;
            }
        }
    }
    if (batch_ppu_has_scanline_counter<Mode>(buffers)) {
        const auto clock = batch_ppu_dots_until_scanline_clock(hot.frame_dot);
        if (clock < next) {
            next = clock;
        }
    }
    return next;
}

// Hot-path variant: PPU scalars come from (and return to) `hot`, which the
// caller keeps in registers across the whole kernel. `buffers` is still needed
// for the per-frame presentation-snapshot capture (scroll/latch/snapshot
// arrays are global — they are touched per frame, not per instruction).
template <BankMode Mode = kBankModeRuntime>
NESLE_CUDA_HD inline BatchPpuStepResult batch_ppu_step_env_hot(BatchBuffers& buffers,
                                                               std::uint32_t env,
                                                               std::uint32_t ppu_cycles,
                                                               PpuHotState& hot) {
    BatchPpuStepResult result;
    result.cycles = ppu_cycles;

    constexpr std::uint32_t kFrameDots =
        static_cast<std::uint32_t>(kPpuDotsPerScanline) *
        static_cast<std::uint32_t>(kPpuScanlinesPerFrame);
    constexpr std::uint32_t kSpriteZeroHitDot =
        static_cast<std::uint32_t>(kPpuCoarseSpriteZeroHitScanline) *
            static_cast<std::uint32_t>(kPpuDotsPerScanline) +
        static_cast<std::uint32_t>(kPpuCoarseSpriteZeroHitDot);
    constexpr std::uint32_t kVblankDot =
        static_cast<std::uint32_t>(kPpuVblankStartScanline) *
            static_cast<std::uint32_t>(kPpuDotsPerScanline) +
        static_cast<std::uint32_t>(kPpuVblankFlagDot);
    constexpr std::uint32_t kPreRenderDot =
        static_cast<std::uint32_t>(kPpuPreRenderScanline) *
            static_cast<std::uint32_t>(kPpuDotsPerScanline) +
        static_cast<std::uint32_t>(kPpuVblankFlagDot);

    const auto start_dot = hot.frame_dot;
    const bool short_frame = batch_ppu_short_frame(hot);
    auto end_dot = start_dot + ppu_cycles;
    if (short_frame && end_dot >= kFrameDots - 1u) {
        end_dot += 1u;  // the pre-render line's last dot does not exist in this frame
    }
    result.frames_completed = end_dot / kFrameDots;

    auto crossed = [&](std::uint32_t frame_offset, std::uint32_t event_dot) {
        const auto absolute_dot = frame_offset * kFrameDots + event_dot;
        return absolute_dot > start_dot && absolute_dot <= end_dot;
    };

    for (std::uint32_t frame_offset = 0; frame_offset <= result.frames_completed; ++frame_offset) {
        if (batch_ppu_has_sprite0_replay(buffers)) {
            // Worked out again at each scheduled point with the writes logged so far: the
            // flag goes up once the hit's dot is reached, else the next point is kept.
            const auto scheduled = buffers.ppu.sprite0_dot[env];
            const auto sprite0 = scheduled & kSprite0DotMask;
            if (sprite0 != 0 && crossed(frame_offset, sprite0)) {
                const bool known = (scheduled & kSprite0Known) != 0 && ((scheduled >> 20) & 0x1FFu) == hot.log_n;
                const auto found = known ? Sprite0Hit{sprite0, (scheduled & kSprite0Hit) != 0}
                                         : frame_sprite0_hit(buffers, env, hot.log_bank, hot.log_n);
                const bool reached = frame_offset * kFrameDots + found.dot <= end_dot;
                if (found.hit && reached) {
                    hot.status = static_cast<std::uint8_t>(hot.status | 0x40);
                }
                buffers.ppu.sprite0_dot[env] =
                    found.dot != 0 && !reached
                        ? (found.dot | kSprite0Evaluate | kSprite0Known | (found.hit ? kSprite0Hit : 0u) |
                           (static_cast<std::uint32_t>(hot.log_n) << 20))
                        : 0u;
            }
        } else if (batch_ppu_has_sprite0_scan<Mode>(buffers)) {
            const auto hit = buffers.ppu.sprite0_dot[env];
            if (hit != 0 && crossed(frame_offset, hit) && (hot.mask & 0x18) == 0x18) {
                hot.status = static_cast<std::uint8_t>(hot.status | 0x40);
            }
        } else if (crossed(frame_offset, kSpriteZeroHitDot) && (hot.mask & 0x18) != 0) {
            hot.status = static_cast<std::uint8_t>(hot.status | 0x40);
        }
        const bool replay = batch_ppu_has_sprite0_replay(buffers);
        if (replay && crossed(frame_offset, kReplayVblankFlagDot)) {
            hot.status = static_cast<std::uint8_t>(hot.status | 0x80);
        }
        if (crossed(frame_offset, kVblankDot)) {
            const auto had_nmi = hot.nmi_pending != 0;
            if (replay) {
                // FCEUmm raises NMI 12 dots after the flag, whether or not a read took it.
                if ((hot.ctrl & 0x80) != 0) {
                    hot.nmi_pending = 1;
                }
            } else {
                batch_ppu_set_vblank_hot(hot, true);
            }
            result.nmi_started = result.nmi_started || (!had_nmi && hot.nmi_pending != 0);
            if (buffers.ppu.frame_log != nullptr) {
                // Freeze the frame just drawn for render(): its log, and the nametable
                // RAM and OAM it was drawn from (the NMI handler that changes them for
                // the next frame has not run yet).
                if (buffers.cart.bank_kind == kBankingMmc2 && buffers.mapper.regs != nullptr &&
                    buffers.defer_mmc2_resolve == 0) {
                    frame_resolve_mmc2(buffers, env, hot.log_bank, hot.log_n);
                }
                hot.log_bank = frame_log_freeze(buffers, env, hot.log_bank, hot.log_n);
            }
            if (buffers.ppu.snap_nametable != nullptr) {
                const auto* oam_src = buffers.ppu.oam + static_cast<std::uint64_t>(env) * kOamBytes;
                auto* oam_dst = buffers.ppu.snap_oam + static_cast<std::uint64_t>(env) * kOamBytes;
                copy_bytes_fast(oam_dst, oam_src, static_cast<std::uint32_t>(kOamBytes));
                const auto* nt_src =
                    buffers.ppu.nametable_ram + static_cast<std::uint64_t>(env) * kNametableRamBytes;
                auto* nt_dst =
                    buffers.ppu.snap_nametable + static_cast<std::uint64_t>(env) * kNametableRamBytes;
                copy_bytes_fast(nt_dst, nt_src, static_cast<std::uint32_t>(kNametableRamBytes));
            }
        }
        if (batch_ppu_has_sprite0_replay(buffers) && crossed(frame_offset, kReplayStatusClearDot)) {
            batch_ppu_set_vblank_hot(hot, false);
            hot.status = static_cast<std::uint8_t>(hot.status & 0x1F);
        }
        if (crossed(frame_offset, kPreRenderDot)) {
            batch_ppu_set_vblank_hot(hot, false);
            hot.status = static_cast<std::uint8_t>(hot.status & 0x1F);
            if (batch_ppu_has_sprite0_replay(buffers)) {
                // This frame's log starts from the state the game left in vblank, and the
                // sprite-0 hit is looked for when sprite 0's first line is due.
                hot.log_bank = frame_log_capture(buffers, env, hot.ctrl, hot.mask, hot.log_bank);
                hot.log_n = 0;
                const std::uint32_t y = buffers.ppu.oam[static_cast<std::uint64_t>(env) * kOamBytes];
                buffers.ppu.sprite0_dot[env] =
                    y < 239u ? (nesle_dot(static_cast<int>(y + 1u) * 341 - 20) | kSprite0Evaluate) : 0u;
            } else if (batch_ppu_has_sprite0_scan<Mode>(buffers)) {
                buffers.ppu.sprite0_dot[env] = batch_ppu_sprite0_hit_dot<Mode>(buffers, env, hot.ctrl);
            }
        }
    }

    if (batch_ppu_has_scanline_counter<Mode>(buffers)) {
        batch_ppu_clock_scanline_counter(buffers, env, start_dot, end_dot, hot);
    }

    hot.frame_dot = end_dot % kFrameDots;
    hot.frame += result.frames_completed;
    if (result.frames_completed != 0) {
        hot.log_bank = static_cast<std::uint8_t>(hot.log_bank & ~kLogShortFrame);
    }

    return result;
}

NESLE_CUDA_HD inline BatchPpuStepResult batch_ppu_step_env(BatchBuffers& buffers,
                                                           std::uint32_t env,
                                                           std::uint32_t ppu_cycles) {
    auto hot = load_ppu_hot_state(buffers, env);
    const auto result = batch_ppu_step_env_hot(buffers, env, ppu_cycles, hot);
    store_ppu_hot_state(buffers, env, hot);
    return result;
}

}  // namespace nesle::cuda

#undef NESLE_CUDA_HD
