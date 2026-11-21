#pragma once

// The picture render() returns: the frame log frozen at the last vblank, replayed
// through FCEUmm's line renderer (batch_frame.cuh), in the RGB stable_retro returns.

#include <cstdint>

#include "nesle/cuda/batch_frame.cuh"
#include "nesle/cuda/state.cuh"

#ifdef __CUDACC__
#define NESLE_CUDA_RENDER_HD __host__ __device__
#else
#define NESLE_CUDA_RENDER_HD
#endif

namespace nesle::cuda {

NESLE_CUDA_RENDER_HD inline std::uint8_t* env_frame_rgb(BatchBuffers& buffers, std::uint32_t env) {
    return buffers.frames_rgb + static_cast<std::uint64_t>(env) * kFrameWidth * kFrameHeight * kRgbChannels;
}

// Where an env's picture comes from: the bank frozen at the last vblank with the
// nametable RAM and OAM frozen with it, or, without a frame log (host tests, older
// callers), the live state as a frame with no writes (its header built in `scratch`).
NESLE_CUDA_RENDER_HD inline ReplaySource render_source(const BatchBuffers& b,
                                                       std::uint32_t env,
                                                       std::uint32_t* scratch) {
    const std::uint8_t* nametable = b.ppu.nametable_ram + static_cast<std::uint64_t>(env) * kNametableRamBytes;
    const std::uint8_t* oam = b.ppu.oam + static_cast<std::uint64_t>(env) * kOamBytes;
    if (b.ppu.frame_log == nullptr) {
        frame_log_fill_header(b, env, b.ppu.ctrl[env], b.ppu.mask[env], scratch);
        return replay_source(b, env, scratch, 0, nametable, oam);
    }
    if (b.ppu.snap_nametable != nullptr) {
        nametable = b.ppu.snap_nametable + static_cast<std::uint64_t>(env) * kNametableRamBytes;
    }
    if (b.ppu.snap_oam != nullptr) {
        oam = b.ppu.snap_oam + static_cast<std::uint64_t>(env) * kOamBytes;
    }
    const std::uint32_t* h = frame_log_bank(b, env, b.ppu.log_bank[env] & 1u);
    return replay_source(b, env, h, h[0] & ~kLogOverflow, nametable, oam);
}

// Line `l` from FCEUmm's state at its start: background, sprites, greyscale, then
// the RGB of each pixel for the line's emphasis.
NESLE_CUDA_RENDER_HD inline void render_line_rgb(const ReplaySource& src,
                                                 ReplayState& s,
                                                 ReplayLine& line,
                                                 int l,
                                                 std::uint8_t* out) {
    replay_line(src, s, line, l, true);
    const std::uint32_t emphasis = replay_compose(s, line);
    const std::uint32_t tint = src.header[15] & 7u;
    for (int x = 0; x < 256; ++x) {
        fceumm_rgb(line.px[x], emphasis, tint, out + 3 * x);
    }
}

// Serial: one env's whole frame.
NESLE_CUDA_RENDER_HD inline void render_batch_rgb_frame_env(BatchBuffers& buffers, std::uint32_t env) {
    std::uint32_t scratch[kLogHeaderWords];
    const ReplaySource src = render_source(buffers, env, scratch);
    ReplayState s;
    replay_load_header(s, src.header);
    ReplayLine line;
    replay_begin_frame(src, s);
    std::uint8_t* frame = env_frame_rgb(buffers, env);
    for (int l = 0; l < kFrameHeight; ++l) {
        render_line_rgb(src, s, line, l, frame + static_cast<std::uint32_t>(l) * kFrameWidth * kRgbChannels);
        replay_end_of_line(src, s, l, true);
    }
}

// For drawing lines in parallel: FCEUmm's state at the start of every line. Lines are
// only drawn here where sprite 0 is checked (the check changes what comes after).
NESLE_CUDA_RENDER_HD inline void render_line_states(const ReplaySource& src, ReplayState* states) {
    ReplayState s;
    replay_load_header(s, src.header);
    ReplayLine line;
    replay_begin_frame(src, s);
    for (int l = 0; l < kFrameHeight; ++l) {
        states[l] = s;
        replay_line(src, s, line, l, s.sphitx != 0x100);
        s.spork = (s.mask & 0x10u) != 0 ? 0 : s.spork;  // CopySprites
        replay_end_of_line(src, s, l, true);
    }
}

}  // namespace nesle::cuda

#undef NESLE_CUDA_RENDER_HD
