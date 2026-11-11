#pragma once

// Per-frame PPU write log and the FCEUmm line renderer that replays it.
//
// The step kernel does not draw: it records, per env and per frame, the PPU state at
// the start of the frame (the pre-render line) and every PPU-visible write the CPU
// makes while lines are being drawn ($2000/$2001/$2005/$2006/$2007, palette writes,
// CHR bank and mirroring changes, MMC2 registers), each stamped with the PPU dot of
// the writing instruction and its length. Rendering replays that log through a copy
// of the line renderer in stable_retro's FCEUmm (ppu.c: RefreshLine, pputile.h,
// FetchSpriteData, RefreshSprites, CopySprites, Fixit1/Fixit2, CheckSpriteHit), so
// mid-frame scroll splits, CHR bank switches, MMC2 $FD/$FE latches and MMC3 IRQ splits
// land on the scanline and tile where FCEUmm draws them. The same replay finds the
// sprite-0 hit and resolves the MMC2 latches the frame's fetches leave.
//
// Two banks per env: the step kernel writes one while the renderer reads the other,
// which was frozen at the last vblank, so render() always shows the last whole frame.

#include <cstdint>

#include "nesle/cuda/mapper.cuh"
#include "nesle/cuda/state.cuh"

#ifdef __CUDACC__
#define NESLE_CUDA_FRAME_HD __host__ __device__
#define NESLE_CUDA_FRAME_INLINE __forceinline__
#else
#define NESLE_CUDA_FRAME_HD
#define NESLE_CUDA_FRAME_INLINE inline
#endif
// Once-a-frame work the step kernel calls out of line, so its registers and stack stay
// out of the per-instruction path (the step kernel's parameter is __grid_constant__, so
// passing it by reference does not copy it to local memory).
#if defined(__CUDA_ARCH__)
#define NESLE_CUDA_FRAME_COLD __noinline__
#else
#define NESLE_CUDA_FRAME_COLD inline
#endif

namespace nesle::cuda {

// ----------------------------------------------------------------- timing

constexpr std::uint32_t kFrameDotsPerFrame = 341u * 262u;
// The log runs from the pre-render settle to vblank; writes in vblank only change the
// state the next frame starts from.
constexpr std::uint32_t kFrameLogCaptureDot = 261u * 341u + 1u;
constexpr std::uint32_t kFrameLogVblankDot = 241u * 341u + 1u;
// FCEUmm's line grid sits this many dots later than NeSLE's for the same CPU instant:
// NeSLE raises NMI at line 241 dot 1, FCEUmm at line 241 dot 12, and both CPUs then
// run the same instructions.
constexpr int kFceummDotOffset = 11;

// FCEUmm frame time of NeSLE frame dot `dot`: line 0 starts at 0, the pre-render line
// is negative. Every other frame ("kook") FCEUmm's pre-render line is a dot short, and
// so is NeSLE's (batch_ppu.cuh): the writes before the wrap sit a dot closer to line 0.
NESLE_CUDA_FRAME_HD NESLE_CUDA_FRAME_INLINE int fceumm_time(std::uint32_t dot, int kook) {
    const int d = static_cast<int>(dot);
    return dot >= kFrameLogCaptureDot ? d - static_cast<int>(kFrameDotsPerFrame) + kFceummDotOffset + kook
                                      : d + kFceummDotOffset;
}

// NeSLE frame dot of FCEUmm time `fceumm` on a visible line.
NESLE_CUDA_FRAME_HD NESLE_CUDA_FRAME_INLINE std::uint32_t nesle_dot(int fceumm) {
    return static_cast<std::uint32_t>(fceumm - kFceummDotOffset);
}

// ------------------------------------------------------------- log layout

constexpr std::uint32_t kLogHeaderWords = 16;
// Writes per frame a bank holds (4 KB per env for both banks). Final Mission's per-line
// scroll needs 192; past the limit the rest of the frame's writes are dropped.
constexpr std::uint32_t kLogEvents = 256;
constexpr std::uint32_t kLogBankWords = kLogHeaderWords + 2u * kLogEvents;
constexpr std::uint32_t kFrameLogWords = 2u * kLogBankWords;  // per env
constexpr std::uint32_t kLogOverflow = 0x80000000u;
constexpr std::uint32_t kLogKook = 0x100u;
// log_bank bits: 0 the bank render() replays, 1 the next frame's kook, 2 this frame is
// being shortened (from its pre-render line to its wrap).
constexpr std::uint8_t kLogNextKook = 2u;
constexpr std::uint8_t kLogShortFrame = 4u;

// Header words: 0 count (| kLogOverflow), 1 ctrl | mask << 8 | fine x << 16,
// 2 v | t << 16, 3-4 CHR 1 KB page per slot, 5 mirroring page map | MMC2 latch 0 << 8 |
// latch 1 << 16, 6 MMC2 CHR registers, 7-14 palette RAM, 15 the emphasis FCEUmm's
// emphasis palette holds when the frame is shown (bits 0-2) | kLogKook.
// Event: word 0 start dot (17 bits) | instruction cycles << 17 (4 bits) | kind << 24,
// word 1 value | aux << 8.
enum FrameLogEvent : std::uint8_t {
    kEvCtrl = 1,       // $2000
    kEvMask = 2,       // $2001
    kEvScroll = 3,     // $2005, aux = write toggle before the write
    kEvAddr = 4,       // $2006, aux = write toggle before the write
    kEvDataWrite = 5,  // $2007 write below the palette: address increment, aux = repeats
    kEvDataRead = 6,   // $2007 read (address increment)
    kEvPalette = 7,    // $2007 palette write: value, aux = palette index; address increment
    kEvChr = 8,        // aux = CHR slot, value = 1 KB page
    kEvMirror = 9,     // value = nametable page map
    kEvMmc2 = 10,      // aux = MMC2 register (0-3 CHR, 6-7 latches), value
};

NESLE_CUDA_FRAME_HD NESLE_CUDA_FRAME_INLINE std::uint32_t* frame_log_bank(const BatchBuffers& b,
                                                                          std::uint32_t env,
                                                                          std::uint32_t bank) {
    return b.ppu.frame_log + static_cast<std::uint64_t>(env) * kFrameLogWords + bank * kLogBankWords;
}

NESLE_CUDA_FRAME_HD NESLE_CUDA_FRAME_INLINE int log_event_time(const std::uint32_t* events,
                                                              std::uint32_t i,
                                                              int kook) {
    return fceumm_time(events[2u * i] & 0x1FFFFu, kook);
}

// ------------------------------------------------------------- recording

// The live PPU and board state a frame starts from, as a log header.
NESLE_CUDA_FRAME_HD inline void frame_log_fill_header(const BatchBuffers& b,
                                                      std::uint32_t env,
                                                      std::uint8_t ctrl,
                                                      std::uint8_t mask,
                                                      std::uint32_t* h) {
    h[0] = 0;
    h[1] = ctrl | (static_cast<std::uint32_t>(mask) << 8) |
           (static_cast<std::uint32_t>(b.ppu.x != nullptr ? b.ppu.x[env] : 0) << 16);
    const std::uint32_t v = b.ppu.v != nullptr ? b.ppu.v[env] : 0;
    const std::uint32_t t = b.ppu.t != nullptr ? b.ppu.t[env] : ((ctrl & 3u) << 10);
    h[2] = v | (t << 16);
    h[3] = h[4] = h[5] = h[6] = h[15] = 0;
    if (b.cart.bank_kind >= kBankingFirstPaged && b.mapper.chr_page != nullptr) {
        const std::uint32_t* pages = b.mapper.chr_page + static_cast<std::uint64_t>(env) * kChrPages;
        for (std::uint32_t i = 0; i < 8u; ++i) {
            h[3 + (i >> 2)] |= ((pages[i] >> 10) & 0xFFu) << ((i & 3u) * 8u);
        }
        h[5] = b.mapper.mirroring[env];
        if (b.cart.bank_kind == kBankingMmc2) {
            const std::uint8_t* r = paged::regs(b, env);
            h[5] |= (static_cast<std::uint32_t>(r[paged::kMmc2Latch0] & 1u) << 8) |
                    (static_cast<std::uint32_t>(r[paged::kMmc2Latch1] & 1u) << 16);
            h[6] = r[0] | (static_cast<std::uint32_t>(r[1]) << 8) | (static_cast<std::uint32_t>(r[2]) << 16) |
                   (static_cast<std::uint32_t>(r[3]) << 24);
        }
    }
    const std::uint8_t* pal = b.ppu.palette_ram + static_cast<std::uint64_t>(env) * kPaletteRamBytes;
    for (std::uint32_t i = 0; i < 8u; ++i) {
        h[7 + i] = pal[4 * i] | (static_cast<std::uint32_t>(pal[4 * i + 1]) << 8) |
                   (static_cast<std::uint32_t>(pal[4 * i + 2]) << 16) |
                   (static_cast<std::uint32_t>(pal[4 * i + 3]) << 24);
    }
}

// Pre-render line: the writing bank starts over from the live state (the caller's
// hot.log_n becomes 0). Returns the new log_bank. (Scalars in and out: a reference to
// the register-resident PpuHotState would push it to memory for the whole kernel.)
NESLE_CUDA_FRAME_HD NESLE_CUDA_FRAME_COLD std::uint8_t frame_log_capture(const BatchBuffers& b,
                                                                       std::uint32_t env,
                                                                       std::uint8_t ctrl,
                                                                       std::uint8_t mask,
                                                                       std::uint8_t log_bank) {
    std::uint32_t* h = frame_log_bank(b, env, (log_bank & 1u) ^ 1u);
    frame_log_fill_header(b, env, ctrl, mask, h);
    const bool kook = (log_bank & kLogNextKook) != 0;
    if (kook) h[15] = kLogKook;
    return static_cast<std::uint8_t>(((log_bank ^ kLogNextKook) & ~kLogShortFrame) | (kook ? kLogShortFrame : 0u));
}

// FCEUmm's emphasis palette (palette entries 0x40-0x7F) after the frame in bank `h`:
// it counts, per line, the emphasis of the last $2001 write that had any (taken from
// the mask at line 0), and at the frame's end tints the palette for the most frequent
// one; a frame with none leaves the previous tint.
NESLE_CUDA_FRAME_HD inline std::uint32_t frame_log_emphasis(const std::uint32_t* h, std::uint32_t previous) {
    const std::uint32_t* events = h + kLogHeaderWords;
    const std::uint32_t count = h[0] & ~kLogOverflow;
    std::uint32_t mask = (h[1] >> 8) & 0xFFu;
    bool any = (mask & 0xE0u) != 0;
    for (std::uint32_t i = 0; i < count && !any; ++i) {
        any = (events[2u * i] >> 24) == kEvMask && (events[2u * i + 1u] & 0xE0u) != 0;
    }
    if (!any) return previous;
    const int kook = (h[15] & kLogKook) != 0 ? 1 : 0;
    std::uint32_t i = 0;
    for (; i < count && log_event_time(events, i, kook) < 0; ++i) {
        if ((events[2u * i] >> 24) == kEvMask) mask = events[2u * i + 1u] & 0xFFu;
    }
    std::uint32_t deemp = mask >> 5;
    std::uint32_t lines[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int l = 0; l < 240; ++l) {
        for (; i < count && log_event_time(events, i, kook) < l * 341; ++i) {
            const std::uint32_t value = events[2u * i + 1u] & 0xFFu;
            if ((events[2u * i] >> 24) == kEvMask && (value & 0xE0u) != 0) deemp = value >> 5;
        }
        ++lines[deemp];
    }
    std::uint32_t most = 0, best = 0;
    for (std::uint32_t e = 1; e < 7u; ++e) {
        if (lines[e] > most) {
            most = lines[e];
            best = e;
        }
    }
    return best != 0 ? best : previous;
}

// Vblank: the writing bank becomes the one render() replays. Returns the new log_bank.
NESLE_CUDA_FRAME_HD NESLE_CUDA_FRAME_COLD std::uint8_t frame_log_freeze(const BatchBuffers& b,
                                                                      std::uint32_t env,
                                                                      std::uint8_t log_bank,
                                                                      std::uint16_t log_n) {
    std::uint32_t* h = frame_log_bank(b, env, (log_bank & 1u) ^ 1u);
    const std::uint32_t* shown = frame_log_bank(b, env, log_bank & 1u);
    h[0] = (h[0] & kLogOverflow) | log_n;
    h[15] = (h[15] & ~7u) | frame_log_emphasis(h, shown[15] & 7u);
    return static_cast<std::uint8_t>(log_bank ^ 1u);
}

// A reset env presents its reset state: both banks start from it, with no writes, and
// the live nametable RAM and OAM become the presentation snapshot.
NESLE_CUDA_FRAME_HD inline void frame_log_reset(BatchBuffers& b, std::uint32_t env) {
    if (b.ppu.frame_log == nullptr) {
        return;
    }
    frame_log_fill_header(b, env, b.ppu.ctrl[env], b.ppu.mask[env], frame_log_bank(b, env, 0));
    frame_log_fill_header(b, env, b.ppu.ctrl[env], b.ppu.mask[env], frame_log_bank(b, env, 1));
    b.ppu.log_count[env] = 0;
    b.ppu.log_bank[env] = 0;
    if (b.ppu.snap_nametable != nullptr) {
        copy_bytes_fast(b.ppu.snap_nametable + static_cast<std::uint64_t>(env) * kNametableRamBytes,
                        b.ppu.nametable_ram + static_cast<std::uint64_t>(env) * kNametableRamBytes,
                        static_cast<std::uint32_t>(kNametableRamBytes));
    }
    if (b.ppu.snap_oam != nullptr) {
        copy_bytes_fast(b.ppu.snap_oam + static_cast<std::uint64_t>(env) * kOamBytes,
                        b.ppu.oam + static_cast<std::uint64_t>(env) * kOamBytes,
                        static_cast<std::uint32_t>(kOamBytes));
    }
}

NESLE_CUDA_FRAME_HD NESLE_CUDA_FRAME_INLINE bool frame_log_active(const BatchBuffers& b, const PpuHotState* hot) {
    return hot != nullptr && b.ppu.frame_log != nullptr &&
           (hot->now < kFrameLogVblankDot || hot->now >= kFrameLogCaptureDot);
}

// One PPU-visible write, at the instruction that `hot.now` started. Writes in vblank
// are not logged: the next frame's header is taken after them.
NESLE_CUDA_FRAME_HD NESLE_CUDA_FRAME_INLINE void frame_log_event(const BatchBuffers& b,
                                                                 std::uint32_t env,
                                                                 PpuHotState* hot_ptr,
                                                                 std::uint8_t kind,
                                                                 std::uint32_t value,
                                                                 std::uint32_t aux = 0) {
    if (!frame_log_active(b, hot_ptr)) {
        return;
    }
    PpuHotState& hot = *hot_ptr;
    std::uint32_t* bank = frame_log_bank(b, env, (hot.log_bank & 1u) ^ 1u);
    if (kind == kEvDataWrite && (hot.mask & 0x18u) == 0 && hot.log_n != 0 && hot.log_n <= kLogEvents) {
        // A VRAM upload with rendering off draws nothing: its address increments only
        // need to add up, so a run of them is one event (repeat count in aux).
        std::uint32_t* last = bank + kLogHeaderWords + 2u * (hot.log_n - 1u);
        if ((last[0] >> 24) == kEvDataWrite && (last[1] >> 8) < 0xFFu) {
            last[1] += 0x100u;
            return;
        }
    }
    if (hot.log_n >= kLogEvents) {
        bank[0] |= kLogOverflow;
        return;
    }
    std::uint32_t* e = bank + kLogHeaderWords + 2u * hot.log_n;
    e[0] = (hot.now & 0x1FFFFu) | (static_cast<std::uint32_t>(kind) << 24);
    e[1] = (value & 0xFFu) | ((aux & 0xFFu) << 8);
    ++hot.log_n;
}

// The CPU core adds an instruction's cycles when it ends; FCEUmm stamps the accesses
// with the end time. Called after the instruction for the events it logged.
NESLE_CUDA_FRAME_HD NESLE_CUDA_FRAME_INLINE void frame_log_set_cycles(const BatchBuffers& b,
                                                                      std::uint32_t env,
                                                                      const PpuHotState& hot,
                                                                      std::uint16_t first,
                                                                      std::uint32_t cycles) {
    std::uint32_t* bank = frame_log_bank(b, env, (hot.log_bank & 1u) ^ 1u);
    const std::uint32_t c = (cycles > 15u ? 15u : cycles) << 17;
    for (std::uint32_t i = first; i < hot.log_n && i < kLogEvents; ++i) {
        bank[kLogHeaderWords + 2u * i] |= c;
    }
}

// ---------------------------------------------------------------- colours

// FCEUmm's default 64-colour palette (its "palette" table, as stable_retro's core
// carries it), 8 bits per channel.
#define NESLE_FCEUMM_PALETTE                                                                                     \
    116, 116, 116, 36, 24, 140, 0, 0, 168, 68, 0, 156, 140, 0, 116, 168, 0, 16, 164, 0, 0, 124, 8, 0, 64, 44, 0,  \
        0, 68, 0, 0, 80, 0, 0, 60, 20, 24, 60, 92, 0, 0, 0, 0, 0, 0, 0, 0, 0, 188, 188, 188, 0, 112, 236, 32, 56,   \
        236, 128, 0, 240, 188, 0, 188, 228, 0, 88, 216, 40, 0, 200, 76, 12, 136, 112, 0, 0, 148, 0, 0, 168, 0, 0,   \
        144, 56, 0, 128, 136, 0, 0, 0, 0, 0, 0, 0, 0, 0, 252, 252, 252, 60, 188, 252, 92, 148, 252, 64, 136, 252,   \
        244, 120, 252, 252, 116, 180, 252, 116, 96, 252, 152, 56, 240, 188, 60, 128, 208, 16, 76, 220, 72, 88, 248, \
        152, 0, 232, 216, 120, 120, 120, 0, 0, 0, 0, 0, 0, 252, 252, 252, 168, 228, 252, 196, 212, 252, 212, 200,   \
        252, 252, 196, 252, 252, 196, 216, 252, 188, 176, 252, 216, 168, 252, 228, 160, 224, 252, 160, 168, 240,    \
        188, 176, 252, 204, 156, 252, 240, 196, 196, 196, 0, 0, 0, 0, 0, 0
// SetNESDeemph's channel multipliers (1/32768) for emphasis 1-6, then the 0.75 dim of
// emphasis 7.
#define NESLE_FCEUMM_EMPHASIS                                                                     \
    40599, 26017, 33390, 29655, 33521, 24281, 24576, 29982, 35586, 32112, 33619, 29753, 32342, 24576, \
        24346, 28901, 21397, 41844, 32079, 3309, 24576

constexpr std::uint8_t kFceummPalette[64 * 3] = {NESLE_FCEUMM_PALETTE};
constexpr std::uint16_t kFceummEmphasis[3 * 7] = {NESLE_FCEUMM_EMPHASIS};
#ifdef __CUDACC__
static __device__ __constant__ const std::uint8_t kFceummPaletteDevice[64 * 3] = {NESLE_FCEUMM_PALETTE};
static __device__ __constant__ const std::uint16_t kFceummEmphasisDevice[3 * 7] = {NESLE_FCEUMM_EMPHASIS};
#endif
#undef NESLE_FCEUMM_PALETTE
#undef NESLE_FCEUMM_EMPHASIS

// The RGB stable_retro returns for colour `index` on a line whose $2001 emphasis is
// `emphasis` (FCEUmm marks each line with it), when the emphasis palette holds `tint`.
// The core hands frames over as RGB565, so each channel keeps 5/6/5 bits.
NESLE_CUDA_FRAME_HD NESLE_CUDA_FRAME_INLINE void fceumm_rgb(std::uint32_t index,
                                                            std::uint32_t emphasis,
                                                            std::uint32_t tint,
                                                            std::uint8_t* out) {
#ifdef __CUDA_ARCH__
    const std::uint8_t* palette = kFceummPaletteDevice;
    const std::uint16_t* scale = kFceummEmphasisDevice;
#else
    const std::uint8_t* palette = kFceummPalette;
    const std::uint16_t* scale = kFceummEmphasis;
#endif
    const std::uint8_t* rgb = palette + 3u * (index & 0x3Fu);
    std::uint32_t c[3] = {rgb[0], rgb[1], rgb[2]};
    if (emphasis != 0) {
        const std::uint32_t e = emphasis == 7u ? 7u : tint;  // tint 0: entries never set, black
        for (std::uint32_t k = 0; k < 3u; ++k) {
            const std::uint32_t v = e == 0 ? 0u : (c[k] * scale[7u * k + e - 1u]) >> 15;
            c[k] = v > 255u ? 255u : v;
        }
    }
    out[0] = static_cast<std::uint8_t>(c[0] & 0xF8u);
    out[1] = static_cast<std::uint8_t>(c[1] & 0xFCu);
    out[2] = static_cast<std::uint8_t>(c[2] & 0xF8u);
}

// --------------------------------------------------------------- replay

// What one replay reads: the board, the event log, and which memories (live ones in the
// step kernel, the vblank snapshot in the renderer).
struct ReplaySource {
    const BatchBuffers* b;
    std::uint32_t env;
    const std::uint32_t* header;
    const std::uint8_t* nametable;  // kNametableRamBytes
    const std::uint8_t* oam;        // kOamBytes
    const std::uint32_t* events;    // 2 words each
    std::uint32_t count;
    int kook;
};

NESLE_CUDA_FRAME_HD NESLE_CUDA_FRAME_INLINE ReplaySource replay_source(const BatchBuffers& b,
                                                                       std::uint32_t env,
                                                                       const std::uint32_t* header,
                                                                       std::uint32_t count,
                                                                       const std::uint8_t* nametable,
                                                                       const std::uint8_t* oam) {
    return ReplaySource{&b,  env, header, nametable, oam, header + kLogHeaderWords,
                        count < kLogEvents ? count : kLogEvents, (header[15] & kLogKook) != 0 ? 1 : 0};
}

// Up to eight sprites of one line: pattern bytes, attributes, x.
struct ReplaySprites {
    std::uint8_t n;
    std::uint8_t lo[8];
    std::uint8_t hi[8];
    std::uint8_t attr[8];
    std::uint8_t x[8];
};

// FCEUmm's PPU and board state between lines; everything a line needs to be drawn.
struct ReplayState {
    std::uint16_t v;
    std::uint16_t t;
    std::uint8_t x;
    std::uint8_t ctrl;
    std::uint8_t mask;
    std::uint8_t mirroring;  // paged boards
    std::uint8_t chr[8];     // paged boards: 1 KB page per slot
    std::uint8_t creg[4];    // MMC2
    std::uint8_t latch[2];   // MMC2
    std::uint8_t pal[32];
    std::uint32_t pshift0;
    std::uint32_t pshift1;
    std::uint8_t atlatch;
    ReplaySprites fetched;   // FetchSpriteData (SPRBUF) and its SpriteBlurp
    std::uint8_t blurp;
    ReplaySprites drawn;     // RefreshSprites' sprite line (sprlinebuf) and its palette
    std::uint8_t spal[16];
    std::uint8_t spork;      // a sprite line waits for CopySprites
    std::uint8_t hit;        // PPU_status bit 6
    std::uint8_t sphitdata;
    std::int16_t sphitx;     // 0x100 = no sprite-0 check on this line
    std::int16_t hit_x;      // where the hit was found
    std::uint16_t cursor;    // next event
};

// Line scratch: FCEUmm's per-line pixel buffer (palette value, 0x40 = transparent
// background, 0xFF = not drawn yet) and its RefreshLine bookkeeping.
struct ReplayLine {
    std::uint8_t px[256];
    int firsttile;
    int outpos;
    bool tofix;
};

NESLE_CUDA_FRAME_HD NESLE_CUDA_FRAME_INLINE int event_start(const ReplaySource& src, std::uint32_t i) {
    return log_event_time(src.events, i, src.kook);
}

NESLE_CUDA_FRAME_HD NESLE_CUDA_FRAME_INLINE int event_end(const ReplaySource& src, std::uint32_t i) {
    return event_start(src, i) + 3 * static_cast<int>((src.events[2u * i] >> 17) & 0xFu);
}

NESLE_CUDA_FRAME_HD NESLE_CUDA_FRAME_INLINE std::uint32_t event_kind(const ReplaySource& src, std::uint32_t i) {
    return src.events[2u * i] >> 24;
}

NESLE_CUDA_FRAME_HD NESLE_CUDA_FRAME_INLINE void replay_load_header(ReplayState& s, const std::uint32_t* h) {
    s.ctrl = static_cast<std::uint8_t>(h[1]);
    s.mask = static_cast<std::uint8_t>(h[1] >> 8);
    s.x = static_cast<std::uint8_t>((h[1] >> 16) & 7u);
    s.v = static_cast<std::uint16_t>(h[2] & 0x7FFFu);
    s.t = static_cast<std::uint16_t>((h[2] >> 16) & 0x7FFFu);
    for (int i = 0; i < 8; ++i) s.chr[i] = static_cast<std::uint8_t>(h[3 + (i >> 2)] >> ((i & 3) * 8));
    s.mirroring = static_cast<std::uint8_t>(h[5]);
    s.latch[0] = static_cast<std::uint8_t>((h[5] >> 8) & 1u);
    s.latch[1] = static_cast<std::uint8_t>((h[5] >> 16) & 1u);
    for (int i = 0; i < 4; ++i) s.creg[i] = static_cast<std::uint8_t>(h[6] >> (i * 8));
    for (int i = 0; i < 32; ++i) s.pal[i] = static_cast<std::uint8_t>(h[7 + (i >> 2)] >> ((i & 3) * 8));
    s.pshift0 = s.pshift1 = 0;
    s.atlatch = 0;
    s.fetched.n = 0;
    s.blurp = 0;
    s.drawn.n = 0;
    s.spork = 0;
    s.hit = 0;
    s.sphitdata = 0;
    s.sphitx = 0x100;
    s.hit_x = -1;
    s.cursor = 0;
}

// CHR byte through the replay's own bank state (paged boards), else the fixed wiring.
NESLE_CUDA_FRAME_HD NESLE_CUDA_FRAME_INLINE std::uint8_t replay_chr(const ReplaySource& src,
                                                                    const ReplayState& s,
                                                                    std::uint32_t address) {
    const BatchBuffers& b = *src.b;
    address &= 0x1FFFu;
    if (b.cart.bank_kind >= kBankingFirstPaged) {
        const std::uint32_t offset =
            ((static_cast<std::uint32_t>(s.chr[address >> 10]) & b.cart.chr_page_mask) << 10) |
            (address & 0x3FFu);
        if (b.cart.chr_rom != nullptr) return b.cart.chr_rom[offset];
        return b.ppu.chr_ram[static_cast<std::uint64_t>(src.env) * kChrRamBytes + (offset & 0x1FFFu)];
    }
    if (b.cart.chr_rom == nullptr) {
        return b.ppu.chr_ram == nullptr
                   ? 0
                   : b.ppu.chr_ram[static_cast<std::uint64_t>(src.env) * kChrRamBytes + address];
    }
    return b.cart.chr_rom[address % b.cart.chr_rom_size];
}

NESLE_CUDA_FRAME_HD NESLE_CUDA_FRAME_INLINE std::uint8_t replay_nt(const ReplaySource& src,
                                                                   const ReplayState& s,
                                                                   std::uint32_t address) {
    const BatchBuffers& b = *src.b;
    const std::uint32_t index = (address - 0x2000u) & 0x0FFFu;
    std::uint32_t offset;
    if (b.cart.bank_kind >= kBankingFirstPaged) {
        offset = (((s.mirroring >> (index >> 10)) & 1u) << 10) | (index & 0x3FFu);
    } else {
        offset = mirror_nametable_address(b.cart.nametable_arrangement, static_cast<std::uint16_t>(address));
    }
    return src.nametable[offset];
}

NESLE_CUDA_FRAME_HD NESLE_CUDA_FRAME_INLINE void replay_mmc2_select(ReplayState& s, std::uint32_t half) {
    const std::uint32_t bank = s.creg[2u * half + s.latch[half]];
    for (std::uint32_t i = 0; i < 4u; ++i) s.chr[4u * half + i] = static_cast<std::uint8_t>(bank * 4u + i);
}

// FCEUmm's MMC2 PPU_hook: a fetch from $0FD0-$0FEF / $1FD0-$1FEF flips a latch.
NESLE_CUDA_FRAME_HD NESLE_CUDA_FRAME_INLINE void replay_mmc2_hook(const ReplaySource& src,
                                                                  ReplayState& s,
                                                                  std::uint32_t address) {
    if (src.b->cart.bank_kind != kBankingMmc2) return;
    const std::uint32_t high = (address >> 8) & 0xFFu;
    if (high >= 0x20u || (high & 0x0Fu) != 0x0Fu) return;
    const std::uint32_t low = address & 0xF0u;
    if (low != 0xD0u && low != 0xE0u) return;
    const std::uint32_t half = high < 0x10u ? 0u : 1u;
    s.latch[half] = low == 0xE0u ? 1u : 0u;
    replay_mmc2_select(s, half);
}

NESLE_CUDA_FRAME_HD NESLE_CUDA_FRAME_INLINE bool replay_rendering(const ReplayState& s) {
    return (s.mask & 0x18u) != 0;
}

// Fixit1, and a $2007 read while drawing (FCEUmm A2007): v steps to the next pixel row.
NESLE_CUDA_FRAME_HD NESLE_CUDA_FRAME_INLINE void replay_vertical_increment(ReplayState& s) {
    std::uint32_t rad = s.v;
    if ((rad & 0x7000u) == 0x7000u) {
        rad ^= 0x7000u;
        if ((rad & 0x3E0u) == 0x3A0u) rad ^= 0xBA0u;
        else if ((rad & 0x3E0u) == 0x3E0u) rad ^= 0x3E0u;
        else rad += 0x20u;
    } else {
        rad += 0x1000u;
    }
    s.v = static_cast<std::uint16_t>(rad & 0x7FFFu);
}

NESLE_CUDA_FRAME_HD inline void replay_apply(const ReplaySource& src, ReplayState& s, std::uint32_t i,
                                             bool drawing) {
    const std::uint32_t w1 = src.events[2u * i + 1u];
    const std::uint32_t value = w1 & 0xFFu;
    const std::uint32_t aux = (w1 >> 8) & 0xFFu;
    switch (event_kind(src, i)) {
        case kEvCtrl:
            s.ctrl = static_cast<std::uint8_t>(value);
            s.t = static_cast<std::uint16_t>((s.t & 0xF3FFu) | ((value & 3u) << 10));
            break;
        case kEvMask:
            s.mask = static_cast<std::uint8_t>(value);
            break;
        case kEvScroll:
            if (aux == 0) {
                s.t = static_cast<std::uint16_t>((s.t & 0xFFE0u) | (value >> 3));
                s.x = static_cast<std::uint8_t>(value & 7u);
            } else {
                s.t = static_cast<std::uint16_t>((s.t & 0x8C1Fu) | ((value & 0xF8u) << 2) | ((value & 7u) << 12));
            }
            break;
        case kEvAddr:
            if (aux == 0) {
                s.t = static_cast<std::uint16_t>((s.t & 0x00FFu) | ((value & 0x3Fu) << 8));
            } else {
                s.t = static_cast<std::uint16_t>((s.t & 0xFF00u) | value);
                s.v = s.t;
            }
            break;
        case kEvDataWrite:
            s.v = static_cast<std::uint16_t>((s.v + (aux + 1u) * ((s.ctrl & 4u) != 0 ? 32u : 1u)) & 0x7FFFu);
            break;
        case kEvDataRead:
            if (drawing && replay_rendering(s)) {
                replay_vertical_increment(s);
            } else {
                s.v = static_cast<std::uint16_t>((s.v + ((s.ctrl & 4u) != 0 ? 32u : 1u)) & 0x7FFFu);
            }
            break;
        case kEvPalette:
            s.pal[aux & 31u] = static_cast<std::uint8_t>(value);
            s.v = static_cast<std::uint16_t>((s.v + ((s.ctrl & 4u) != 0 ? 32u : 1u)) & 0x7FFFu);
            break;
        case kEvChr:
            s.chr[aux & 7u] = static_cast<std::uint8_t>(value);
            break;
        case kEvMirror:
            s.mirroring = static_cast<std::uint8_t>(value);
            break;
        case kEvMmc2:
            if (aux < 4u) {
                s.creg[aux] = static_cast<std::uint8_t>(value);
                replay_mmc2_select(s, aux >> 1);
            } else if (aux == 6u || aux == 7u) {
                s.latch[aux - 6u] = static_cast<std::uint8_t>(value & 1u);
                replay_mmc2_select(s, aux - 6u);
            }
            break;
        default:
            break;
    }
}

// Pre-render line: the writes before line 0's sprite-free start (ResetRL, 16 dots before
// it), and v = t there.
NESLE_CUDA_FRAME_HD inline void replay_begin_frame(const ReplaySource& src, ReplayState& s) {
    while (s.cursor < src.count && event_start(src, s.cursor) < -16) {
        replay_apply(src, s, s.cursor++, false);
    }
    if (replay_rendering(s)) s.v = s.t;
}

// RefreshLine's tile loop for tiles [firsttile, last) of the current line (pputile.h).
// Without `pixels` only what later lines and the board see is kept.
// kLatchesOnly: frame_resolve_mmc2's replay, which needs MMC2's latches and nothing drawn
// (a separate copy: in the sprite-0 replay it pushed the step kernel's registers to local).
template <bool kLatchesOnly = false>
NESLE_CUDA_FRAME_HD inline void replay_tiles(const ReplaySource& src, ReplayState& s, ReplayLine& line,
                                             int last, bool pixels) {
    const int first = line.firsttile;
    const std::uint8_t backdrop = static_cast<std::uint8_t>((s.pal[0] & 0x3Fu) | 0x40u);
    if (!replay_rendering(s)) {
        // Rendering off: backdrop, and FCEUmm advances its output by every tile here,
        // the two prefetch tiles included.
        if (pixels) {
            for (int p = line.outpos; p < line.outpos + 8 * (last - first) && p < 256; ++p) line.px[p] = backdrop;
        }
        line.outpos += 8 * (last - first);
        line.firsttile = last;
        return;
    }
    int tile = first;
    if (!pixels && (kLatchesOnly || src.b->cart.bank_kind != kBankingMmc2)) {
        // No pixels: the address moves, and only the last two tiles' fetches can still be
        // seen (the shift registers hold two tiles). For MMC2's latches every pattern
        // address still counts, and that needs only the tile number.
        for (; tile < last - 2; ++tile) {
            const std::uint32_t v = s.v;
            if (kLatchesOnly) {
                const std::uint32_t tile_index = replay_nt(src, s, 0x2000u | (v & 0x0FFFu));
                replay_mmc2_hook(src, s, (tile_index << 4) + (((s.ctrl & 0x10u) << 8) | ((v >> 12) & 7u)));
            }
            s.v = static_cast<std::uint16_t>(((v & 0x1Fu) == 0x1Fu) ? (v ^ 0x41Fu) : (v + 1u));
        }
    }
    for (; tile < last; ++tile) {
        if (tile >= 2) {
            if (pixels && line.outpos < 256) {
                const std::uint32_t lo = (s.pshift0 >> (8u - s.x)) & 0xFFu;
                const std::uint32_t hi = (s.pshift1 >> (8u - s.x)) & 0xFFu;
                for (int k = 0; k < 8; ++k) {
                    const std::uint32_t c = ((lo >> (7 - k)) & 1u) | (((hi >> (7 - k)) & 1u) << 1);
                    const std::uint32_t at = (k + s.x) < 8 ? (s.atlatch & 3u) : ((s.atlatch >> 2) & 3u);
                    line.px[line.outpos + k] =
                        c == 0 ? backdrop : static_cast<std::uint8_t>(s.pal[at * 4u + c] & 0x3Fu);
                }
            }
            line.outpos += 8;
        }
        const std::uint32_t v = s.v;
        const std::uint32_t nt_base = 0x2000u | (v & 0x0C00u);
        const std::uint32_t tile_index = replay_nt(src, s, nt_base | (v & 0x3FFu));
        const std::uint32_t zz = v & 0x1Fu;
        const std::uint32_t attr_byte = replay_nt(src, s, nt_base | 0x3C0u | (zz >> 2) | ((v & 0x380u) >> 4));
        const std::uint32_t cc = (attr_byte >> ((zz & 2u) + ((v & 0x40u) >> 4))) & 3u;
        const std::uint32_t vadr = (tile_index << 4) + (((s.ctrl & 0x10u) << 8) | ((v >> 12) & 7u));
        // Both bytes through the bank before the hook: FCEUmm takes the CHR pointer first.
        const std::uint8_t p0 = replay_chr(src, s, vadr);
        const std::uint8_t p1 = replay_chr(src, s, vadr + 8u);
        replay_mmc2_hook(src, s, vadr);
        s.atlatch = static_cast<std::uint8_t>((s.atlatch >> 2) | (cc << 2));
        s.pshift0 = (s.pshift0 << 8) | p0;
        s.pshift1 = (s.pshift1 << 8) | p1;
        s.v = static_cast<std::uint16_t>(((v & 0x1Fu) == 0x1Fu) ? (v ^ 0x41Fu) : (v + 1u));
    }
    if (pixels) {
        if (first <= 2 && 2 < last && (s.mask & 0x02u) == 0) {
            for (int p = 0; p < 8; ++p) line.px[p] = backdrop;  // left 8 pixels of background hidden
        }
        if ((s.mask & 0x08u) == 0) {  // background off, sprites on
            int tstart = first - 2;
            int tcount = last - first;
            if (tstart < 0) {
                tcount += tstart;
                tstart = 0;
            }
            for (int p = tstart * 8; p < (tstart + tcount) * 8 && p < 256; ++p) line.px[p] = backdrop;
        }
    }
    line.firsttile = last;
}

// CheckSpriteHit: sprite 0's opaque pixels against the background drawn left of
// line pixel `p` - 16.
NESLE_CUDA_FRAME_HD NESLE_CUDA_FRAME_INLINE void replay_check_hit(ReplayState& s, const ReplayLine& line, int p) {
    if (s.sphitx == 0x100) return;
    for (int x = s.sphitx; x < s.sphitx + 8 && x < p - 16 && x < 255; ++x) {
        if ((s.sphitdata & (0x80u >> (x - s.sphitx))) != 0 && (line.px[x] & 0x40u) == 0) {
            s.hit = 1;
            s.hit_x = static_cast<std::int16_t>(x);
            s.sphitx = 0x100;
            return;
        }
    }
}

// FCEUmm's FCEUPPU_LineUpdate at line pixel `p` (RefreshLine): the tiles up to it, its
// once-per-line vertical step (Fixit1) past pixel 268, the sprite-0 check.
template <bool kLatchesOnly = false>
NESLE_CUDA_FRAME_HD NESLE_CUDA_FRAME_INLINE void replay_update(const ReplaySource& src, ReplayState& s,
                                                               ReplayLine& line, int p, bool pixels) {
    int last = p >> 3;
    if (pixels && s.sphitx != 0x100 && s.sphitx < p - 16 && !(s.sphitx < (last - 2) * 8)) {
        ++last;  // one tile more, so the check sees every pixel left of p - 16
    }
    if (last > 34) last = 34;
    if (last <= line.firsttile) return;
    replay_tiles<kLatchesOnly>(src, s, line, last, pixels);
    if (p >= 268 && line.tofix) {
        if (replay_rendering(s)) replay_vertical_increment(s);
        line.tofix = false;
    }
    if (pixels) replay_check_hit(s, line, p);
}

// Line `l`: its tiles with the writes that land while it is drawn (DoLine up to EndRL).
// `pixels` fills line.px with the background (FCEUmm's Plinef).
template <bool kLatchesOnly = false>
NESLE_CUDA_FRAME_HD inline void replay_line(const ReplaySource& src, ReplayState& s, ReplayLine& line, int l,
                                            bool pixels) {
    const int start = l * 341 - 16;  // ResetRL: 16 dots before the line's dot 0
    line.firsttile = 0;
    line.outpos = 0;
    line.tofix = true;
    if (pixels) {
        for (int p = 0; p < 256; ++p) line.px[p] = 0xFF;
    }
    while (s.cursor < src.count && event_start(src, s.cursor) < start + 272) {
        const std::uint32_t kind = event_kind(src, s.cursor);
        if (kind != kEvDataWrite && kind != kEvPalette) {  // FCEUmm's $2007 write draws nothing first
            replay_update<kLatchesOnly>(src, s, line, event_end(src, s.cursor) - start, pixels);
        }
        replay_apply(src, s, s.cursor++, true);
    }
    replay_update<kLatchesOnly>(src, s, line, 272, pixels);  // EndRL
    if (line.tofix) {
        if (replay_rendering(s)) replay_vertical_increment(s);
        line.tofix = false;
    }
    if (pixels) replay_check_hit(s, line, 272);
}

// CopySprites onto line.px (sprites on at the line's end), then greyscale. Leaves the
// colour index (0-63) in each pixel and returns the line's emphasis (FCEUmm marks the
// line with the mask at its end).
NESLE_CUDA_FRAME_HD inline std::uint32_t replay_compose(ReplayState& s, ReplayLine& line) {
    if ((s.mask & 0x10u) != 0 && s.spork != 0) {
        s.spork = 0;
        const ReplaySprites& d = s.drawn;
        for (std::uint32_t xpix = (s.mask & 0x04u) != 0 ? 0u : 8u; xpix < 256u; ++xpix) {
            // The lowest-numbered opaque sprite owns the pixel, in front or behind.
            for (std::uint32_t n = 0; n < d.n; ++n) {
                const std::uint32_t k = xpix - d.x[n];
                if (k >= 8u) continue;
                const std::uint32_t bit = (d.attr[n] & 0x40u) != 0 ? k : 7u - k;
                const std::uint32_t c = ((d.lo[n] >> bit) & 1u) | (((d.hi[n] >> bit) & 1u) << 1);
                if (c == 0) continue;
                if ((d.attr[n] & 0x20u) == 0 || (line.px[xpix] & 0x40u) != 0) {
                    line.px[xpix] = s.spal[(d.attr[n] & 3u) * 4u + c];
                }
                break;
            }
        }
    }
    const std::uint32_t keep = (s.mask & 0x01u) != 0 && replay_rendering(s) ? 0x30u : 0x3Fu;
    for (int p = 0; p < 256; ++p) {
        line.px[p] = static_cast<std::uint8_t>(line.px[p] & keep);
    }
    return s.mask >> 5;
}

// The rest of line `l` after its pixels (DoLine after EndRL): sprite fetch for line l + 1,
// Fixit2 (dot 262), RefreshSprites (dot 325). `sprites` false skips the fetch where no
// later line can need it.
NESLE_CUDA_FRAME_HD inline void replay_end_of_line(const ReplaySource& src, ReplayState& s, int l, bool sprites) {
    const int start = l * 341 - 16;
    s.sphitx = 0x100;
    if (replay_rendering(s) && (sprites || src.b->cart.bank_kind == kBankingMmc2)) {  // FetchSpriteData
        const std::uint32_t c = s.ctrl;
        const std::uint32_t h = (c & 0x20u) != 0 ? 16u : 8u;
        const std::uint32_t vofs = (c & 0x08u) != 0 && (c & 0x20u) == 0 ? 0x1000u : 0u;
        ReplaySprites& f = s.fetched;
        f.n = 0;
        s.blurp = 0;
        for (std::uint32_t n = 0; n < 64u; ++n) {
            const std::uint8_t* spr = src.oam + 4u * n;
            const std::uint32_t t = static_cast<std::uint32_t>(l - static_cast<int>(spr[0]));
            if (t >= h) continue;
            if (f.n >= 8u) break;  // FCEUmm's 8-sprite limit (its overflow flag is not modelled)
            if (n == 0) s.blurp = 1;
            std::uint32_t vadr = h == 16u ? ((spr[1] & 1u) << 12) + ((spr[1] & 0xFEu) << 4) : (spr[1] << 4) + vofs;
            if ((spr[2] & 0x80u) != 0) {
                vadr += 7u;
                vadr -= t;
                vadr += (c & 0x20u) >> 1;
                vadr -= t & 8u;
            } else {
                vadr += t;
                vadr += t & 8u;
            }
            vadr &= 0x1FFFu;
            // Both bytes come through the bank the first one used: FCEUmm takes the CHR
            // pointer before the MMC2 hook can switch it.
            f.lo[f.n] = replay_chr(src, s, vadr);
            f.hi[f.n] = replay_chr(src, s, vadr + 8u);
            replay_mmc2_hook(src, s, vadr);
            f.attr[f.n] = spr[2];
            f.x[f.n] = spr[3];
            ++f.n;
        }
    }
    while (s.cursor < src.count && event_start(src, s.cursor) < start + 278) {
        replay_apply(src, s, s.cursor++, true);
    }
    if (replay_rendering(s)) {  // Fixit2: horizontal position from t
        s.v = static_cast<std::uint16_t>((s.v & 0xFBE0u) | (s.t & 0x041Fu));
    }
    while (s.cursor < src.count && event_start(src, s.cursor) < start + 341) {
        replay_apply(src, s, s.cursor++, true);
    }
    if ((s.mask & 0x10u) != 0) {  // RefreshSprites
        s.spork = 0;
        if (s.fetched.n != 0) {
            s.drawn = s.fetched;
            for (int i = 0; i < 16; ++i) s.spal[i] = static_cast<std::uint8_t>(s.pal[16 + i] & 0x3Fu);
            const std::uint32_t j = s.drawn.lo[0] | s.drawn.hi[0];
            if (s.blurp != 0 && s.hit == 0 && j != 0) {
                std::uint32_t data = j;
                if ((s.drawn.attr[0] & 0x40u) != 0) {
                    data = 0;
                    for (int k = 0; k < 8; ++k) data |= ((j >> k) & 1u) << (7 - k);
                }
                s.sphitx = s.drawn.x[0];
                s.sphitdata = static_cast<std::uint8_t>(data);
            }
            s.spork = 1;
        }
    }
}

// ------------------------------------------------- step-kernel uses

// The writing bank as a replay over the live memories (step kernel, mid-frame).
NESLE_CUDA_FRAME_HD NESLE_CUDA_FRAME_INLINE ReplaySource frame_live_source(const BatchBuffers& b,
                                                                           std::uint32_t env,
                                                                           std::uint8_t log_bank,
                                                                           std::uint16_t log_n) {
    return replay_source(b, env, frame_log_bank(b, env, (log_bank & 1u) ^ 1u), log_n,
                         b.ppu.nametable_ram + static_cast<std::uint64_t>(env) * kNametableRamBytes,
                         b.ppu.oam + static_cast<std::uint64_t>(env) * kOamBytes);
}

// FCEUmm's sprite-0 hit for this frame from the log so far. `dot` is the frame dot from
// which an instruction sees the flag, or, with no hit, the end of sprite 0's lines (later
// writes cannot add one after that), or 0 when sprite 0 is below the picture.
struct Sprite0Hit {
    std::uint32_t dot;
    bool hit;
};

// FCEUmm sets the flag in the RefreshLine of a PPU access (a $2002 read) at line pixel
// p, which draws the tiles up to p / 8 (one more while the sprite-0 check is pending and
// sprite 0 starts in the tile being drawn) and finds the hit pixel x once it is both
// drawn and left of p - 16. An access is stamped with its instruction's end, so a
// 4-cycle $2002 read that starts 12 dots earlier sees the flag.
NESLE_CUDA_FRAME_HD NESLE_CUDA_FRAME_INLINE int sprite0_first_seen(int x, int sphitx) {
    for (int p = x + 17;; ++p) {
        int last = p >> 3;
        if (sphitx < p - 16 && !(sphitx < (last - 2) * 8)) ++last;
        if (last > 34) last = 34;
        if (x < (last - 2) * 8) return p;
    }
}

NESLE_CUDA_FRAME_HD NESLE_CUDA_FRAME_COLD Sprite0Hit frame_sprite0_hit(const BatchBuffers& b,
                                                                     std::uint32_t env,
                                                                     std::uint8_t log_bank,
                                                                     std::uint16_t log_n) {
    const ReplaySource src = frame_live_source(b, env, log_bank, log_n);
    ReplayState s;
    replay_load_header(s, src.header);
    ReplayLine line;
    const int first = static_cast<int>(src.oam[0]) + 1;
    if (first >= 240) return Sprite0Hit{0, false};
    replay_begin_frame(src, s);
    const int last = first + 15 < 239 ? first + 15 : 239;
    for (int l = 0; l <= last; ++l) {
        const int sphitx = s.sphitx;
        replay_line(src, s, line, l, l >= first);
        if (s.hit != 0) {
            return Sprite0Hit{nesle_dot(l * 341 - 16 + sprite0_first_seen(s.hit_x, sphitx) - 12), true};
        }
        s.spork = 0;  // CopySprites ran (nothing it draws matters here)
        replay_end_of_line(src, s, l, l >= first - 1);
    }
    return Sprite0Hit{nesle_dot(last * 341 + 256), false};
}

// Vblank, MMC2: the latches the frame's own fetches left (FCEUmm flips them as it draws).
NESLE_CUDA_FRAME_HD NESLE_CUDA_FRAME_COLD void frame_resolve_mmc2(const BatchBuffers& b,
                                                                       std::uint32_t env,
                                                                       std::uint8_t log_bank,
                                                                       std::uint16_t log_n) {
    const ReplaySource src = frame_live_source(b, env, log_bank, log_n);
    ReplayState s;
    replay_load_header(s, src.header);
    ReplayLine line;
    replay_begin_frame(src, s);
    for (int l = 0; l < 240; ++l) {
        replay_line<true>(src, s, line, l, false);
        replay_end_of_line(src, s, l, true);
    }
    while (s.cursor < src.count) {
        replay_apply(src, s, s.cursor++, false);
    }
    std::uint8_t* r = paged::regs(b, env);
    for (std::uint32_t half = 0; half < 2u; ++half) {
        if (r[paged::kMmc2Latch0 + half] != s.latch[half]) {
            r[paged::kMmc2Latch0 + half] = s.latch[half];
            paged::set_chr4(b, env, half, r[2u * half + s.latch[half]]);
        }
    }
}

#undef NESLE_CUDA_FRAME_COLD

}  // namespace nesle::cuda

#undef NESLE_CUDA_FRAME_HD
#undef NESLE_CUDA_FRAME_INLINE
