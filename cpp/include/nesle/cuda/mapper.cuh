#pragma once

#include <cstdint>

#include "nesle/cuda/state.cuh"

#ifdef __CUDACC__
// The banking helpers sit in the PPU pattern-fetch path (one per pattern table
// byte per pixel) as well as the CPU bus, so they are force-inlined for the
// same reason batch_render.cuh does it: nvcc's heuristics otherwise leave a
// call in the innermost loop.
#define NESLE_CUDA_MAPPER_HD __host__ __device__
#define NESLE_CUDA_MAPPER_INLINE __forceinline__
#else
#define NESLE_CUDA_MAPPER_HD
#define NESLE_CUDA_MAPPER_INLINE inline
#endif

// Cartridge banking shared by the bus and the renderer. Mapper 0 (NROM) keeps
// its historical zero-indirection path: no per-env register load, no second
// dependent memory access. The window boards (UxROM, UN1ROM, BNROM) reduce to
// the shape described by CartridgeView (switchable window at the bottom of
// $8000-$FFFF, fixed window on top). The paged boards (CNROM, MMC1, MMC2, MMC3)
// resolve their registers into per-env 8 KB PRG / 1 KB CHR page tables, so
// their fetches are also one branch on the batch-uniform `bank_kind` rather
// than a per-mapper dispatch; only register writes dispatch per board.

namespace nesle::cuda {

// Compile-time cartridge specialization for the bus and PPU-fetch paths.
//
// bank_kind is batch-uniform, so the console step kernel can pick the banking
// shape once per launch and hand it down as a template argument instead of
// re-testing it on every PRG byte, CHR byte and nametable fetch. Measured on
// Super Mario Bros. (mapper 0): dispatching on the runtime field instead costs
// 34% of console-step throughput, because the compiler cannot drop the tests and
// the accessors then have to thread `env` through the innermost loop.
//
//   kBankModeRuntime  dispatch on cart.bank_kind (default, so callers that do
//                     not know the cartridge keep working unchanged)
//   kBankModeNrom     compile-time mapper 0: the pre-mapper expressions, no tests
//   kBankModeBanked   compile-time window board (UxROM/BNROM): no NROM test at all
//   kBankModePaged    compile-time paged board (CNROM/MMC1/MMC2/MMC3): page
//                     tables, runtime mirroring and the mapper IRQ line
enum BankMode : std::uint8_t {
    kBankModeRuntime = 0,
    kBankModeNrom = 1,
    kBankModeBanked = 2,
    kBankModePaged = 3,
};

// True when Mode proves the cartridge has no bank registers, so the NROM
// expressions can be emitted without a runtime test.
NESLE_CUDA_MAPPER_HD constexpr bool bank_mode_is_nrom(BankMode mode) noexcept {
    return mode == kBankModeNrom;
}

// True when the cartridge goes through the paged path. Compile-time for the
// three specialized modes, so the NROM and window instantiations never test it.
template <BankMode Mode>
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE bool cart_is_paged(const CartridgeView& cart) {
    if constexpr (Mode == kBankModePaged) {
        return true;
    } else if constexpr (Mode == kBankModeRuntime) {
        return cart.bank_kind >= kBankingFirstPaged;
    } else {
        return false;
    }
}

// ------------------------------------------------------------ paged boards
//
// CNROM, MMC1, MMC2 and MMC3 keep their registers in MapperStateSoA::regs and
// resolve them into an 8 KB PRG page table and a 1 KB CHR page table on every
// register write (the "sync"), so a fetch is the same two loads on every board.
// Register semantics follow FCEUmm (boards/datalatch.c, mmc1.c, mmc2and4.c,
// mmc3.c), the reference the reward functions were written against.
//
// regs layout per board:
//   CNROM  [0] CHR bank latch
//   MMC1   [0] control  [1] CHR 0  [2] CHR 1  [3] PRG  [4] shift value  [5] shift count
//   MMC2   [0..3] CHR $FD/0, $FE/0, $FD/1, $FE/1  [4] PRG  [5] mirroring  [6] latch 0  [7] latch 1
//   MMC3   [0..7] R0-R7  [8] bank select  [9] $A000  [10] $A001  [11] IRQ latch
//          [12] IRQ counter  [13] IRQ enabled  [14] IRQ reload pending
// The IRQ line itself is CpuStateSoA::irq_pending (level: it stays asserted
// until the board acknowledges it).

namespace paged {

// VRC2/4: [0] PRG 0 [1] PRG 1 [2] mirroring [3] PRG swap [4..19] CHR 0-7 (lo, hi byte) [20] IRQ latch
//         [21] IRQ control [22] IRQ counter [23,24] prescaler (int16, little endian)
// FME-7:  [0] command [1..8] CHR 0-7 [9] $6000 bank [10..12] PRG $8000/$A000/$C000 [13] mirroring
//         [14] IRQ control [15,16] IRQ counter (little endian)
constexpr std::uint32_t kVrcPrg0 = 0, kVrcPrg1 = 1, kVrcMirror = 2, kVrcSwap = 3, kVrcChr = 4, kVrcIrqLatch = 20,
                        kVrcIrqControl = 21, kVrcIrqCounter = 22, kVrcPrescaler = 23;
constexpr std::uint32_t kFmeCommand = 0, kFmeChr = 1, kFmePrg6 = 9, kFmePrg = 10, kFmeMirror = 13, kFmeIrqControl = 14,
                        kFmeIrqCounter = 15;

// VRC6: [0] PRG 16 KB [1] PRG 8 KB [2] control ($B003) [4..11] CHR 0-7 [20..24] IRQ as VRC4
// 40:   [0] $C000 bank [1] IRQ enabled [2,3] IRQ cycle count (little endian)
// 232:  [0] outer block [1] inner bank
constexpr std::uint32_t kVrc6Prg16 = 0, kVrc6Prg8 = 1, kVrc6Control = 2;

NESLE_CUDA_MAPPER_HD constexpr bool kind_is_vrc(std::uint8_t kind) {  // boards with the VRC4/VRC6 IRQ
    return (kind >= kBankingVrc21 && kind <= kBankingVrc25) || kind == kBankingVrc6a || kind == kBankingVrc6b;
}

constexpr std::uint32_t kMmc1Control = 0, kMmc1Chr0 = 1, kMmc1Chr1 = 2, kMmc1Prg = 3,
                        kMmc1Shift = 4, kMmc1ShiftCount = 5;
constexpr std::uint32_t kMmc2Prg = 4, kMmc2Mirroring = 5, kMmc2Latch0 = 6, kMmc2Latch1 = 7;
constexpr std::uint32_t kMmc3BankSelect = 8, kMmc3Mirroring = 9, kMmc3PrgRam = 10,
                        kMmc3IrqLatch = 11, kMmc3IrqCounter = 12, kMmc3IrqEnabled = 13,
                        kMmc3IrqReload = 14, kMmc3Outer = 15;  // [15]: 189's 32 KB PRG register, 187's $5000
constexpr std::uint32_t kM187Unlock = 16;  // 187: $8001 ignored until $8000 is written
// 114 (and 182, which FCEUmm's database runs as 114): [15] $5000-$7FFF register, [16] data write armed by $A000
constexpr std::uint32_t kM114Armed = 16;
// 121 (FCEUmm 121.c): [16..23] EXPREGS[0..7]
constexpr std::uint32_t kM121Ex = 16;
// 121's protection latch (FCEUmm 121.c Sync).
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE void m121_sync(std::uint8_t* ex) {
    switch (ex[5] & 0x3Fu) {
        case 0x20: case 0x29: case 0x2B: case 0x3C: case 0x3F: ex[7] = 1; ex[0] = ex[6]; break;
        case 0x26: ex[7] = 0; ex[0] = ex[6]; break;
        case 0x2C: ex[7] = 1; if (ex[6] != 0) ex[0] = ex[6]; break;
        case 0x28: ex[7] = 0; ex[1] = ex[6]; break;
        case 0x2A: ex[7] = 0; ex[2] = ex[6]; break;
        case 0x2F: break;
        default: ex[5] = 0; break;
    }
}

// RAMBO-1 (64): [16..26] R0-R9, RF  [27] command  [28] mirroring  [29] reload mode  [30] IRQ mode (1 = CPU cycles)
//               [31] IRQ counter  [32] IRQ enabled  [33] IRQ latch  [34] cycle prescaler (0..3)
constexpr std::uint32_t kRamboR = 16, kRamboCmd = 27, kRamboMirror = 28, kRamboRmode = 29, kRamboIrqMode = 30,
                        kRamboCount = 31, kRamboIrqOn = 32, kRamboLatch = 33, kRamboPre = 34;

NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE std::uint8_t* regs(const BatchBuffers& buffers,
                                                                 std::uint32_t env) {
    return buffers.mapper.regs + static_cast<std::uint64_t>(env) * kMapperRegBytes;
}

// slot: 0..3 for $8000/$A000/$C000/$E000. bank: 8 KB page number, wrapped to the image.
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE void set_prg8(const BatchBuffers& buffers,
                                                            std::uint32_t env,
                                                            std::uint32_t slot,
                                                            std::uint32_t bank) {
    buffers.mapper.prg_page[static_cast<std::uint64_t>(env) * kPrgPages + slot] =
        (bank & buffers.cart.prg_page_mask) << 13;
}

// slot: 0..7 for each 1 KB of $0000-$1FFF. bank: 1 KB page number, wrapped.
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE void set_chr1(const BatchBuffers& buffers,
                                                            std::uint32_t env,
                                                            std::uint32_t slot,
                                                            std::uint32_t bank) {
    buffers.mapper.chr_page[static_cast<std::uint64_t>(env) * kChrPages + slot] =
        (bank & buffers.cart.chr_page_mask) << 10;
}

// TQROM (119): a 1 KB page of the 8 KB CHR RAM on a CHR ROM board (kChrPageRam).
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE void set_chr1_ram(const BatchBuffers& buffers, std::uint32_t env,
                                                                std::uint32_t slot, std::uint32_t bank) {
    buffers.mapper.chr_page[static_cast<std::uint64_t>(env) * kChrPages + slot] = ((bank & 7u) << 10) | kChrPageRam;
}

NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE void set_prg16(const BatchBuffers& buffers,
                                                             std::uint32_t env,
                                                             std::uint32_t slot16,
                                                             std::uint32_t bank) {
    set_prg8(buffers, env, slot16 * 2u, bank * 2u);
    set_prg8(buffers, env, slot16 * 2u + 1u, bank * 2u + 1u);
}

NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE void set_chr4(const BatchBuffers& buffers,
                                                            std::uint32_t env,
                                                            std::uint32_t slot4,
                                                            std::uint32_t bank) {
    for (std::uint32_t i = 0; i < 4u; ++i) {
        set_chr1(buffers, env, slot4 * 4u + i, bank * 4u + i);
    }
}

NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE void set_prg32(const BatchBuffers& buffers,
                                                             std::uint32_t env,
                                                             std::uint32_t bank) {
    for (std::uint32_t i = 0; i < 4u; ++i) {
        set_prg8(buffers, env, i, bank * 4u + i);
    }
}

NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE void set_chr8(const BatchBuffers& buffers,
                                                            std::uint32_t env,
                                                            std::uint32_t bank) {
    for (std::uint32_t i = 0; i < 8u; ++i) {
        set_chr1(buffers, env, i, bank * 8u + i);
    }
}

NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE void set_mirroring(const BatchBuffers& buffers,
                                                                 std::uint32_t env,
                                                                 std::uint8_t page_map) {
    // A four-screen header is hard-wired; the board's mirroring register has no
    // effect (FCEUmm's mirrorhard).
    if (buffers.cart.nametable_arrangement == kNametableFourScreen) {
        return;
    }
    buffers.mapper.mirroring[env] = page_map;
}

// The header's arrangement as a page map. Four-screen folds to vertical, as
// mirror_nametable_address does: only 2 KB of CIRAM exists per env.
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE std::uint8_t header_page_map(const CartridgeView& cart) {
    return cart.nametable_arrangement == kNametableHorizontal ? kPageMapHorizontal : kPageMapVertical;
}

// JY Company (90, FCEUmm 90.c without the 209/211 nametable control): [0..3] PRG [4..11] CHR low [12..19] CHR high
//   [20..23] mode $D000-$D003 [24] IRQ mode [25] prescaler [26] counter [27] xor [28] IRQ enabled [29,30] multiplier
//   [31] $5803 [32] $6000 ROM mapped [33] $6000 bank
constexpr std::uint32_t kJyChrLo = 4, kJyChrHi = 12, kJyMode = 20, kJyIrqMode = 24, kJyPre = 25, kJyCount = 26,
                        kJyXor = 27, kJyIrqOn = 28, kJyMul = 29, kJyRegie = 31, kJy6000On = 32, kJy6000 = 33;
// 209 nametable control: [34..37] $B000-$B003 (names low) [38..41] $B004-$B007 (names high)
//   [42] bit q = bit 8 of quadrant q's CHR ROM nametable page (the low 8 bits are at kNtChrPage + q)
constexpr std::uint32_t kJyNames = 34, kJyNamesHi = 38, kJyNtHi = 42;
// 83 (FCEUmm yoko.c M83): [0..10] reg [11] mode [12] bank [13..16] $5100-$5103 [17] IRQ enabled
//   [18,19] IRQ counter (little endian) [20] 2 KB CHR seen [21] 1 KB CHR seen
constexpr std::uint32_t kM83Mode = 11, kM83Bank = 12, kM83Low = 13, kM83IrqOn = 17, kM83Count = 18, kM83Is2k = 20,
                        kM83Not2k = 21;

// Recompute the page tables and mirroring from the registers.
// ------------------------------------------------------------------ MMC5 (old FCEUmm mmc5.c)
// regs: [0..3] PRGBanks [4..11] CHRBanksA low [12..19] their bits 8-9 [20..23] CHRBanksB low [24..27] bits 8-9
// [28] WRAMPage [29,30] WRAMMaskEnable [31] mmc5ABMode [32] IRQScanline [33] IRQEnable [34] CHRMode [35] NTAMirroring
// [36] MMC5IRQR [37] MMC5LineCounter [38] mmc5psize [39] mmc5vsize [40,41] mul [42] $5130 [43] split mode
// [44] NTFill [45] ATFill (kNtChrPage slots, so the frame log carries them) [46] split scroll [47] split page.
constexpr std::uint32_t kM5Prg = 0, kM5ALo = 4, kM5AHi = 12, kM5BLo = 20, kM5BHi = 24, kM5WramPage = 28,
                        kM5WramMask = 29, kM5AB = 31, kM5IrqLine = 32, kM5IrqOn = 33, kM5ChrMode = 34, kM5Nta = 35,
                        kM5IrqStatus = 36, kM5LineCount = 37, kM5PSize = 38, kM5VSize = 39, kM5Mul = 40,
                        kM5Hi = 42, kM5Split = 43, kM5NtFill = 44, kM5AtFill = 45;

NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE std::uint8_t* mmc5_ext(const BatchBuffers& buffers, std::uint32_t env) {
    return buffers.mapper.ext_ram + static_cast<std::uint64_t>(env) * buffers.cart.ext_ram_bytes;
}

// MMC5WRAMIndex[v & 7] for the cart's WRAM size: the 8 KB WRAM bank, or 255 (none).
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE std::uint32_t mmc5_wram_index(const CartridgeView& cart, std::uint32_t v) {
    const std::uint32_t x = v & 7u;
    switch ((cart.ext_ram_bytes - kMmc5ExWram) >> 13) {  // MMC5WRAMsize = KB / 8
        case 1: return x > 3u ? 255u : 0u;
        case 2: return (x & 4u) >> 2;
        case 4: return x > 3u ? 255u : (x & 3u);
        case 8: return x;
        default: return 255u;
    }
}

// The 1 KB CHR page of slot `slot` in set A (sprites) or B (background), as MMC5CHRA / MMC5CHRB map them.
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE std::uint32_t mmc5_chr_page(const std::uint16_t* a, const std::uint16_t* b,
                                                                         std::uint32_t vsize, bool set_b,
                                                                         std::uint32_t slot) {
    switch (vsize & 3u) {
        case 0: return (set_b ? b[3] : a[7]) * 8u + slot;
        case 1: return (set_b ? b[3] : a[(slot >> 2) * 4u + 3u]) * 4u + (slot & 3u);
        case 2: return (set_b ? b[((slot >> 1) & 1u) * 2u + 1u] : a[(slot >> 1) * 2u + 1u]) * 2u + (slot & 1u);
        default: return set_b ? b[slot & 3u] : a[slot];
    }
}

NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE void mmc5_banks(const std::uint8_t* r, std::uint16_t* a, std::uint16_t* b) {
    for (std::uint32_t i = 0; i < 8u; ++i) a[i] = static_cast<std::uint16_t>(r[kM5ALo + i] | (r[kM5AHi + i] << 8));
    for (std::uint32_t i = 0; i < 4u; ++i) b[i] = static_cast<std::uint16_t>(r[kM5BLo + i] | (r[kM5BHi + i] << 8));
}

NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE void sync(const BatchBuffers& buffers, std::uint32_t env) {
    const std::uint8_t* r = regs(buffers, env);
    switch (buffers.cart.bank_kind) {
        case kBankingCnrom: {
            for (std::uint32_t slot = 0; slot < 4u; ++slot) {
                set_prg8(buffers, env, slot, slot);  // 16 KB images mirror through the mask
            }
            for (std::uint32_t slot = 0; slot < 8u; ++slot) {
                set_chr1(buffers, env, slot, static_cast<std::uint32_t>(r[0]) * 8u + slot);
            }
            break;
        }
        case kBankingMmc1: {
            const std::uint32_t control = r[kMmc1Control];
            if (buffers.cart.mapper == 105) {  // NWC 1990 (mmc1.c NWC hooks): r[6] = NWCRec, the last CHR hook value
                if ((r[6] & 0x08u) != 0) {
                    const std::uint32_t prg = r[kMmc1Prg];
                    switch (control & 0x0Cu) {
                        case 0x0Cu: set_prg16(buffers, env, 0, 8u | (prg & 7u)); set_prg16(buffers, env, 1, 0x0Fu); break;
                        case 0x08u: set_prg16(buffers, env, 1, 8u | (prg & 7u)); set_prg16(buffers, env, 0, 8u); break;
                        default:
                            set_prg16(buffers, env, 0, 8u | (prg & 6u));
                            set_prg16(buffers, env, 1, 8u | ((prg & 6u) + 1u));
                            break;
                    }
                } else {
                    set_prg32(buffers, env, (r[6] >> 1) & 3u);
                }
                set_chr8(buffers, env, 0);
                switch (control & 0x03u) {
                    case 0: set_mirroring(buffers, env, kPageMapSingleLow); break;
                    case 1: set_mirroring(buffers, env, kPageMapSingleHigh); break;
                    case 2: set_mirroring(buffers, env, kPageMapVertical); break;
                    default: set_mirroring(buffers, env, kPageMapHorizontal); break;
                }
                break;
            }
            // PRG. offs is SUROM's 256 KB outer bank (CHR 0 bit 4); the page mask
            // drops it on smaller images, as FCEUmm's PRGmask16 does.
            const std::uint32_t offs = r[kMmc1Chr0] & 0x10u;
            const std::uint32_t prg = r[kMmc1Prg];
            switch (control & 0x0Cu) {
                case 0x0Cu:
                    set_prg16(buffers, env, 0, prg + offs);
                    set_prg16(buffers, env, 1, 0x0Fu + offs);
                    break;
                case 0x08u:
                    set_prg16(buffers, env, 1, prg + offs);
                    set_prg16(buffers, env, 0, offs);
                    break;
                default:
                    set_prg16(buffers, env, 0, (prg & ~1u) + offs);
                    set_prg16(buffers, env, 1, (prg & ~1u) + offs + 1u);
                    break;
            }
            // CHR: two 4 KB banks, or one 8 KB bank.
            if ((control & 0x10u) != 0) {
                set_chr4(buffers, env, 0, r[kMmc1Chr0]);
                set_chr4(buffers, env, 1, r[kMmc1Chr1]);
            } else {
                set_chr4(buffers, env, 0, r[kMmc1Chr0] & 0xFEu);
                set_chr4(buffers, env, 1, r[kMmc1Chr0] | 0x01u);
            }
            switch (control & 0x03u) {
                case 0: set_mirroring(buffers, env, kPageMapSingleLow); break;
                case 1: set_mirroring(buffers, env, kPageMapSingleHigh); break;
                case 2: set_mirroring(buffers, env, kPageMapVertical); break;
                default: set_mirroring(buffers, env, kPageMapHorizontal); break;
            }
            break;
        }
        case kBankingMmc2: {
            if (buffers.cart.mapper == 10) {  // MMC4: 16 KB PRG, last 16 KB fixed; latches as MMC2 (FCEUmm's hook)
                set_prg16(buffers, env, 0, r[kMmc2Prg]);
                set_prg16(buffers, env, 1, ~0u);
            } else {
                set_prg8(buffers, env, 0, r[kMmc2Prg]);
                set_prg8(buffers, env, 1, ~2u);
                set_prg8(buffers, env, 2, ~1u);
                set_prg8(buffers, env, 3, ~0u);
            }
            set_chr4(buffers, env, 0, r[r[kMmc2Latch0] & 1u]);
            set_chr4(buffers, env, 1, r[2u + (r[kMmc2Latch1] & 1u)]);
            set_mirroring(buffers, env, r[kMmc2Mirroring] != 0 ? kPageMapVertical : kPageMapHorizontal);
            break;
        }
        case kBankingMmc3: {
            if (buffers.cart.mapper == 64 || buffers.cart.mapper == 158) {
                const std::uint8_t* q = r + kRamboR;
                if ((r[kRamboCmd] & 0x20u) != 0) {
                    set_chr1(buffers, env, 0, q[0]);
                    set_chr1(buffers, env, 1, q[8]);
                    set_chr1(buffers, env, 2, q[1]);
                    set_chr1(buffers, env, 3, q[9]);
                } else {
                    set_chr1(buffers, env, 0, q[0] & 0xFEu);
                    set_chr1(buffers, env, 1, q[0] | 1u);
                    set_chr1(buffers, env, 2, q[1] & 0xFEu);
                    set_chr1(buffers, env, 3, q[1] | 1u);
                }
                for (std::uint32_t i = 0; i < 4u; ++i) set_chr1(buffers, env, 4u + i, q[2u + i]);
                set_prg8(buffers, env, 0, q[6]);
                set_prg8(buffers, env, 1, q[7]);
                set_prg8(buffers, env, 2, q[10]);
                set_prg8(buffers, env, 3, ~0u);
                if (buffers.cart.mapper == 158) {  // M158PPU: nametable q follows bit 7 of the bank in CHR slot q
                    const std::uint32_t lo[4] = {(r[kRamboCmd] & 0x20u) != 0 ? q[0] : q[0] & 0xFEu,
                                                 (r[kRamboCmd] & 0x20u) != 0 ? q[8] : q[0] | 1u,
                                                 (r[kRamboCmd] & 0x20u) != 0 ? q[1] : q[1] & 0xFEu,
                                                 (r[kRamboCmd] & 0x20u) != 0 ? q[9] : q[1] | 1u};
                    std::uint8_t map = 0;
                    for (std::uint32_t i = 0; i < 4u; ++i) map |= static_cast<std::uint8_t>(((lo[i] >> 7) & 1u) << i);
                    set_mirroring(buffers, env, map);
                    break;
                }
                set_mirroring(buffers, env, r[kRamboMirror] != 0 ? kPageMapVertical : kPageMapHorizontal);
                break;
            }
            const std::uint32_t select = r[kMmc3BankSelect];
            // FCEUmm GENPWRAP keeps 7 bits; ~1 is the second-last page, ~0 the last.
            const std::uint32_t r6 = r[6] & 0x7Fu;
            if ((select & 0x40u) != 0) {
                set_prg8(buffers, env, 2, r6);
                set_prg8(buffers, env, 0, 0x7Eu);
            } else {
                set_prg8(buffers, env, 0, r6);
                set_prg8(buffers, env, 2, 0x7Eu);
            }
            set_prg8(buffers, env, 1, r[7] & 0x7Fu);
            set_prg8(buffers, env, 3, 0x7Fu);
            if (buffers.cart.mapper == 245) {  // M245PW: PRG A19 from bit 1 of the last CHR register written
                const std::uint32_t hi = (r[kMmc3Outer] & 2u) << 5;
                set_prg8(buffers, env, (select & 0x40u) != 0 ? 2u : 0u, (r[6] & 0x3Fu) | hi);
                set_prg8(buffers, env, (select & 0x40u) != 0 ? 0u : 2u, 0x3Eu | hi);
                set_prg8(buffers, env, 1, (r[7] & 0x3Fu) | hi);
                set_prg8(buffers, env, 3, 0x3Fu | hi);
            }
            if (buffers.cart.mapper == 189) set_prg32(buffers, env, r[kMmc3Outer] & 7u);  // M189PW: 32 KB from $4120
            if (buffers.cart.mapper == 187 && (r[kMmc3Outer] & 0x80u) != 0) {  // M187PW: $5000/$6000 PRG override
                const std::uint32_t bank = r[kMmc3Outer] & 0x1Fu;
                if ((r[kMmc3Outer] & 0x20u) != 0) {
                    set_prg32(buffers, env, (r[kMmc3Outer] & 0x40u) != 0 ? bank >> 2 : bank >> 1);
                } else {
                    set_prg16(buffers, env, 0, bank);
                    set_prg16(buffers, env, 1, bank);
                }
            }
            // CHR: 2 KB banks R0/R1 and 1 KB banks R2-R5; bit 7 swaps the halves.
            const std::uint32_t flip = (select & 0x80u) != 0 ? 4u : 0u;
            set_chr1(buffers, env, 0u ^ flip, r[0] & 0xFEu);
            set_chr1(buffers, env, 1u ^ flip, r[0] | 0x01u);
            set_chr1(buffers, env, 2u ^ flip, r[1] & 0xFEu);
            set_chr1(buffers, env, 3u ^ flip, r[1] | 0x01u);
            set_chr1(buffers, env, 4u ^ flip, r[2]);
            set_chr1(buffers, env, 5u ^ flip, r[3]);
            set_chr1(buffers, env, 6u ^ flip, r[4]);
            set_chr1(buffers, env, 7u ^ flip, r[5]);
            if (buffers.cart.mapper == 121) {  // M121PW / M121CW
                const std::uint8_t* ex = r + kM121Ex;
                const std::uint32_t hi = (ex[3] & 0x80u) >> 2;
                set_prg8(buffers, env, 0, ((select & 0x40u) != 0 ? 0x1Eu : r[6] & 0x1Fu) | hi);
                if ((ex[5] & 0x3Fu) != 0) {
                    set_prg8(buffers, env, 1, ex[2] | hi);
                    set_prg8(buffers, env, 2, ex[1] | hi);
                    set_prg8(buffers, env, 3, ex[0] | hi);
                } else {
                    set_prg8(buffers, env, 1, (r[7] & 0x1Fu) | hi);
                    set_prg8(buffers, env, 2, ((select & 0x40u) != 0 ? r[6] & 0x1Fu : 0x1Eu) | hi);
                    set_prg8(buffers, env, 3, 0x1Fu | hi);
                }
            }
            if (buffers.cart.mapper == 187 || buffers.cart.mapper == 121) {  // M187CW / M121CW: the 2 KB half reads CHR A18 high
                set_chr1(buffers, env, 0u ^ flip, (r[0] & 0xFEu) | 0x100u);
                set_chr1(buffers, env, 1u ^ flip, r[0] | 0x101u);
                set_chr1(buffers, env, 2u ^ flip, (r[1] & 0xFEu) | 0x100u);
                set_chr1(buffers, env, 3u ^ flip, r[1] | 0x101u);
            }
            if (buffers.cart.mapper == 47) {  // M47PW / M47CW: 128 KB PRG and CHR blocks from $6000 bit 0
                const std::uint32_t ex = r[kMmc3Outer] & 1u;
                const std::uint32_t prg[4] = {(select & 0x40u) != 0 ? 0xFEu : r[6], r[7], (select & 0x40u) != 0 ? r[6] : 0xFEu, 0xFFu};
                for (std::uint32_t i = 0; i < 4u; ++i) set_prg8(buffers, env, i, (prg[i] & 0x0Fu) | (ex << 4));
                const std::uint32_t chr[8] = {r[0] & 0xFEu, r[0] | 1u, r[1] & 0xFEu, r[1] | 1u, r[2], r[3], r[4], r[5]};
                for (std::uint32_t i = 0; i < 8u; ++i) set_chr1(buffers, env, i ^ flip, (chr[i] & 0x7Fu) | (ex << 7));
            }
            if ((buffers.cart.mapper == 114 || buffers.cart.mapper == 182) && (r[kMmc3Outer] & 0x80u) != 0) {
                set_prg16(buffers, env, 0, r[kMmc3Outer] & 0x0Fu);  // M114PWRAP
                set_prg16(buffers, env, 1, r[kMmc3Outer] & 0x0Fu);
            }
            if (buffers.cart.mapper == 119) {  // TQWRAP: bank bit 6 picks CHR RAM, bits 0-5 the bank
                const std::uint32_t banks[8] = {r[0] & 0xFEu, r[0] | 1u, r[1] & 0xFEu, r[1] | 1u, r[2], r[3], r[4], r[5]};
                for (std::uint32_t i = 0; i < 8u; ++i) {
                    if ((banks[i] & 0x40u) != 0) {
                        set_chr1_ram(buffers, env, i ^ flip, banks[i]);
                    } else {
                        set_chr1(buffers, env, i ^ flip, banks[i] & 0x3Fu);
                    }
                }
            }
            if (buffers.cart.mapper == 74) {  // M74CW: banks 8 and 9 are the 2 KB CHR RAM (in ext_ram, kChrPageRam)
                const std::uint32_t banks[8] = {r[0] & 0xFEu, r[0] | 1u, r[1] & 0xFEu, r[1] | 1u, r[2], r[3], r[4], r[5]};
                for (std::uint32_t i = 0; i < 8u; ++i) {
                    if (banks[i] == 8u || banks[i] == 9u) set_chr1_ram(buffers, env, i ^ flip, banks[i] & 1u);
                }
            }
            if (buffers.cart.mapper == 118) {
                // TxSROM: CHR A17 (bit 7) of the bank in each 1 KB of $0000-$0FFF drives CIRAM A10 for the
                // nametable at the same index; the bit is not a CHR address line.
                const std::uint32_t lo[4] = {r[0] & 0xFEu, r[0] | 1u, r[1] & 0xFEu, r[1] | 1u};
                const std::uint32_t hi[4] = {r[2], r[3], r[4], r[5]};
                std::uint8_t map = 0;
                for (std::uint32_t q = 0; q < 4u; ++q) {
                    map |= static_cast<std::uint8_t>((((flip ? hi[q] : lo[q]) >> 7) & 1u) << q);
                }
                for (std::uint32_t slot = 0; slot < 8u; ++slot) {
                    const std::uint32_t bank = slot < 4u ? lo[slot] : hi[slot - 4u];
                    set_chr1(buffers, env, slot ^ flip, bank & 0x7Fu);
                }
                set_mirroring(buffers, env, map);
                break;
            }
            set_mirroring(buffers, env,
                          (r[kMmc3Mirroring] & 1u) != 0 ? kPageMapHorizontal : kPageMapVertical);
            break;
        }
        // ---- latch boards: r[0] = latch, r[1] = one-screen page (Camerica)
        case kBankingAxrom:
            set_prg32(buffers, env, r[0] & 0x0Fu);
            set_chr8(buffers, env, 0);
            set_mirroring(buffers, env, (r[0] & 0x10u) != 0 ? kPageMapSingleHigh : kPageMapSingleLow);
            break;
        case kBankingColorDreams:
            set_prg32(buffers, env, r[0] & 0x0Fu);
            set_chr8(buffers, env, r[0] >> 4);
            break;
        case kBankingGxrom:
            set_prg32(buffers, env, (r[0] >> 4) & 0x0Fu);
            set_chr8(buffers, env, r[0] & 0x0Fu);
            break;
        case kBankingCamerica:
            set_prg16(buffers, env, 0, r[0]);
            set_prg16(buffers, env, 1, 0xFFu);
            set_chr8(buffers, env, 0);
            if (r[2] != 0) {  // only once the game wrote $9000 (BF9097): else the header's mirroring
                set_mirroring(buffers, env, r[1] != 0 ? kPageMapSingleHigh : kPageMapSingleLow);
            }
            break;
        case kBankingNina0306:
            set_prg32(buffers, env, (r[0] >> 3) & 1u);
            set_chr8(buffers, env, r[0] & 7u);
            break;
        case kBankingHes:
            set_prg32(buffers, env, (r[0] >> 3) & 7u);
            set_chr8(buffers, env, ((r[0] >> 3) & 8u) | (r[0] & 7u));
            set_mirroring(buffers, env, (r[0] & 0x80u) != 0 ? kPageMapVertical : kPageMapHorizontal);
            break;
        case kBankingJaleco140:
            set_prg32(buffers, env, (r[0] >> 4) & 3u);
            set_chr8(buffers, env, r[0] & 0x0Fu);
            break;
        case kBankingBitCorp38:
            set_prg32(buffers, env, r[0] & 3u);
            set_chr8(buffers, env, (r[0] >> 2) & 3u);
            break;
        case kBankingJaleco87:
            set_prg32(buffers, env, 0);
            set_chr8(buffers, env, ((r[0] >> 1) & 1u) | ((r[0] << 1) & 2u));
            break;
        case kBankingVrc21: case kBankingVrc22: case kBankingVrc23: case kBankingVrc25: {
            if ((r[kVrcSwap] & 2u) != 0) {
                set_prg8(buffers, env, 0, ~1u);
                set_prg8(buffers, env, 2, r[kVrcPrg0]);
            } else {
                set_prg8(buffers, env, 0, r[kVrcPrg0]);
                set_prg8(buffers, env, 2, ~1u);
            }
            set_prg8(buffers, env, 1, r[kVrcPrg1]);
            set_prg8(buffers, env, 3, ~0u);
            for (std::uint32_t i = 0; i < 8u; ++i) {
                std::uint32_t bank = r[kVrcChr + 2u * i] | (static_cast<std::uint32_t>(r[kVrcChr + 2u * i + 1u]) << 4);
                if (buffers.cart.bank_kind == kBankingVrc22) bank >>= 1;
                set_chr1(buffers, env, i, bank);
            }
            switch (r[kVrcMirror] & (buffers.cart.bank_kind == kBankingVrc22 ? 1u : 3u)) {
                case 0: set_mirroring(buffers, env, kPageMapVertical); break;
                case 1: set_mirroring(buffers, env, kPageMapHorizontal); break;
                case 2: set_mirroring(buffers, env, kPageMapSingleLow); break;
                default: set_mirroring(buffers, env, kPageMapSingleHigh); break;
            }
            break;
        }
        case kBankingVrc6a: case kBankingVrc6b:
            set_prg16(buffers, env, 0, r[kVrc6Prg16]);
            set_prg8(buffers, env, 2, r[kVrc6Prg8]);
            set_prg8(buffers, env, 3, ~0u);
            for (std::uint32_t i = 0; i < 8u; ++i) set_chr1(buffers, env, i, r[4u + i]);
            switch ((r[kVrc6Control] >> 2) & 3u) {
                case 0: set_mirroring(buffers, env, kPageMapVertical); break;
                case 1: set_mirroring(buffers, env, kPageMapHorizontal); break;
                case 2: set_mirroring(buffers, env, kPageMapSingleLow); break;
                default: set_mirroring(buffers, env, kPageMapSingleHigh); break;
            }
            break;
        case kBankingSmb2j40:
            set_prg8(buffers, env, 0, 4);
            set_prg8(buffers, env, 1, 5);
            set_prg8(buffers, env, 2, r[0] & 7u);
            set_prg8(buffers, env, 3, 7);
            set_chr8(buffers, env, 0);
            break;
        case kBankingQuattro:
            set_prg16(buffers, env, 0, (r[0] & 0x0Cu) | (r[1] & 3u));
            set_prg16(buffers, env, 1, (r[0] & 0x0Cu) | 3u);
            set_chr8(buffers, env, 0);
            break;
        case kBankingNina001:
            set_prg32(buffers, env, r[0]);
            set_chr4(buffers, env, 0, r[1]);
            set_chr4(buffers, env, 1, r[2]);
            break;
        case kBankingM15: {  // r[0] data, r[1] address bits 0-1
            const std::uint32_t bank = (r[0] & 0x3Fu) << 1;
            switch (r[1] & 3u) {
                case 0:
                    for (std::uint32_t i = 0; i < 4u; ++i) set_prg8(buffers, env, i, bank + i);
                    break;
                case 2:
                    for (std::uint32_t i = 0; i < 4u; ++i) set_prg8(buffers, env, i, bank | (r[0] >> 7));
                    break;
                default: {
                    const std::uint32_t hi = (r[1] & 2u) == 0 ? (bank | 0x0Eu) : bank;
                    set_prg8(buffers, env, 0, bank);
                    set_prg8(buffers, env, 1, bank + 1u);
                    set_prg8(buffers, env, 2, hi);
                    set_prg8(buffers, env, 3, hi + 1u);
                    break;
                }
            }
            set_chr8(buffers, env, 0);
            set_mirroring(buffers, env, (r[0] & 0x40u) != 0 ? kPageMapHorizontal : kPageMapVertical);
            break;
        }
        case kBankingIrem32: {  // [0] PRG 0 [1] mirroring/swap [2] PRG 1 [3..10] CHR
            const bool swap = (r[1] & 2u) != 0;
            set_prg8(buffers, env, swap ? 2u : 0u, r[0]);
            set_prg8(buffers, env, 1, r[2]);
            set_prg8(buffers, env, swap ? 0u : 2u, ~1u);
            set_prg8(buffers, env, 3, ~0u);
            for (std::uint32_t i = 0; i < 8u; ++i) set_chr1(buffers, env, i, r[3u + i]);
            set_mirroring(buffers, env, (r[1] & 1u) != 0 ? kPageMapHorizontal : kPageMapVertical);
            break;
        }
        case kBankingNamco88:  // [0..7] R0-R7 [8] command
            set_chr1(buffers, env, 0, r[0] & 0xFEu);
            set_chr1(buffers, env, 1, r[0] | 1u);
            set_chr1(buffers, env, 2, r[1] & 0xFEu);
            set_chr1(buffers, env, 3, r[1] | 1u);
            for (std::uint32_t i = 0; i < 4u; ++i) set_chr1(buffers, env, 4u + i, r[2u + i] | 0x40u);
            set_prg8(buffers, env, 0, r[6]);
            set_prg8(buffers, env, 1, r[7]);
            set_prg8(buffers, env, 2, ~1u);
            set_prg8(buffers, env, 3, ~0u);
            break;
        case kBankingM91: {  // [0..3] CHR 2 KB [4,5] PRG [6] IRQ on [7] IRQ count [8] outer bank
            const std::uint32_t outer = r[8];
            set_prg8(buffers, env, 0, r[4] | ((outer & 6u) << 3));
            set_prg8(buffers, env, 1, r[5] | ((outer & 6u) << 3));
            set_prg8(buffers, env, 2, 0x0Eu | ((outer & 6u) << 3));
            set_prg8(buffers, env, 3, 0x0Fu | ((outer & 6u) << 3));
            for (std::uint32_t i = 0; i < 4u; ++i) {
                const std::uint32_t bank = r[i] | ((outer & 1u) << 8);
                set_chr1(buffers, env, 2u * i, bank * 2u);
                set_chr1(buffers, env, 2u * i + 1u, bank * 2u + 1u);
            }
            break;
        }
        case kBankingM42:  // [0] $6000 bank [1] CHR [2] mirroring (1 = vertical) [3] IRQ on [4,5] IRQ count
            set_prg32(buffers, env, ~0u);
            set_chr8(buffers, env, r[1]);
            set_mirroring(buffers, env, r[2] != 0 ? kPageMapVertical : kPageMapHorizontal);
            break;
        case kBankingM228: {  // [0,1] address latch (lo, hi) [2] data latch [3..6] $5000 nibbles
            const std::uint32_t areg = r[0] | (static_cast<std::uint32_t>(r[1]) << 8);
            std::uint32_t page = (areg >> 7) & 0x3Fu;
            if ((page & 0x30u) == 0x30u) page -= 0x10u;
            const std::uint32_t lo = (page << 1) + (((areg >> 6) & 1u) & ((areg >> 5) & 1u));
            set_prg16(buffers, env, 0, lo);
            set_prg16(buffers, env, 1, lo + (((areg >> 5) & 1u) ^ 1u));
            set_chr8(buffers, env, (r[2] & 3u) | ((areg & 0x0Fu) << 2));
            set_mirroring(buffers, env, ((areg >> 13) & 1u) != 0 ? kPageMapHorizontal : kPageMapVertical);
            break;
        }
        case kBankingM83: {
            const std::uint32_t mode = r[kM83Mode], bank = r[kM83Bank];
            switch (mode & 3u) {
                case 0: set_mirroring(buffers, env, kPageMapVertical); break;
                case 1: set_mirroring(buffers, env, kPageMapHorizontal); break;
                case 2: set_mirroring(buffers, env, kPageMapSingleLow); break;
                default: set_mirroring(buffers, env, kPageMapSingleHigh); break;
            }
            if (r[kM83Is2k] != 0 && r[kM83Not2k] == 0) {
                const std::uint32_t q[4] = {r[0], r[1], r[6], r[7]};
                for (std::uint32_t i = 0; i < 4u; ++i) {
                    set_chr1(buffers, env, 2u * i, q[i] * 2u);
                    set_chr1(buffers, env, 2u * i + 1u, q[i] * 2u + 1u);
                }
            } else {
                for (std::uint32_t i = 0; i < 8u; ++i) set_chr1(buffers, env, i, r[i] | ((bank & 0x30u) << 4));
            }
            if ((mode & 0x40u) != 0) {
                set_prg16(buffers, env, 0, bank & 0x3Fu);
                set_prg16(buffers, env, 1, (bank & 0x30u) | 0x0Fu);
            } else {
                set_prg8(buffers, env, 0, r[8]);
                set_prg8(buffers, env, 1, r[9]);
                set_prg8(buffers, env, 2, r[10]);
                set_prg8(buffers, env, 3, ~0u);
            }
            break;
        }
        case kBankingJy90: {  // tekprom / tekvrom / mira
            const std::uint32_t m0 = r[kJyMode], m3 = r[kJyMode + 3];
            const std::uint32_t bm = (m3 & 6u) << 5;
            const std::uint8_t* p = r;
            switch (m0 & 7u) {
                case 0: set_prg32(buffers, env, 0x0Fu | ((m3 & 6u) << 3)); break;
                case 1:
                    set_prg16(buffers, env, 0, (p[1] & 0x1Fu) | ((m3 & 6u) << 4));
                    set_prg16(buffers, env, 1, 0x1Fu | ((m3 & 6u) << 4));
                    break;
                case 4: set_prg32(buffers, env, (p[3] & 0x0Fu) | ((m3 & 6u) << 3)); break;
                case 5:
                    set_prg16(buffers, env, 0, (p[1] & 0x1Fu) | ((m3 & 6u) << 4));
                    set_prg16(buffers, env, 1, (p[3] & 0x1Fu) | ((m3 & 6u) << 4));
                    break;
                default:  // 2, 3, 6, 7 (the bit reversal of 3 and 7 is not implemented in FCEUmm either)
                    for (std::uint32_t i = 0; i < 3u; ++i) set_prg8(buffers, env, i, (p[i] & 0x3Fu) | bm);
                    set_prg8(buffers, env, 3, (((m0 & 4u) != 0 ? p[3] : 0xFFu) & 0x3Fu) | bm);
                    break;
            }
            std::uint32_t cbank = 0, cmask = 0xFFFFu;
            if ((m3 & 0x20u) == 0) {
                cbank = (m3 & 1u) | ((m3 & 0x18u) >> 2);
                const std::uint32_t shift = 5u + ((m0 >> 3) & 3u);
                cbank <<= shift;
                cmask = (1u << shift) - 1u;
            }
            auto chr = [&](std::uint32_t i) {
                return ((r[kJyChrLo + i] | (static_cast<std::uint32_t>(r[kJyChrHi + i]) << 8)) & cmask) | cbank;
            };
            switch (m0 & 0x18u) {
                case 0x00u: set_chr8(buffers, env, chr(0)); break;
                case 0x08u:  // latch registers stay at 0 / 4 on mapper 90
                    set_chr4(buffers, env, 0, chr(0));
                    set_chr4(buffers, env, 1, chr(4));
                    break;
                case 0x10u:
                    for (std::uint32_t i = 0; i < 8u; i += 2u) {
                        set_chr1(buffers, env, i, chr(i) * 2u);
                        set_chr1(buffers, env, i + 1u, chr(i) * 2u + 1u);
                    }
                    break;
                default:
                    for (std::uint32_t i = 0; i < 8u; ++i) set_chr1(buffers, env, i, chr(i));
                    break;
            }
            if (buffers.cart.mapper == 209 && (m0 & 0x20u) != 0) {  // mira: 209's nametable control
                std::uint8_t map = 0;  // the CHR ROM pages are set by jy209_names
                for (std::uint32_t q = 0; q < 4u; ++q) {
                    const std::uint32_t name = r[kJyNames + q];
                    if ((m0 & 0x40u) != 0 || (r[kJyMode + 1] & 0x80u) != (name & 0x80u)) {  // CHR ROM nametable
                        map = static_cast<std::uint8_t>(map | (1u << (4u + q)));
                    } else {
                        map = static_cast<std::uint8_t>(map | ((name & 1u) << q));
                    }
                }
                set_mirroring(buffers, env, map);
                break;
            }
            switch (r[kJyMode + 1] & 3u) {
                case 0: set_mirroring(buffers, env, kPageMapVertical); break;
                case 1: set_mirroring(buffers, env, kPageMapHorizontal); break;
                case 2: set_mirroring(buffers, env, kPageMapSingleLow); break;
                default: set_mirroring(buffers, env, kPageMapSingleHigh); break;
            }
            break;
        }
        case kBankingVrc1:  // [0..2] PRG [3,4] CHR [5] mode
            set_prg8(buffers, env, 0, r[0]);
            set_prg8(buffers, env, 1, r[1]);
            set_prg8(buffers, env, 2, r[2]);
            set_prg8(buffers, env, 3, ~0u);
            set_chr4(buffers, env, 0, r[3] | ((r[5] & 2u) << 3));
            set_chr4(buffers, env, 1, r[4] | ((r[5] & 4u) << 2));
            set_mirroring(buffers, env, (r[5] & 1u) != 0 ? kPageMapHorizontal : kPageMapVertical);
            break;
        case kBankingM246:  // [0..3] PRG [4..7] CHR 2 KB
            for (std::uint32_t i = 0; i < 4u; ++i) {
                set_prg8(buffers, env, i, r[i]);
                set_chr1(buffers, env, 2u * i, r[4u + i] * 2u);
                set_chr1(buffers, env, 2u * i + 1u, r[4u + i] * 2u + 1u);
            }
            break;
        case kBankingSa72008:  // [0] latch
            if (buffers.cart.mapper == 149) {  // SA0036 (SA72007Synco): CHR from bit 7
                set_prg32(buffers, env, 0);
                set_chr8(buffers, env, r[0] >> 7);
                break;
            }
            set_prg32(buffers, env, (r[0] >> 2) & 1u);
            set_chr8(buffers, env, r[0] & 3u);
            break;
        case kBankingM243:  // [0..3] latch [4] command
            set_prg32(buffers, env, r[0]);
            set_chr8(buffers, env, r[1]);
            set_mirroring(buffers, env, (r[2] & 1u) != 0 ? kPageMapHorizontal : kPageMapVertical);
            break;
        case kBankingM188: {  // [0] latch
            const std::uint32_t prg16_count = (buffers.cart.prg_page_mask + 1u) >> 1;  // FCEUmm ROM_size, 16 KB units
            if (r[0] != 0) {
                set_prg16(buffers, env, 0, (r[0] & 0x10u) != 0 ? (r[0] & 7u) : ((r[0] & 7u) | 8u));
            } else {
                set_prg16(buffers, env, 0, 7u + (prg16_count >> 4));
            }
            set_prg16(buffers, env, 1, 7);
            set_chr8(buffers, env, 0);
            break;
        }
        case kBankingVrc3:  // [0] PRG [1,2] IRQ reload [3,4] IRQ counter [5] IRQ mode bit 2 [6] auto-enable [7] enabled
            set_prg16(buffers, env, 0, r[0]);
            set_prg16(buffers, env, 1, ~0u);
            set_chr8(buffers, env, 0);
            break;
        case kBankingM227: {  // [0,1] address latch
            const std::uint32_t l = r[0] | (static_cast<std::uint32_t>(r[1]) << 8);
            const std::uint32_t s = l & 1u, p = ((l >> 2) & 0x1Fu) + ((l & 0x100u) >> 3), big = (l >> 9) & 1u;
            if (((l >> 7) & 1u) != 0) {
                if (s != 0) {
                    set_prg32(buffers, env, p >> 1);
                } else {
                    set_prg16(buffers, env, 0, p);
                    set_prg16(buffers, env, 1, p);
                }
            } else {
                set_prg16(buffers, env, 0, s != 0 ? (p & 0x3Eu) : p);
                set_prg16(buffers, env, 1, big != 0 ? (p | 7u) : (p & 0x38u));
            }
            set_mirroring(buffers, env, ((l >> 1) & 1u) != 0 ? kPageMapHorizontal : kPageMapVertical);
            set_chr8(buffers, env, 0);
            break;
        }
        case kBankingM8:  // [0] latch
            set_prg16(buffers, env, 0, r[0] >> 3);
            set_prg16(buffers, env, 1, 1);
            set_chr8(buffers, env, r[0] & 3u);
            break;
        case kBankingCprom:  // [0] latch: 4 KB CHR RAM bank at $1000 out of 16 KB; banks 2-3 are the prg_ram half
            set_prg32(buffers, env, 0);
            set_chr4(buffers, env, 0, 0);
            for (std::uint32_t i = 0; i < 4u; ++i) {
                const std::uint32_t page = (r[0] & 3u) * 4u + i;
                if (page >= 8u) {
                    set_chr1_ram(buffers, env, 4u + i, page);
                } else {
                    set_chr1(buffers, env, 4u + i, page);
                }
            }
            break;
        case kBankingTaito48:  // [0..7] regs [8] mirroring (FCEUmm mirr) [9] IRQ on [10] latch [11,12] counter
            set_prg8(buffers, env, 0, r[0]);
            set_prg8(buffers, env, 1, r[1]);
            set_prg8(buffers, env, 2, ~1u);
            set_prg8(buffers, env, 3, ~0u);
            for (std::uint32_t i = 0; i < 2u; ++i) {
                set_chr1(buffers, env, 2u * i, r[2u + i] * 2u);
                set_chr1(buffers, env, 2u * i + 1u, r[2u + i] * 2u + 1u);
            }
            for (std::uint32_t i = 0; i < 4u; ++i) set_chr1(buffers, env, 4u + i, r[4u + i]);
            set_mirroring(buffers, env, r[8] != 0 ? kPageMapVertical : kPageMapHorizontal);
            break;
        case kBankingJaleco18:  // [0..3] PRG [4..11] CHR [12] mirroring [13] IRQ on [14,15] latch [16,17] counter
            for (std::uint32_t i = 0; i < 8u; ++i) set_chr1(buffers, env, i, r[4u + i]);
            for (std::uint32_t i = 0; i < 3u; ++i) set_prg8(buffers, env, i, r[i]);
            set_prg8(buffers, env, 3, ~0u);
            set_mirroring(buffers, env, (r[12] & 2u) != 0 ? kPageMapSingleLow
                                        : (r[12] & 1u) != 0 ? kPageMapVertical : kPageMapHorizontal);
            break;
        case kBankingM117:  // [0..3] PRG [4..11] CHR [12] mirroring [13] IRQ flags [14] counter [15] latch
            for (std::uint32_t i = 0; i < 4u; ++i) set_prg8(buffers, env, i, r[i]);
            for (std::uint32_t i = 0; i < 8u; ++i) set_chr1(buffers, env, i, r[4u + i]);
            set_mirroring(buffers, env, (r[12] & 1u) != 0 ? kPageMapHorizontal : kPageMapVertical);
            break;
        case kBankingM222:  // [0,1] PRG [2..9] CHR [10] mirroring [11] IRQ on [12] counter [13] write on a visible line
            set_prg8(buffers, env, 0, r[0]);
            set_prg8(buffers, env, 1, r[1]);
            set_prg16(buffers, env, 1, ~0u);
            for (std::uint32_t i = 0; i < 8u; ++i) set_chr1(buffers, env, i, r[2u + i]);
            set_mirroring(buffers, env, (r[10] & 1u) != 0 ? kPageMapHorizontal : kPageMapVertical);
            break;
        case kBankingBandai16:  // [0..7] CHR [8] PRG [9] mirroring [10] IRQ on [11,12] counter [13,14] latch [15..23] EEPROM
            for (std::uint32_t i = 0; i < 8u; ++i) set_chr1(buffers, env, i, r[i]);
            set_prg16(buffers, env, 0, r[8]);
            set_prg16(buffers, env, 1, ~0u);
            switch (r[9] & 3u) {
                case 0: set_mirroring(buffers, env, kPageMapVertical); break;
                case 1: set_mirroring(buffers, env, kPageMapHorizontal); break;
                case 2: set_mirroring(buffers, env, kPageMapSingleLow); break;
                default: set_mirroring(buffers, env, kPageMapSingleHigh); break;
            }
            break;
        case kBankingFfe:  // [0] latch [1] mirroring (FCEUmm mirr) [2] IRQ on [3..5] counter (17 bits)
            // Mapper17_Init sets ffemode = 1, then calls Mapper6_Init, which clears it: both run in latch mode.
            set_chr8(buffers, env, r[0] & 3u);
            set_prg16(buffers, env, 0, (r[0] >> 2) & 0x3Fu);
            set_prg16(buffers, env, 1, 7);
            switch (r[1] & 3u) {
                case 0: set_mirroring(buffers, env, kPageMapSingleLow); break;
                case 1: set_mirroring(buffers, env, kPageMapSingleHigh); break;
                case 2: set_mirroring(buffers, env, kPageMapVertical); break;
                default: set_mirroring(buffers, env, kPageMapHorizontal); break;
            }
            break;
        case kBankingN163: {  // [0..2] PRG [3..10] CHR [11..18] CHR in effect [19..22] NTAPage [23,24] IRQ count
                              // [25] IRQ on [26] IRAM address (dopol) [27] gorfus; NT CHR pages at kNtChrPage
            for (std::uint32_t i = 0; i < 3u; ++i) set_prg8(buffers, env, i, r[i]);
            set_prg8(buffers, env, 3, 0x3Fu);
            for (std::uint32_t i = 0; i < 8u; ++i) set_chr1(buffers, env, i, r[11u + i]);
            std::uint8_t map = 0;
            for (std::uint32_t q = 0; q < 4u; ++q) {  // DoNTARAMROM
                const std::uint32_t v = r[19u + q];
                if (v >= 0xE0u) {
                    map = static_cast<std::uint8_t>(map | ((v & 1u) << q));
                } else {
                    map = static_cast<std::uint8_t>(map | (1u << (4u + q)));  // page set by n163_write
                }
            }
            set_mirroring(buffers, env, map);
            break;
        }
        case kBankingSunsoft4:  // [0..3] CHR [4] nt1 [5] nt2 [6] mirr [7] PRG; NT CHR pages at kNtChrPage (68.c)
            for (std::uint32_t i = 0; i < 4u; ++i) {
                set_chr1(buffers, env, 2u * i, r[i] * 2u);
                set_chr1(buffers, env, 2u * i + 1u, r[i] * 2u + 1u);
            }
            set_prg16(buffers, env, 0, r[7]);
            set_prg16(buffers, env, 1, ~0u);
            if ((r[6] & 0x10u) != 0) {  // M68NTfix: CHR ROM nametables
                set_mirroring(buffers, env, kPageMapNtChr);
            } else {
                switch (r[6] & 3u) {
                    case 0: set_mirroring(buffers, env, kPageMapVertical); break;
                    case 1: set_mirroring(buffers, env, kPageMapHorizontal); break;
                    case 2: set_mirroring(buffers, env, kPageMapSingleLow); break;
                    default: set_mirroring(buffers, env, kPageMapSingleHigh); break;
                }
            }
            break;
        case kBankingMmc5: {  // MMC5PRG, MMC5CHRA/B (the last synced set is the CPU's view), $5105
            const std::uint8_t* p = r + kM5Prg;
            auto ram = [&](std::uint32_t slot, std::uint32_t v) {  // MMC5WRAM for $8000-$DFFF
                const std::uint32_t idx = mmc5_wram_index(buffers.cart, v);
                buffers.mapper.prg_page[static_cast<std::uint64_t>(env) * kPrgPages + slot] =
                    idx == 255u ? kPrgPageOpen
                                : kPrgPageRam | ((idx * 8192u) & (buffers.cart.ext_ram_bytes - kMmc5ExWram - 1u));
            };
            switch (r[kM5PSize] & 3u) {
                case 0:
                    set_prg32(buffers, env, (p[1] & 0x7Fu) >> 2);
                    break;
                case 1:
                    if ((p[1] & 0x80u) != 0) {
                        set_prg16(buffers, env, 0, p[1] >> 1);
                    } else {
                        ram(0, p[1] & 7u & 0xFEu);
                        ram(1, (p[1] & 7u & 0xFEu) + 1u);
                    }
                    set_prg16(buffers, env, 1, (p[3] & 0x7Fu) >> 1);
                    break;
                case 2:
                    if ((p[1] & 0x80u) != 0) {
                        set_prg16(buffers, env, 0, (p[1] & 0x7Fu) >> 1);
                    } else {
                        ram(0, p[1] & 7u & 0xFEu);
                        ram(1, (p[1] & 7u & 0xFEu) + 1u);
                    }
                    if ((p[2] & 0x80u) != 0) set_prg8(buffers, env, 2, p[2] & 0x7Fu);
                    else ram(2, p[2] & 7u);
                    set_prg8(buffers, env, 3, p[3] & 0x7Fu);
                    break;
                default:
                    for (std::uint32_t x = 0; x < 3u; ++x) {
                        if ((p[x] & 0x80u) != 0) set_prg8(buffers, env, x, p[x] & 0x7Fu);
                        else ram(x, p[x] & 7u);
                    }
                    set_prg8(buffers, env, 3, p[3] & 0x7Fu);
                    break;
            }
            std::uint16_t a[8], b[4];
            mmc5_banks(r, a, b);
            for (std::uint32_t i = 0; i < 8u; ++i) {
                set_chr1(buffers, env, i, mmc5_chr_page(a, b, r[kM5VSize], r[kM5AB] != 0, i));
            }
            set_mirroring(buffers, env, r[kM5Nta]);  // 2 bits per quadrant: CIRAM 0/1, ExRAM, fill
            break;
        }
        case kBankingFme7: {
            for (std::uint32_t i = 0; i < 3u; ++i) set_prg8(buffers, env, i, r[kFmePrg + i] & 0x3Fu);
            set_prg8(buffers, env, 3, ~0u);
            for (std::uint32_t i = 0; i < 8u; ++i) set_chr1(buffers, env, i, r[kFmeChr + i]);
            switch (r[kFmeMirror] & 3u) {
                case 0: set_mirroring(buffers, env, kPageMapVertical); break;
                case 1: set_mirroring(buffers, env, kPageMapHorizontal); break;
                case 2: set_mirroring(buffers, env, kPageMapSingleLow); break;
                default: set_mirroring(buffers, env, kPageMapSingleHigh); break;
            }
            break;
        }
        default:
            break;
    }
}

// Power-on registers (FCEUmm's Power handlers), then the matching page tables.
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE void power_on(const BatchBuffers& buffers, std::uint32_t env) {
    std::uint8_t* r = regs(buffers, env);
    for (int i = 0; i < kMapperRegBytes; ++i) {
        r[i] = 0;
    }
    buffers.mapper.mirroring[env] = header_page_map(buffers.cart);
    switch (buffers.cart.bank_kind) {
        case kBankingMmc1:
            r[kMmc1Control] = 0x0C;  // PRG mode 3: last 16 KB fixed at $C000
            if (buffers.cart.mapper == 105) {  // MMC1CMReset: control $1F; the CHR hook leaves NWCRec = CHR1 = 0
                r[kMmc1Control] = 0x1F;
            }
            break;
        case kBankingMmc2:
            r[kMmc2Latch0] = 1;
            r[kMmc2Latch1] = 1;
            break;
        case kBankingVrc21: case kBankingVrc22: case kBankingVrc23: case kBankingVrc25:
            r[kVrcPrg1] = 1;
            r[kVrcMirror] = buffers.cart.nametable_arrangement == kNametableHorizontal ? 1u : 0u;
            break;
        case kBankingVrc6a: case kBankingVrc6b:
            r[kVrc6Control] = buffers.cart.nametable_arrangement == kNametableHorizontal ? 4u : 0u;
            break;
        case kBankingNina001:
            r[2] = 1;
            break;
        case kBankingM42:
            r[2] = 1;
            break;
        case kBankingM228:
            r[1] = 0x80;
            break;
        case kBankingM246:
            r[0] = r[1] = r[2] = r[3] = 0xFF;
            break;
        case kBankingM243:
            r[1] = 3;
            break;
        case kBankingMmc5:  // GenMMC5Reset
            for (std::uint32_t i = 0; i < 4u; ++i) r[kM5Prg + i] = 0xFF;
            for (std::uint32_t i = 0; i < 8u; ++i) { r[kM5ALo + i] = 0xFF; r[kM5AHi + i] = 3; }
            for (std::uint32_t i = 0; i < 4u; ++i) { r[kM5BLo + i] = 0xFF; r[kM5BHi + i] = 3; }
            r[kM5WramMask] = r[kM5WramMask + 1] = 0xFF;
            r[kM5PSize] = r[kM5VSize] = 3;
            r[kM5Nta] = r[kM5NtFill] = r[kM5AtFill] = 0xFF;
            break;
        case kBankingN163:
            r[19] = r[20] = r[21] = r[22] = 0xFF;
            r[27] = 0xFF;
            break;
        case kBankingFfe:  // mirr = ((info->mirror & 1) ^ 1) | 2: the header's H/V
            r[1] = static_cast<std::uint8_t>((buffers.cart.nametable_arrangement == kNametableHorizontal ? 1u : 0u) | 2u);
            break;
        case kBankingJaleco18:
            r[1] = 1;
            r[2] = 0xFE;
            r[3] = 0xFF;
            break;
        case kBankingM117:
            r[0] = 0xFC;
            r[1] = 0xFD;
            r[2] = 0xFE;
            r[3] = 0xFF;
            break;
        case kBankingJy90:  // M90Power
            for (std::uint32_t i = 0; i < 20u; ++i) r[i] = 0xFF;
            r[kJyMul] = r[kJyMul + 1] = r[kJyRegie] = 0xFF;
            break;
        case kBankingFme7:
            r[kFmePrg + 1] = 1;
            r[kFmePrg + 2] = 2;
            r[kFmeMirror] = buffers.cart.nametable_arrangement == kNametableHorizontal ? 1u : 0u;
            break;
        case kBankingMmc3:
            if (buffers.cart.mapper == 64 || buffers.cart.mapper == 158) {  // RAMBO1Power: every bank register ~0, vertical mirroring
                for (std::uint32_t i = 0; i < 11u; ++i) r[kRamboR + i] = 0xFF;
                r[kRamboMirror] = 1;
                break;
            }
            if (buffers.cart.mapper == 121) r[kM121Ex + 3] = 0x80;  // M121Power
            r[1] = 2;
            r[2] = 4;
            r[3] = 5;
            r[4] = 6;
            r[5] = 7;
            r[7] = 1;
            // M4Power: mirroring from the header. GenMMC3Power (187): vertical.
            r[kMmc3Mirroring] =
                buffers.cart.mapper != 187 && buffers.cart.mapper != 119 && buffers.cart.mapper != 182 &&
                buffers.cart.mapper != 114 && buffers.cart.mapper != 47 &&
                buffers.cart.nametable_arrangement == kNametableHorizontal ? 1u : 0u;
            break;
        default:
            break;
    }
    sync(buffers, env);
}

// Registers restored from a save state (fcs::board_registers), then the matching
// page tables. Mirroring starts from the header, as at power-on: CNROM has no
// mirroring register, the other boards set theirs in sync().
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE void restore(const BatchBuffers& buffers,
                                                           std::uint32_t env,
                                                           const std::uint8_t* saved) {
    std::uint8_t* r = regs(buffers, env);
    for (int i = 0; i < kMapperRegBytes; ++i) {
        r[i] = saved[i];
    }
    buffers.mapper.mirroring[env] = header_page_map(buffers.cart);
    sync(buffers, env);
}

// A CPU write to $8000-$FFFF. `rom_byte` is the PRG byte at the address, for
// bus conflicts.
// Bandai FCG ($6000-$FFFF, A & $F) with FCEUmm's x24c0x 24C02 model. EEPROM bytes live in prg_ram[0..255];
// regs [15] addr [16] word [17] latch [18] bitcount [19] sda [20] scl [21] out [22] oe [23] state.
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE void bandai16_write(BatchBuffers& buffers, std::uint32_t env,
                                                                  std::uint16_t address, std::uint8_t value) {
    std::uint8_t* r = regs(buffers, env);
    const std::uint32_t a = address & 0x0Fu;
    if (a < 0x0Au) {
        r[a] = value;
        sync(buffers, env);
        return;
    }
    switch (a) {
        case 0x0Au:
            buffers.cpu.irq_pending[env] = 0;
            r[10] = static_cast<std::uint8_t>(value & 1u);
            r[11] = r[13];
            r[12] = r[14];
            return;
        case 0x0Bu: r[13] = value; return;
        case 0x0Cu: r[14] = value; return;
        case 0x0Du: break;
        default: return;
    }
    enum { kStandby = 0, kAddress = 1, kWord = 2, kRead = 3, kWrite = 4 };
    std::uint8_t* data = buffers.cpu.prg_ram + static_cast<std::uint64_t>(env) * kPrgRamBytes;
    const std::uint8_t sda = (value >> 6) & 1u, scl = (value >> 5) & 1u;
    r[22] = static_cast<std::uint8_t>(value >> 7);
    if (r[20] != 0 && scl != 0) {
        if (r[19] != 0 && sda == 0) {  // START
            r[23] = kAddress;
            r[18] = 0;
            r[15] = 0;
        } else if (r[19] == 0 && sda != 0) {  // STOP
            r[23] = kStandby;
        }
    } else if (r[20] == 0 && scl != 0) {  // rising edge
        switch (r[23]) {
            case kAddress:
                if (r[18] < 7u) {
                    r[15] = static_cast<std::uint8_t>((r[15] << 1) | sda);
                } else {
                    r[23] = sda != 0 ? kRead : kWord;  // 24C02
                }
                ++r[18];
                break;
            case kWord:
                if (r[18] == 8u) {
                    r[16] = 0;
                    r[21] = 0;
                } else {
                    r[16] = static_cast<std::uint8_t>((r[16] << 1) | sda);
                    if (r[18] == 16u) {
                        r[18] = 7;
                        r[23] = kWrite;
                    }
                }
                ++r[18];
                break;
            case kRead:
                if (r[18] == 8u) {
                    r[21] = 0;
                    r[17] = data[r[16]];
                    r[18] = 0;
                } else {
                    r[21] = static_cast<std::uint8_t>(r[17] >> 7);
                    r[17] = static_cast<std::uint8_t>(r[17] << 1);
                    if (++r[18] == 8u) ++r[16];
                }
                break;
            case kWrite:
                if (r[18] == 8u) {
                    r[21] = 0;
                    r[17] = 0;
                    r[18] = 0;
                } else {
                    r[17] = static_cast<std::uint8_t>((r[17] << 1) | sda);
                    if (++r[18] == 8u) {
                        data[r[16]] = r[17];
                        ++r[16];
                    }
                }
                break;
            default: break;
        }
    }
    r[19] = sda;
    r[20] = scl;
}

// Namco 163 writes (n106.c Mapper19_write and Mapper19C0D8_write), $4020-$FFFF.
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE void n163_write(BatchBuffers& buffers, std::uint32_t env,
                                                              std::uint16_t address, std::uint8_t value) {
    std::uint8_t* r = regs(buffers, env);
    const std::uint32_t a = address & 0xF800u;
    auto chr = [&](std::uint32_t x) {  // DoCHRRAMROM: a CHR RAM pick ($E0+) leaves the slot as it was
        if (((r[27] >> ((x >> 2) + 6u)) & 1u) != 0 || r[3u + x] < 0xE0u) r[11u + x] = r[3u + x];
    };
    if (a >= 0x8000u && a <= 0xB800u) {
        const std::uint32_t x = (a - 0x8000u) >> 11;
        r[3u + x] = value;
        chr(x);
    } else if (a >= 0xC000u && a <= 0xD800u) {
        const std::uint32_t q = (a - 0xC000u) >> 11;
        r[19u + q] = value;
        r[kNtChrPage + q] = static_cast<std::uint8_t>(value & buffers.cart.chr_page_mask);
    } else {
        switch (a) {
            case 0x4800u:  // sound RAM write: not kept
                if ((r[26] & 0x80u) != 0) r[26] = static_cast<std::uint8_t>((r[26] & 0x80u) | ((r[26] + 1u) & 0x7Fu));
                return;
            case 0xF800u: r[26] = value; return;
            case 0x5000u:
                r[23] = value;
                buffers.cpu.irq_pending[env] = 0;
                return;
            case 0x5800u:
                r[24] = static_cast<std::uint8_t>(value & 0x7Fu);
                r[25] = static_cast<std::uint8_t>(value & 0x80u);
                buffers.cpu.irq_pending[env] = 0;
                return;
            case 0xE000u: r[0] = static_cast<std::uint8_t>(value & 0x3Fu); break;
            case 0xE800u:
                r[27] = static_cast<std::uint8_t>(value & 0xC0u);
                for (std::uint32_t x = 0; x < 8u; ++x) chr(x);
                r[1] = static_cast<std::uint8_t>(value & 0x3Fu);
                break;
            case 0xF000u: r[2] = static_cast<std::uint8_t>(value & 0x3Fu); break;
            default: return;
        }
    }
    sync(buffers, env);
}

// MMC5 writes: Mapper5_write ($5000-$5BFF), MMC5_ExRAMWr ($5C00-$5FFF), MMC5_WriteROMRAM ($6000-$FFFF).
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE void mmc5_write(BatchBuffers& buffers, std::uint32_t env,
                                                              std::uint16_t address, std::uint8_t value) {
    std::uint8_t* r = regs(buffers, env);
    std::uint8_t* ext = mmc5_ext(buffers, env);
    if (address >= 0x6000u) {
        if (((r[kM5WramMask] & 3u) | ((r[kM5WramMask + 1] & 3u) << 2)) != 6u) return;
        std::uint32_t page;
        if (address < 0x8000u) {
            const std::uint32_t idx = mmc5_wram_index(buffers.cart, r[kM5WramPage]);
            if (idx == 255u) return;
            page = kPrgPageRam | ((idx * 8192u) & (buffers.cart.ext_ram_bytes - kMmc5ExWram - 1u));
        } else {
            page = buffers.mapper.prg_page[static_cast<std::uint64_t>(env) * kPrgPages + ((address >> 13) & 3u)];
        }
        if ((page & kPrgPageRam) != 0) ext[kMmc5ExWram + ((page & 0xFFFFFu) | (address & 0x1FFFu))] = value;
        return;  // ROM (write-protected) or no RAM
    }
    if (address >= 0x5C00u) {
        if ((r[kM5ChrMode] & 3u) != 3u) ext[kMmc5ExRam + (address & 0x3FFu)] = value;
        return;
    }
    switch (address) {
        case 0x5100u: r[kM5PSize] = value; break;
        case 0x5101u: r[kM5VSize] = value; break;
        case 0x5102u: r[kM5WramMask] = value; return;
        case 0x5103u: r[kM5WramMask + 1] = value; return;
        case 0x5104u: r[kM5ChrMode] = value; return;
        case 0x5105u: r[kM5Nta] = value; break;
        case 0x5106u: r[kM5NtFill] = value; return;
        case 0x5107u: r[kM5AtFill] = value; return;
        case 0x5113u: r[kM5WramPage] = value; return;
        case 0x5114u: case 0x5115u: case 0x5116u: case 0x5117u: r[kM5Prg + (address & 3u)] = value; break;
        case 0x5120u: case 0x5121u: case 0x5122u: case 0x5123u:
        case 0x5124u: case 0x5125u: case 0x5126u: case 0x5127u:
            r[kM5AB] = 0;
            r[kM5ALo + (address & 7u)] = value;
            r[kM5AHi + (address & 7u)] = static_cast<std::uint8_t>(r[kM5Hi] & 3u);
            break;
        case 0x5128u: case 0x5129u: case 0x512Au: case 0x512Bu:
            r[kM5AB] = 1;
            r[kM5BLo + (address & 3u)] = value;
            r[kM5BHi + (address & 3u)] = static_cast<std::uint8_t>(r[kM5Hi] & 3u);
            break;
        case 0x5130u: r[kM5Hi] = value; return;
        case 0x5200u: r[kM5Split] = value; return;
        case 0x5201u: r[46] = static_cast<std::uint8_t>((value >> 3) & 0x1Fu); return;
        case 0x5202u: r[47] = static_cast<std::uint8_t>(value & 0x3Fu); return;
        case 0x5203u: buffers.cpu.irq_pending[env] = 0; r[kM5IrqLine] = value; return;
        case 0x5204u: buffers.cpu.irq_pending[env] = 0; r[kM5IrqOn] = static_cast<std::uint8_t>(value & 0x80u); return;
        case 0x5205u: r[kM5Mul] = value; return;
        case 0x5206u: r[kM5Mul + 1] = value; return;
        default: return;  // $5000-$5015 sound and the rest
    }
    sync(buffers, env);
}

// MMC5_hb(scanline): at the start of visible lines 0-239 and after line 239, while rendering is on.
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE void mmc5_line(BatchBuffers& buffers, std::uint32_t env,
                                                             std::uint32_t scanline) {
    std::uint8_t* r = regs(buffers, env);
    if (scanline + 1u >= 241u) {
        r[kM5IrqStatus] = static_cast<std::uint8_t>(r[kM5IrqStatus] & ~0xC0u);
        r[kM5LineCount] = 0;
        buffers.cpu.irq_pending[env] = 0;
        return;
    }
    if ((r[kM5IrqStatus] & 0x40u) == 0) {
        r[kM5IrqStatus] = static_cast<std::uint8_t>((r[kM5IrqStatus] | 0x40u) & ~0x80u);
        r[kM5LineCount] = 0;
        buffers.cpu.irq_pending[env] = 0;
    } else if (++r[kM5LineCount] == r[kM5IrqLine]) {
        r[kM5IrqStatus] = static_cast<std::uint8_t>(r[kM5IrqStatus] | 0x80u);
        if ((r[kM5IrqOn] & 0x80u) != 0) buffers.cpu.irq_pending[env] = 1;
    }
}

NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE void write(BatchBuffers& buffers,
                                                         std::uint32_t env,
                                                         std::uint16_t address,
                                                         std::uint8_t value,
                                                         std::uint8_t rom_byte) {
    std::uint8_t* r = regs(buffers, env);
    const std::uint8_t kind = buffers.cart.bank_kind;
    if (address < 0x8000u) {
        // $4020-$7FFF: only the boards with a register there respond.
        if (kind == kBankingNina0306 || kind == kBankingHes) {
            if (address >= 0x6000u || (address & 0x0100u) == 0) return;
        } else if (kind == kBankingJaleco140 || kind == kBankingJaleco87) {
            if (address < 0x6000u) return;
        } else if (kind == kBankingBitCorp38) {
            if (address < 0x7000u) return;
        } else if (kind == kBankingM91) {
            if (address < 0x6000u) return;
            if (address < 0x7000u) {
                if ((address & 7u) < 4u) {
                    r[address & 3u] = value;
                } else if ((address & 7u) < 6u) {
                    r[9] = static_cast<std::uint8_t>(value & 1u);  // submapper 1 only (Super Fighter III)
                    return;
                } else {
                    return;
                }
            } else {
                switch (address & 3u) {
                    case 0: case 1: r[4u + (address & 1u)] = value; break;
                    case 2: r[6] = 0; r[7] = 0; buffers.cpu.irq_pending[env] = 0; return;
                    default: r[6] = 1; buffers.cpu.irq_pending[env] = 0; return;
                }
            }
            sync(buffers, env);
            return;
        } else if (kind == kBankingM228) {
            if (address >= 0x5000u && address < 0x6000u) r[3u + (address & 3u)] = static_cast<std::uint8_t>(value & 0x0Fu);
            return;
        } else if (kind == kBankingM83) {
            if (address >= 0x5100u && address <= 0x5103u) r[kM83Low + (address & 3u)] = value;  // UNLYOKOWriteLow
            return;
        } else if (kind == kBankingJy90) {
            if (address < 0x5000u || address >= 0x6000u) return;
            switch (address & 0x5C03u) {  // M90TekWrite
                case 0x5800u: r[kJyMul] = value; break;
                case 0x5801u: r[kJyMul + 1] = value; break;
                case 0x5803u: r[kJyRegie] = value; break;
                default: break;
            }
            return;
        } else if (kind == kBankingSa72008) {
            if (address >= 0x6000u || (address & 0x0100u) == 0 || buffers.cart.mapper == 149) return;
            r[0] = value;
            sync(buffers, env);
            return;
        } else if (kind == kBankingM243) {  // S74LS374NAWrite, $4100-$7FFF
            if (address < 0x4100u) return;
            if ((address & 0x4101u) == 0x4100u) {
                r[4] = static_cast<std::uint8_t>(value & 7u);
                return;
            }
            if ((address & 0x4101u) != 0x4101u) return;
            switch (r[4]) {
                case 0: r[0] = 0; r[1] = 3; break;
                case 2: r[3] = static_cast<std::uint8_t>((value & 1u) << 3); break;
                case 4: r[1] = static_cast<std::uint8_t>((r[1] & 6u) | (value & 3u)); break;
                case 5: r[0] = static_cast<std::uint8_t>(value & 1u); break;
                case 6: r[1] = static_cast<std::uint8_t>((r[1] & 1u) | r[3] | ((value & 3u) << 1)); break;
                case 7: r[2] = static_cast<std::uint8_t>(value & 1u); break;
                default: break;
            }
            sync(buffers, env);
            return;
        } else if (kind == kBankingN163) {
            n163_write(buffers, env, address, value);
            return;
        } else if (kind == kBankingMmc5) {
            mmc5_write(buffers, env, address, value);
            return;
        } else if (kind == kBankingFfe) {
            if (address == 0x42FEu || address == 0x42FFu) {  // FFEWriteMirr
                r[1] = static_cast<std::uint8_t>(((address << 1) & 2u) | ((value >> 4) & 1u));
                sync(buffers, env);
            } else if (address == 0x4501u) {
                r[2] = 0;
                buffers.cpu.irq_pending[env] = 0;
            } else if (address == 0x4502u) {
                r[3] = value;
                buffers.cpu.irq_pending[env] = 0;
            } else if (address == 0x4503u) {
                r[4] = value;
                r[2] = 1;
                buffers.cpu.irq_pending[env] = 0;
            }
            return;  // $4504-$4517 (FFE mode registers) are written but never used in latch mode
        } else if (kind == kBankingBandai16) {
            bandai16_write(buffers, env, address, value);
            return;
        } else if (kind == kBankingM246) {
            if (address < 0x6000u || address >= 0x6800u) return;
            r[address & 7u] = value;
            sync(buffers, env);
            return;
        } else if (kind == kBankingNina001) {
            if (address < 0x7FFDu) return;
            r[address - 0x7FFDu] = value;
            sync(buffers, env);
            return;
        } else if (kind == kBankingMmc3 && (buffers.cart.mapper == 114 || buffers.cart.mapper == 182)) {
            if (address < 0x5000u) return;  // M114ExWrite: $5000-$7FFF
            r[kMmc3Outer] = value;
            sync(buffers, env);
            return;
        } else if (kind == kBankingMmc3 && buffers.cart.mapper == 47) {
            if (address < 0x6000u) return;  // M47Write
            r[kMmc3Outer] = static_cast<std::uint8_t>(value & 1u);
            sync(buffers, env);
            return;
        } else if (kind == kBankingMmc3 && buffers.cart.mapper == 121) {
            if (address < 0x5000u || address >= 0x6000u) return;
            constexpr std::uint8_t kProt[4] = {0x83, 0x83, 0x42, 0x00};  // M121LoWrite
            r[kM121Ex + 4] = kProt[value & 3u];
            if ((address & 0x5180u) == 0x5180u) {
                r[kM121Ex + 3] = value;
                sync(buffers, env);
            }
            return;
        } else if (kind == kBankingMmc3 && buffers.cart.mapper == 187) {
            if (address != 0x5000u && address != 0x6000u) return;  // M187WriteLo
            r[kMmc3Outer] = value;
            sync(buffers, env);
            return;
        } else if (kind == kBankingMmc3 && buffers.cart.mapper == 189) {
            if (address < 0x4120u) return;
            r[kMmc3Outer] = static_cast<std::uint8_t>(value | (value >> 4));
            sync(buffers, env);
            return;
        } else {
            return;
        }
        r[0] = value;
        sync(buffers, env);
        return;
    }
    switch (kind) {
        case kBankingNina0306: case kBankingHes: case kBankingJaleco140: case kBankingBitCorp38:
        case kBankingJaleco87:
            return;  // no register at $8000-$FFFF
        case kBankingAxrom: case kBankingColorDreams: case kBankingGxrom:
            r[0] = value;
            break;
        case kBankingVrc21: case kBankingVrc22: case kBankingVrc23: case kBankingVrc25: {
            std::uint32_t sel;
            switch (kind) {
                case kBankingVrc21: sel = ((address >> 1) & 3u) | ((address >> 6) & 3u); break;
                case kBankingVrc22: sel = ((address >> 1) & 1u) | ((address << 1) & 2u); break;
                case kBankingVrc23: sel = (address & 3u) | ((address >> 2) & 3u); break;
                default:
                    sel = ((address >> 1) & 1u) | ((address << 1) & 2u) | ((address >> 3) & 1u) | ((address >> 1) & 2u);
                    break;
            }
            const std::uint32_t page = address & 0xF000u;
            if (page == 0x8000u) {
                r[kVrcPrg0] = static_cast<std::uint8_t>(value & 0x1Fu);
            } else if (page == 0xA000u) {
                r[kVrcPrg1] = static_cast<std::uint8_t>(value & 0x1Fu);
            } else if (page == 0x9000u) {
                if (kind == kBankingVrc22 || sel < 2u) {
                    r[kVrcMirror] = value;
                } else {
                    r[kVrcSwap] = value;
                }
            } else if (page >= 0xB000u && page <= 0xE000u) {
                const std::uint32_t bank = ((page - 0xB000u) >> 12) * 2u + (sel >> 1);
                r[kVrcChr + 2u * bank + (sel & 1u)] = static_cast<std::uint8_t>(value & (sel & 1u ? 0x1Fu : 0x0Fu));
            } else {  // $F000: IRQ
                switch (sel) {
                    case 0: r[kVrcIrqLatch] = static_cast<std::uint8_t>((r[kVrcIrqLatch] & 0xF0u) | (value & 0x0Fu)); return;
                    case 1: r[kVrcIrqLatch] = static_cast<std::uint8_t>((r[kVrcIrqLatch] & 0x0Fu) | (value << 4)); return;
                    case 2:
                        r[kVrcIrqControl] = static_cast<std::uint8_t>(value & 7u);
                        if ((value & 2u) != 0) {
                            r[kVrcIrqCounter] = r[kVrcIrqLatch];
                            r[kVrcPrescaler] = 341 & 0xFF;
                            r[kVrcPrescaler + 1] = 341 >> 8;
                        }
                        buffers.cpu.irq_pending[env] = 0;
                        return;
                    default:
                        // ack: enabled = enable-after-ack bit
                        r[kVrcIrqControl] = static_cast<std::uint8_t>((r[kVrcIrqControl] & ~2u) |
                                                                      ((r[kVrcIrqControl] & 1u) << 1));
                        buffers.cpu.irq_pending[env] = 0;
                        return;
                }
            }
            break;
        }
        case kBankingVrc6a: case kBankingVrc6b: {
            const std::uint32_t sel = kind == kBankingVrc6a ? (address & 3u)
                                                            : (((address & 1u) << 1) | ((address >> 1) & 1u));
            switch (address & 0xF000u) {
                case 0x8000u: r[kVrc6Prg16] = value; break;
                case 0xC000u: r[kVrc6Prg8] = value; break;
                case 0xB000u:
                    if (sel != 3u) return;  // $B000-$B002: sound
                    r[kVrc6Control] = value;
                    break;
                case 0xD000u: r[4u + sel] = value; break;
                case 0xE000u: r[8u + sel] = value; break;
                case 0xF000u:
                    if (sel == 0u) {
                        r[kVrcIrqLatch] = value;
                    } else if (sel == 1u) {
                        r[kVrcIrqControl] = static_cast<std::uint8_t>(value & 7u);
                        if ((value & 2u) != 0) {
                            r[kVrcIrqCounter] = r[kVrcIrqLatch];
                            r[kVrcPrescaler] = 341 & 0xFF;
                            r[kVrcPrescaler + 1] = 341 >> 8;
                        }
                        buffers.cpu.irq_pending[env] = 0;
                    } else if (sel == 2u) {
                        r[kVrcIrqControl] = static_cast<std::uint8_t>((r[kVrcIrqControl] & ~2u) |
                                                                      ((r[kVrcIrqControl] & 1u) << 1));
                        buffers.cpu.irq_pending[env] = 0;
                    }
                    return;
                default: return;  // $9000-$AFFF: sound
            }
            break;
        }
        case kBankingSmb2j40:
            switch (address & 0xE000u) {
                case 0x8000u:
                    r[1] = 0; r[2] = 0; r[3] = 0;
                    buffers.cpu.irq_pending[env] = 0;
                    return;
                case 0xA000u: r[1] = 1; return;
                case 0xE000u: r[0] = value; break;
                default: return;
            }
            break;
        case kBankingNina001:
            r[0] = value;
            break;
        case kBankingM91:
            if (address >= 0xA000u) return;
            r[8] = static_cast<std::uint8_t>(address & 7u);
            break;
        case kBankingM42:
            switch (address & 0xE003u) {
                case 0x8000u: r[1] = value; break;
                case 0xE000u: r[0] = static_cast<std::uint8_t>(value & 0x0Fu); return;
                case 0xE001u: r[2] = static_cast<std::uint8_t>(((value >> 3) & 1u) ^ 1u); break;
                case 0xE002u:
                    r[3] = static_cast<std::uint8_t>(value & 2u);
                    if (r[3] == 0) { r[4] = 0; r[5] = 0; }
                    buffers.cpu.irq_pending[env] = 0;
                    return;
                default: return;
            }
            break;
        case kBankingM15:
            r[0] = value;
            r[1] = static_cast<std::uint8_t>(address & 3u);
            break;
        case kBankingM83:  // M83Write: exact addresses
            switch (address) {
                case 0x8000u: r[kM83Is2k] = 1; [[fallthrough]];
                case 0xB000u: case 0xB0FFu: case 0xB1FFu:
                    r[kM83Bank] = value;
                    r[kM83Mode] = static_cast<std::uint8_t>(r[kM83Mode] | 0x40u);
                    break;
                case 0x8100u: r[kM83Mode] = static_cast<std::uint8_t>(value | (r[kM83Mode] & 0x40u)); break;
                case 0x8200u:
                    r[kM83Count] = value;
                    buffers.cpu.irq_pending[env] = 0;
                    return;
                case 0x8201u:
                    r[kM83IrqOn] = static_cast<std::uint8_t>(r[kM83Mode] & 0x80u);
                    r[kM83Count + 1] = value;
                    return;
                case 0x8300u: case 0x8301u: case 0x8302u:
                    r[8u + (address & 3u)] = value;
                    r[kM83Mode] = static_cast<std::uint8_t>(r[kM83Mode] & 0xBFu);
                    break;
                case 0x8310u: case 0x8311u: case 0x8316u: case 0x8317u: r[address & 7u] = value; break;
                case 0x8312u: case 0x8313u: case 0x8314u: case 0x8315u:
                    r[address & 7u] = value;
                    r[kM83Not2k] = 1;
                    break;
                default: return;
            }
            break;
        case kBankingJy90:
            switch (address & 0xF000u) {
                case 0x8000u:
                    if (address > 0x8FF0u) return;
                    r[address & 3u] = value;
                    break;
                case 0x9000u: r[kJyChrLo + (address & 7u)] = value; break;
                case 0xA000u: r[kJyChrHi + (address & 7u)] = value; break;
                case 0xC000u:
                    switch (address & 7u) {
                        case 0:
                            r[kJyIrqOn] = static_cast<std::uint8_t>(value & 1u);
                            if ((value & 1u) == 0) buffers.cpu.irq_pending[env] = 0;
                            break;
                        case 1: r[kJyIrqMode] = value; break;
                        case 2:
                            r[kJyIrqOn] = 0;
                            buffers.cpu.irq_pending[env] = 0;
                            break;
                        case 3: r[kJyIrqOn] = 1; break;
                        case 4: r[kJyPre] = static_cast<std::uint8_t>(value ^ r[kJyXor]); break;
                        case 5: r[kJyCount] = static_cast<std::uint8_t>(value ^ r[kJyXor]); break;
                        case 6: r[kJyXor] = value; break;
                        default: break;  // $C007 prescaler size: unused by FCEUmm's counter
                    }
                    return;
                case 0xD000u:
                    if (address >= 0xD600u) return;
                    r[kJyMode + (address & 3u)] = value;
                    break;
                case 0xB000u:  // M90NTWrite: nametable registers, used by 209 only
                    if (buffers.cart.mapper != 209) return;
                    r[((address & 4u) != 0 ? kJyNamesHi : kJyNames) + (address & 3u)] = value;
                    for (std::uint32_t q = 0; q < 4u; ++q) {  // CHR ROM nametable pages: 9 bits, low 8 at kNtChrPage
                        const std::uint32_t page = (r[kJyNames + q] | (static_cast<std::uint32_t>(r[kJyNamesHi + q]) << 8)) &
                                                   buffers.cart.chr_page_mask;
                        r[kNtChrPage + q] = static_cast<std::uint8_t>(page);
                        r[kJyNtHi] = static_cast<std::uint8_t>((r[kJyNtHi] & ~(1u << q)) | (((page >> 8) & 1u) << q));
                    }
                    break;
                default: return;  // $B000 nametable registers: no effect on 90; $E000-$FFFF dummy
            }
            if ((r[kJyMode] & 0x80u) != 0) {  // tekprom maps $6000 only while mode bit 7 is set; the mapping then stays
                const std::uint32_t m0 = r[kJyMode], p3 = r[3];
                const std::uint32_t bm = (r[kJyMode + 3] & 6u) << 5;
                std::uint32_t b6;
                switch (m0 & 7u) {
                    case 0: case 4: b6 = ((p3 << 2) + 3u) & 0x3Fu; break;
                    case 1: case 5: b6 = ((p3 << 1) + 1u) & 0x3Fu; break;
                    default: b6 = p3 & 0x3Fu; break;
                }
                r[kJy6000On] = 1;
                r[kJy6000] = static_cast<std::uint8_t>(b6 | bm);
            }
            break;
        case kBankingVrc1:
            switch (address & 0xF000u) {
                case 0x8000u: r[0] = value; break;
                case 0x9000u: r[5] = value; break;
                case 0xA000u: r[1] = value; break;
                case 0xC000u: r[2] = value; break;
                case 0xE000u: r[3] = static_cast<std::uint8_t>(value & 0x0Fu); break;
                case 0xF000u: r[4] = static_cast<std::uint8_t>(value & 0x0Fu); break;
                default: return;
            }
            break;
        case kBankingSa72008:
            if (buffers.cart.mapper != 149) return;
            r[0] = value;  // SADWrite: 149's latch is at $8000-$FFFF, no bus conflict
            break;
        case kBankingM246: case kBankingM243:
            return;
        case kBankingM188: case kBankingM8: case kBankingCprom: case kBankingFfe:
            r[0] = value;
            break;
        case kBankingBandai16:
            bandai16_write(buffers, env, address, value);
            return;
        case kBankingN163:
            n163_write(buffers, env, address, value);
            return;
        case kBankingMmc5:
            mmc5_write(buffers, env, address, value);
            return;
        case kBankingSunsoft4: {
            switch (address & 0xF000u) {
                case 0x8000u: case 0x9000u: case 0xA000u: case 0xB000u: r[(address >> 12) & 3u] = value; break;
                case 0xC000u: r[4] = value; break;
                case 0xD000u: r[5] = value; break;
                case 0xE000u: r[6] = value; break;
                default: r[7] = static_cast<std::uint8_t>(value & 7u); break;  // M68WriteROM (kogame unused: no sub-cart)
            }
            const std::uint32_t n1 = (r[4] | 0x80u) & buffers.cart.chr_page_mask;
            const std::uint32_t n2 = (r[5] | 0x80u) & buffers.cart.chr_page_mask;
            std::uint32_t q[4];
            switch (r[6] & 3u) {
                case 0: q[0] = q[2] = n1; q[1] = q[3] = n2; break;
                case 1: q[0] = q[1] = n1; q[2] = q[3] = n2; break;
                case 2: q[0] = q[1] = q[2] = q[3] = n1; break;
                default: q[0] = q[1] = q[2] = q[3] = n2; break;
            }
            for (std::uint32_t i = 0; i < 4u; ++i) r[kNtChrPage + i] = static_cast<std::uint8_t>(q[i]);
            break;
        }
        case kBankingJaleco18:
            if (address < 0xE000u) {  // M18WritePrg / M18WriteChr: one nibble of a bank register
                const std::uint32_t base = address < 0xA000u ? 0x8000u : 0xA000u;
                const std::uint32_t i = (((address >> 1) & 1u) | ((address - base) >> 11)) + (base == 0xA000u ? 4u : 0u);
                if (base == 0x8000u && i > 3u) return;
                if (i > 11u) return;
                const std::uint32_t shift = (address & 1u) * 4u;
                r[i] = static_cast<std::uint8_t>((r[i] & (0xF0u >> shift)) | ((value & 0x0Fu) << shift));
                break;
            }
            switch (address & 0xF003u) {
                case 0xE000u: case 0xE001u: case 0xE002u: case 0xE003u: {
                    const std::uint32_t shift = (address & 3u) * 4u;
                    std::uint32_t latch = r[14] | (static_cast<std::uint32_t>(r[15]) << 8);
                    latch = (latch & ~(0xFu << shift)) | ((value & 0xFu) << shift);
                    r[14] = static_cast<std::uint8_t>(latch & 0xFFu);
                    r[15] = static_cast<std::uint8_t>(latch >> 8);
                    return;
                }
                case 0xF000u: r[16] = r[14]; r[17] = r[15]; return;
                case 0xF001u: r[13] = static_cast<std::uint8_t>(value & 1u); buffers.cpu.irq_pending[env] = 0; return;
                case 0xF002u: r[12] = static_cast<std::uint8_t>(value & 3u); break;
                default: return;
            }
            break;
        case kBankingM222:
            switch (address & 0xF003u) {
                case 0x8000u: r[0] = value; break;
                case 0x9000u: r[10] = static_cast<std::uint8_t>(value & 1u); break;
                case 0xA000u: r[1] = value; break;
                case 0xB000u: r[2] = value; break;
                case 0xB002u: r[3] = value; break;
                case 0xC000u: r[4] = value; break;
                case 0xC002u: r[5] = value; break;
                case 0xD000u: r[6] = value; break;
                case 0xD002u: r[7] = value; break;
                case 0xE000u: r[8] = value; break;
                case 0xE002u: r[9] = value; break;
                case 0xF000u:
                    r[11] = value;
                    r[12] = static_cast<std::uint8_t>(r[13] != 0 ? value - 8u : value + 4u);
                    buffers.cpu.irq_pending[env] = 0;
                    break;
                default: break;
            }
            break;  // M222Write syncs on every write
        case kBankingM117:
            if (address < 0x8004u) {
                r[address & 3u] = value;
            } else if (address >= 0xA000u && address <= 0xA007u) {
                r[4u + (address & 7u)] = value;
            } else {
                switch (address) {
                    case 0xC001u: r[15] = value; return;
                    case 0xC003u: r[14] = r[15]; r[13] = static_cast<std::uint8_t>(r[13] | 2u); return;
                    case 0xE000u:
                        r[13] = static_cast<std::uint8_t>((r[13] & ~1u) | (value & 1u));
                        buffers.cpu.irq_pending[env] = 0;
                        return;
                    case 0xC002u: buffers.cpu.irq_pending[env] = 0; return;
                    case 0xD000u: r[12] = static_cast<std::uint8_t>(value & 1u); break;
                    default: return;
                }
            }
            break;
        case kBankingTaito48:
            switch (address & 0xF003u) {
                case 0x8000u: case 0x8001u: r[address & 1u] = static_cast<std::uint8_t>(value & 0x3Fu); break;
                case 0x8002u: case 0x8003u: r[address & 3u] = value; break;
                case 0xA000u: case 0xA001u: case 0xA002u: case 0xA003u: r[4u + (address & 3u)] = value; break;
                case 0xC000u: r[10] = value; return;
                case 0xC001u: r[11] = r[10]; r[12] = 0; return;  // IRQCount = IRQLatch (int16)
                case 0xC002u: r[9] = 1; return;
                case 0xC003u: r[9] = 0; buffers.cpu.irq_pending[env] = 0; return;
                case 0xE000u: r[8] = static_cast<std::uint8_t>(((value >> 6) & 1u) ^ 1u); break;
                default: return;
            }
            break;
        case kBankingM227:
            r[0] = static_cast<std::uint8_t>(address & 0xFFu);
            r[1] = static_cast<std::uint8_t>(address >> 8);
            break;
        case kBankingVrc3:
            switch (address & 0xF000u) {
                case 0x8000u: case 0x9000u: case 0xA000u: case 0xB000u: {  // IRQ reload nibbles
                    const std::uint32_t shift = ((address >> 12) & 3u) * 4u;
                    std::uint32_t reload = r[1] | (static_cast<std::uint32_t>(r[2]) << 8);
                    reload = (reload & ~(0xFu << shift)) | ((value & 0xFu) << shift);
                    r[1] = static_cast<std::uint8_t>(reload & 0xFFu);
                    r[2] = static_cast<std::uint8_t>(reload >> 8);
                    return;
                }
                case 0xC000u:
                    r[5] = static_cast<std::uint8_t>(value & 4u);
                    r[6] = static_cast<std::uint8_t>(value & 1u);
                    r[7] = static_cast<std::uint8_t>(value & 2u);
                    if (r[7] != 0) {
                        if (r[5] == 0) {
                            r[3] = r[1];
                            r[4] = r[2];
                        } else {  // 8-bit mode keeps the counter, ORs the reload's low byte in
                            r[3] = static_cast<std::uint8_t>(r[3] | r[1]);
                        }
                    }
                    buffers.cpu.irq_pending[env] = 0;
                    return;
                case 0xD000u:
                    buffers.cpu.irq_pending[env] = 0;
                    r[7] = r[6];
                    return;
                case 0xF000u: r[0] = value; break;
                default: return;
            }
            break;
        case kBankingM228:
            r[0] = static_cast<std::uint8_t>(address & 0xFFu);
            r[1] = static_cast<std::uint8_t>(address >> 8);
            r[2] = value;
            break;
        case kBankingIrem32:
            switch (address & 0xF000u) {
                case 0x8000u: r[0] = value; break;
                case 0x9000u: r[1] = value; break;
                case 0xA000u: r[2] = value; break;
                case 0xB000u: r[3u + (address & 7u)] = value; break;
                default: return;
            }
            break;
        case kBankingNamco88:
            if ((address & 1u) == 0) {
                r[8] = static_cast<std::uint8_t>(value & 7u);
                return;
            }
            r[r[8] & 7u] = value;
            break;
        case kBankingQuattro:
            if (address < 0xC000u) {
                r[0] = static_cast<std::uint8_t>((value >> 1) & 0x0Cu);
            } else {
                r[1] = static_cast<std::uint8_t>(value & 3u);
            }
            break;
        case kBankingFme7:
            if (address < 0xA000u) {
                r[kFmeCommand] = static_cast<std::uint8_t>(value & 0x0Fu);
                return;
            }
            if (address >= 0xC000u) return;  // $C000-$FFFF: the 5B sound chip
            switch (r[kFmeCommand]) {
                case 8: r[kFmePrg6] = value; return;
                case 9: case 10: case 11: r[kFmePrg + r[kFmeCommand] - 9u] = value; break;
                case 12: r[kFmeMirror] = value; break;
                case 13:
                    r[kFmeIrqControl] = value;
                    buffers.cpu.irq_pending[env] = 0;
                    return;
                case 14: r[kFmeIrqCounter] = value; return;
                case 15: r[kFmeIrqCounter + 1] = value; return;
                default: r[kFmeChr + r[kFmeCommand]] = value; break;
            }
            break;
        case kBankingCamerica:
            if ((address & 0xF000u) == 0x9000u) {
                r[1] = static_cast<std::uint8_t>((value >> 4) & 1u);
                r[2] = 1;
            } else {
                r[0] = value;
            }
            break;
        case kBankingCnrom:
            r[0] = buffers.cart.bus_conflicts != 0 ? static_cast<std::uint8_t>(value & rom_byte)
                                                   : value;
            break;
        case kBankingMmc1:
            // FCEUmm also drops a write within two cycles of a reset write (the
            // second write of a read-modify-write): BatchCpuBusT::rmw_dummy_write.
            if ((value & 0x80u) != 0) {
                r[kMmc1Control] = static_cast<std::uint8_t>(r[kMmc1Control] | 0x0Cu);
                r[kMmc1Shift] = 0;
                r[kMmc1ShiftCount] = 0;
                break;
            }
            r[kMmc1Shift] = static_cast<std::uint8_t>(r[kMmc1Shift] |
                                                      ((value & 1u) << r[kMmc1ShiftCount]));
            if (++r[kMmc1ShiftCount] < 5u) {
                return;  // nothing changed yet
            }
            r[(address >> 13) - 4u] = r[kMmc1Shift];  // $8000/$A000/$C000/$E000 -> 0..3
            r[kMmc1Shift] = 0;
            r[kMmc1ShiftCount] = 0;
            if (buffers.cart.mapper == 105 && address < 0xE000u) {  // MMC1CHR -> NWCCHRHook, called for both halves
                const bool split = (r[kMmc1Control] & 0x10u) != 0;
                const std::uint8_t lo = split ? r[kMmc1Chr0] : static_cast<std::uint8_t>(r[kMmc1Chr0] & 0xFEu);
                const std::uint8_t hi = split ? r[kMmc1Chr1] : static_cast<std::uint8_t>(r[kMmc1Chr0] | 1u);
                if (((lo | hi) & 0x10u) != 0) {
                    r[7] = r[8] = r[9] = r[10] = 0;
                    buffers.cpu.irq_pending[env] = 0;
                }
                r[6] = hi;
            }
            break;
        case kBankingMmc2:
            switch (address & 0xF000u) {
                case 0xA000u: r[kMmc2Prg] = static_cast<std::uint8_t>(value & 0x0Fu); break;
                case 0xB000u: r[0] = static_cast<std::uint8_t>(value & 0x1Fu); break;
                case 0xC000u: r[1] = static_cast<std::uint8_t>(value & 0x1Fu); break;
                case 0xD000u: r[2] = static_cast<std::uint8_t>(value & 0x1Fu); break;
                case 0xE000u: r[3] = static_cast<std::uint8_t>(value & 0x1Fu); break;
                case 0xF000u: r[kMmc2Mirroring] = static_cast<std::uint8_t>((value & 1u) ^ 1u); break;
                default: return;  // $8000-$9FFF: no register
            }
            break;
        case kBankingMmc3:
            if (buffers.cart.mapper == 64 || buffers.cart.mapper == 158) {
                switch (address & 0xF001u) {
                    case 0x8000u: r[kRamboCmd] = value; return;
                    case 0x8001u:
                        if ((r[kRamboCmd] & 0x0Fu) < 10u) {
                            r[kRamboR + (r[kRamboCmd] & 0x0Fu)] = value;
                        } else if ((r[kRamboCmd] & 0x0Fu) == 0x0Fu) {
                            r[kRamboR + 10u] = value;
                        }
                        break;
                    case 0xA000u: r[kRamboMirror] = static_cast<std::uint8_t>((value & 1u) ^ 1u); break;
                    case 0xC000u:
                        r[kRamboLatch] = value;
                        if (r[kRamboRmode] == 1u) r[kRamboCount] = value;
                        return;
                    case 0xC001u:
                        r[kRamboRmode] = 1;
                        r[kRamboCount] = r[kRamboLatch];
                        r[kRamboIrqMode] = static_cast<std::uint8_t>(value & 1u);
                        return;
                    case 0xE000u:
                        r[kRamboIrqOn] = 0;
                        buffers.cpu.irq_pending[env] = 0;
                        if (r[kRamboRmode] == 1u) r[kRamboCount] = r[kRamboLatch];
                        return;
                    case 0xE001u:
                        r[kRamboIrqOn] = 1;
                        if (r[kRamboRmode] == 1u) r[kRamboCount] = r[kRamboLatch];
                        return;
                    default: return;
                }
                break;
            }
            if (buffers.cart.mapper == 114 || buffers.cart.mapper == 182) {  // M114Write: scrambled registers
                constexpr std::uint8_t kPerm[8] = {0, 3, 1, 5, 6, 7, 2, 4};
                switch (address & 0xE001u) {
                    case 0x8001u: address = 0xA000u; break;
                    case 0xA000u:
                        value = static_cast<std::uint8_t>((value & 0xC0u) | kPerm[value & 7u]);
                        address = 0x8000u;
                        r[kM114Armed] = 1;
                        break;
                    case 0xC000u:
                        if (r[kM114Armed] == 0) return;
                        r[kM114Armed] = 0;
                        address = 0x8001u;
                        break;
                    case 0xA001u: address = 0xC000u; break;
                    case 0xC001u: case 0xE000u: case 0xE001u: break;
                    default: return;  // $8000
                }
            }
            if (buffers.cart.mapper == 121 && address < 0xA000u) {  // M121Write
                std::uint8_t* ex = r + kM121Ex;
                switch (address & 0xE003u) {
                    case 0x8000u: break;
                    case 0x8001u:
                        ex[6] = static_cast<std::uint8_t>(((value & 1u) << 5) | ((value & 2u) << 3) | ((value & 4u) << 1) |
                                                          ((value & 8u) >> 1) | ((value & 0x10u) >> 3) | ((value & 0x20u) >> 5));
                        if (ex[7] == 0) m121_sync(ex);
                        break;
                    case 0x8003u:
                        ex[5] = value;
                        m121_sync(ex);
                        address = 0x8000u;
                        break;
                    default: return;
                }
            }
            switch (address & 0xE001u) {
                case 0x8000u:
                    if (buffers.cart.mapper == 187 && address == 0x8000u) r[kM187Unlock] = 1;  // M187Write8000
                    if (buffers.cart.mapper == 245 && ((value ^ r[kMmc3BankSelect]) & 0x80u) != 0) {
                        r[kMmc3Outer] = r[5];  // FixMMC3CHR rewrites every CHR bank, R5 last
                    }
                    r[kMmc3BankSelect] = value;
                    break;
                case 0x8001u:
                    if (buffers.cart.mapper == 187 && address == 0x8001u && r[kM187Unlock] == 0) return;
                    r[r[kMmc3BankSelect] & 7u] = value;
                    if (buffers.cart.mapper == 245 && (r[kMmc3BankSelect] & 7u) < 6u) r[kMmc3Outer] = value;
                    break;
                case 0xA000u: r[kMmc3Mirroring] = value; break;
                case 0xA001u: r[kMmc3PrgRam] = value; return;  // FCEUmm ignores PRG RAM protect on mapper 4
                case 0xC000u: r[kMmc3IrqLatch] = value; return;
                case 0xC001u: r[kMmc3IrqReload] = 1; return;
                case 0xE000u:
                    r[kMmc3IrqEnabled] = 0;
                    buffers.cpu.irq_pending[env] = 0;  // acknowledge
                    return;
                default:  // 0xE001
                    r[kMmc3IrqEnabled] = 1;
                    return;
            }
            break;
        default:
            return;
    }
    sync(buffers, env);
}

// RAMBO-1 (64) IRQ counter clock, as old FCEUmm's tengen.c (RAMBO1HBHook / RAMBO1IRQHook): the counter only
// counts down; the IRQ fires when it wraps from 0 to $FF. A line clock also sets rmode (1 on firing, else 0).
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE void rambo_clock(BatchBuffers& buffers, std::uint32_t env, bool line) {
    std::uint8_t* r = regs(buffers, env);
    if (line) r[kRamboRmode] = 0;
    --r[kRamboCount];
    if (r[kRamboCount] == 0xFFu && r[kRamboIrqOn] != 0) {
        if (line) r[kRamboRmode] = 1;
        buffers.cpu.irq_pending[env] = 1;
    }
}

// JY ClockCounter (FCEUmm 90.c): the prescaler and counter run whether or not the IRQ is enabled.
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE void jy_clock(BatchBuffers& buffers, std::uint32_t env,
                                                            std::uint32_t times) {
    std::uint8_t* r = regs(buffers, env);
    const std::uint32_t dir = r[kJyIrqMode] >> 6;
    if (dir != 1u && dir != 2u) return;
    const std::uint32_t premask = (r[kJyIrqMode] & 4u) != 0 ? 7u : 0xFFu;
    for (std::uint32_t i = 0; i < times; ++i) {
        if (dir == 1u) {
            ++r[kJyPre];
            if ((r[kJyPre] & premask) == 0u && ++r[kJyCount] == 0u && r[kJyIrqOn] != 0) buffers.cpu.irq_pending[env] = 1;
        } else {
            --r[kJyPre];
            if ((r[kJyPre] & premask) == premask && --r[kJyCount] == 0xFFu && r[kJyIrqOn] != 0) {
                buffers.cpu.irq_pending[env] = 1;
            }
        }
    }
}

// 83 (UNLYOKOIRQHook): the counter counts CPU cycles down; below zero it fires once and stops at $FFFF.
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE void m83_cycles(BatchBuffers& buffers, std::uint32_t env,
                                                              std::uint32_t cycles) {
    std::uint8_t* r = regs(buffers, env);
    if (r[kM83IrqOn] == 0) return;
    std::uint32_t count = r[kM83Count] | (static_cast<std::uint32_t>(r[kM83Count + 1]) << 8);
    if (cycles > count) {
        buffers.cpu.irq_pending[env] = 1;
        r[kM83IrqOn] = 0;
        count = 0xFFFFu;
    } else {
        count -= cycles;
    }
    r[kM83Count] = static_cast<std::uint8_t>(count & 0xFFu);
    r[kM83Count + 1] = static_cast<std::uint8_t>(count >> 8);
}

// VRC3 (M73IRQHook): 16-bit or 8-bit up-counter; on overflow it reloads and raises the IRQ.
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE void vrc3_cycles(BatchBuffers& buffers, std::uint32_t env,
                                                               std::uint32_t cycles) {
    std::uint8_t* r = regs(buffers, env);
    if (r[7] == 0) return;
    std::uint32_t count = r[3] | (static_cast<std::uint32_t>(r[4]) << 8);
    const std::uint32_t reload = r[1] | (static_cast<std::uint32_t>(r[2]) << 8);
    for (std::uint32_t i = 0; i < cycles; ++i) {
        if (r[5] != 0) {
            if ((count & 0xFFu) == 0xFFu) {
                count = reload;
                buffers.cpu.irq_pending[env] = 1;
            } else {
                count = (count & 0xFF00u) | ((count + 1u) & 0xFFu);
            }
        } else if (count == 0xFFFFu) {
            count = reload;
            buffers.cpu.irq_pending[env] = 1;
        } else {
            ++count;
        }
    }
    r[3] = static_cast<std::uint8_t>(count & 0xFFu);
    r[4] = static_cast<std::uint8_t>(count >> 8);
}

// CPU cycles elapsed on a board with a cycle-driven IRQ (VRC4, FME-7, RAMBO-1 cycle mode).
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE void cpu_cycles(BatchBuffers& buffers, std::uint32_t env,
                                                              std::uint32_t cycles) {
    std::uint8_t* r = regs(buffers, env);
    const std::uint8_t kind = buffers.cart.bank_kind;
    if (kind == kBankingM83) {
        m83_cycles(buffers, env, cycles);
        return;
    }
    if (kind == kBankingJy90) {  // CPUWrap: IRQ source 0 = M2
        if ((r[kJyIrqMode] & 3u) == 0) jy_clock(buffers, env, cycles);
        return;
    }
    if (kind == kBankingVrc3) {
        vrc3_cycles(buffers, env, cycles);
        return;
    }
    if (kind == kBankingBandai16) {  // BandaiIRQHook: int16 down-counter; below 0 fire once, stop, count = -1
        if (r[10] == 0) return;
        const std::int32_t count =
            static_cast<std::int16_t>(r[11] | (static_cast<std::uint32_t>(r[12]) << 8)) - static_cast<std::int32_t>(cycles);
        std::int16_t next = static_cast<std::int16_t>(count);
        if (count < 0) {
            buffers.cpu.irq_pending[env] = 1;
            r[10] = 0;
            next = -1;
        }
        r[11] = static_cast<std::uint8_t>(static_cast<std::uint16_t>(next) & 0xFFu);
        r[12] = static_cast<std::uint8_t>(static_cast<std::uint16_t>(next) >> 8);
        return;
    }
    if (kind == kBankingN163) {  // NamcoIRQHook: up-counter; at $7FFF fire once, stop, hold $7FFF
        if (r[25] == 0) return;
        std::uint32_t count = (r[23] | (static_cast<std::uint32_t>(r[24]) << 8)) + cycles;
        if ((count & 0xFFFFu) >= 0x7FFFu || count > 0xFFFFu) {
            buffers.cpu.irq_pending[env] = 1;
            r[25] = 0;
            count = 0x7FFFu;
        }
        r[23] = static_cast<std::uint8_t>(count);
        r[24] = static_cast<std::uint8_t>(count >> 8);
        return;
    }
    if (kind == kBankingFfe) {  // FFEIRQHook: up-counter; at $10000 fire once and stop
        if (r[2] == 0) return;
        const std::uint32_t count = (r[3] | (static_cast<std::uint32_t>(r[4]) << 8) |
                                     (static_cast<std::uint32_t>(r[5]) << 16)) + cycles;
        if (count >= 0x10000u) {
            buffers.cpu.irq_pending[env] = 1;
            r[2] = 0;
            r[3] = r[4] = r[5] = 0;
        } else {
            r[3] = static_cast<std::uint8_t>(count);
            r[4] = static_cast<std::uint8_t>(count >> 8);
            r[5] = 0;
        }
        return;
    }
    if (kind == kBankingMmc1) {  // mapper 105 NWCIRQHook: timer with the DIP switches at $E
        if ((r[6] & 0x10u) != 0) return;
        std::uint32_t count = r[7] | (static_cast<std::uint32_t>(r[8]) << 8) | (static_cast<std::uint32_t>(r[9]) << 16) |
                              (static_cast<std::uint32_t>(r[10]) << 24);
        count += cycles;
        if ((count | (0xEu << 25)) >= 0x3E000000u) {
            count = 0;
            buffers.cpu.irq_pending[env] = 1;
        }
        r[7] = static_cast<std::uint8_t>(count);
        r[8] = static_cast<std::uint8_t>(count >> 8);
        r[9] = static_cast<std::uint8_t>(count >> 16);
        r[10] = static_cast<std::uint8_t>(count >> 24);
        return;
    }
    if (kind == kBankingJaleco18) {  // M18IRQHook: count down; at or below 0 fire once and stop
        const std::uint32_t count = r[16] | (static_cast<std::uint32_t>(r[17]) << 8);
        if (r[13] == 0 || count == 0) return;
        if (cycles >= count) {
            buffers.cpu.irq_pending[env] = 1;
            r[16] = r[17] = 0;
            r[13] = 0;
        } else {
            r[16] = static_cast<std::uint8_t>((count - cycles) & 0xFFu);
            r[17] = static_cast<std::uint8_t>((count - cycles) >> 8);
        }
        return;
    }
    if (kind == kBankingM42) {  // M42IRQHook: 15-bit cycle count, line high from 24576
        if (r[3] == 0) return;
        std::uint32_t count = (r[4] | (static_cast<std::uint32_t>(r[5]) << 8)) + cycles;
        count &= 0x7FFFu;
        r[4] = static_cast<std::uint8_t>(count & 0xFFu);
        r[5] = static_cast<std::uint8_t>(count >> 8);
        buffers.cpu.irq_pending[env] = count >= 24576u ? 1 : 0;
        return;
    }
    if (kind == kBankingMmc3) {
        if ((buffers.cart.mapper != 64 && buffers.cart.mapper != 158) || r[kRamboIrqMode] == 0) return;
        std::uint32_t pre = r[kRamboPre] + cycles;  // RAMBO1IRQHook: one clock per 4 CPU cycles
        for (; pre >= 4u; pre -= 4u) rambo_clock(buffers, env, false);
        r[kRamboPre] = static_cast<std::uint8_t>(pre);
        return;
    }
    if (kind_is_vrc(kind)) {
        if ((r[kVrcIrqControl] & 2u) == 0) return;
        auto clock = [&]() {
            if (r[kVrcIrqCounter] == 0xFFu) {
                r[kVrcIrqCounter] = r[kVrcIrqLatch];
                buffers.cpu.irq_pending[env] = 1;
            } else {
                ++r[kVrcIrqCounter];
            }
        };
        if ((r[kVrcIrqControl] & 4u) != 0) {  // cycle mode
            for (std::uint32_t i = 0; i < cycles; ++i) clock();
            return;
        }
        auto pre = static_cast<std::int16_t>(r[kVrcPrescaler] | (r[kVrcPrescaler + 1] << 8));
        for (std::uint32_t i = 0; i < cycles; ++i) {
            pre = static_cast<std::int16_t>(pre - 3);
            if (pre <= 0) {
                pre = static_cast<std::int16_t>(pre + 341);
                clock();
            }
        }
        r[kVrcPrescaler] = static_cast<std::uint8_t>(pre & 0xFF);
        r[kVrcPrescaler + 1] = static_cast<std::uint8_t>((pre >> 8) & 0xFF);
    } else if (kind == kBankingSmb2j40) {
        if (r[1] == 0) return;
        std::uint32_t c = r[2] | (r[3] << 8);
        c += cycles;
        if (c >= 4096u) {
            r[1] = 0;
            c = 0;
            buffers.cpu.irq_pending[env] = 1;
        }
        r[2] = static_cast<std::uint8_t>(c & 0xFF);
        r[3] = static_cast<std::uint8_t>(c >> 8);
    } else if (kind == kBankingFme7) {
        if ((r[kFmeIrqControl] & 0x80u) == 0) return;
        std::uint32_t c = r[kFmeIrqCounter] | (r[kFmeIrqCounter + 1] << 8);
        if (c < cycles && (r[kFmeIrqControl] & 1u) != 0) buffers.cpu.irq_pending[env] = 1;
        c = (c - cycles) & 0xFFFFu;
        r[kFmeIrqCounter] = static_cast<std::uint8_t>(c & 0xFF);
        r[kFmeIrqCounter + 1] = static_cast<std::uint8_t>(c >> 8);
    }
}

// FME-7's $6000-$7FFF: ROM bank unless bit 6 (RAM) is set. Returns true and the byte when ROM is mapped.
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE bool read_6000_rom(const BatchBuffers& buffers, std::uint32_t env,
                                                                 std::uint16_t address, std::uint8_t& out) {
    if (buffers.cart.bank_kind == kBankingSmb2j40) {
        out = buffers.cart.prg_rom[((6u & buffers.cart.prg_page_mask) << 13) + (address & 0x1FFFu)];
        return true;
    }
    if (buffers.cart.bank_kind == kBankingJy90) {  // the last $6000 bank tekprom mapped, else FCEUmm's empty page (0)
        const std::uint8_t* r = regs(buffers, env);
        out = r[kJy6000On] != 0
                  ? buffers.cart.prg_rom[((r[kJy6000] & buffers.cart.prg_page_mask) << 13) + (address & 0x1FFFu)]
                  : std::uint8_t{0};
        return true;
    }
    if (buffers.cart.bank_kind == kBankingMmc3 && buffers.cart.mapper == 47) {  // WRAM nothing can write: zeros
        out = 0;
        return true;
    }
    if (buffers.cart.bank_kind == kBankingMmc5) {  // MMC5_ReadROMRAM at $6000: WRAMPage's bank, else X.DB
        const std::uint32_t idx = mmc5_wram_index(buffers.cart, regs(buffers, env)[kM5WramPage]);
        out = idx == 255u ? static_cast<std::uint8_t>(address >> 8)
                          : mmc5_ext(buffers, env)[kMmc5ExWram + (((idx * 8192u) & (buffers.cart.ext_ram_bytes -
                                                                                         kMmc5ExWram - 1u)) |
                                                                  (address & 0x1FFFu))];
        return true;
    }
    if (buffers.cart.bank_kind == kBankingBandai16) {  // BandaiRead: open bus with the EEPROM's data out in bit 4
        out = static_cast<std::uint8_t>(((address >> 8) & 0xEFu) | (regs(buffers, env)[21] << 4));
        return true;
    }
    if (buffers.cart.bank_kind == kBankingM188) {  // ExtDev
        out = 3;
        return true;
    }
    if (buffers.cart.bank_kind == kBankingM42) {
        out = buffers.cart.prg_rom[((regs(buffers, env)[0] & buffers.cart.prg_page_mask) << 13) + (address & 0x1FFFu)];
        return true;
    }
    if (buffers.cart.bank_kind != kBankingFme7) return false;
    const std::uint8_t reg = regs(buffers, env)[kFmePrg6];
    if ((reg & 0x40u) != 0) return false;
    out = buffers.cart.prg_rom[(((reg & 0x3Fu) & buffers.cart.prg_page_mask) << 13) + (address & 0x1FFFu)];
    return true;
}

// $4020-$5FFF reads the board answers. Returns true and the byte when it does.
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE bool read_expansion(const BatchBuffers& buffers, std::uint32_t env,
                                                                  std::uint16_t address, std::uint8_t& out) {
    if (buffers.cart.bank_kind == kBankingMmc3 && buffers.cart.mapper == 121 && address >= 0x5000u) {  // M121Read
        out = regs(buffers, env)[kM121Ex + 4];
        return true;
    }
    if (buffers.cart.bank_kind == kBankingMmc5 && address >= 0x5000u) {  // MMC5_read, MMC5_ExRAMRd
        std::uint8_t* r = regs(buffers, env);
        if (address >= 0x5C00u) {
            out = mmc5_ext(buffers, env)[kMmc5ExRam + (address & 0x3FFu)];
        } else if (address == 0x5204u) {
            buffers.cpu.irq_pending[env] = 0;
            out = r[kM5IrqStatus];
            r[kM5IrqStatus] = static_cast<std::uint8_t>(r[kM5IrqStatus] & 0x40u);
        } else if (address == 0x5205u || address == 0x5206u) {
            const std::uint32_t product = static_cast<std::uint32_t>(r[kM5Mul]) * r[kM5Mul + 1];
            out = static_cast<std::uint8_t>(address == 0x5205u ? product : product >> 8);
        } else {
            out = static_cast<std::uint8_t>(address >> 8);  // X.DB
        }
        return true;
    }
    if (buffers.cart.bank_kind == kBankingN163 && address >= 0x4800u) {
        std::uint8_t* r = regs(buffers, env);
        if (address < 0x5000u) {  // Namco_Read4800: the 128-byte sound RAM is not kept; reads 0, still steps
            out = 0;
            if ((r[26] & 0x80u) != 0) r[26] = static_cast<std::uint8_t>((r[26] & 0x80u) | ((r[26] + 1u) & 0x7Fu));
        } else {
            out = address < 0x5800u ? r[23] : r[24];  // Namco_Read5000 / Namco_Read5800
        }
        return true;
    }
    if (buffers.cart.bank_kind == kBankingM228 && address >= 0x5000u) {  // M228RamRead
        out = regs(buffers, env)[3u + (address & 3u)];
        return true;
    }
    if (buffers.cart.bank_kind == kBankingMmc3 && buffers.cart.mapper == 187 && address >= 0x5000u) {
        out = 0x83;  // M187Read: prot_data[unlock & 3], 0x83 for both values
        return true;
    }
    if (buffers.cart.bank_kind == kBankingJy90 && address >= 0x5000u) {  // M90TekRead
        const std::uint8_t* r = regs(buffers, env);
        const std::uint32_t product = static_cast<std::uint32_t>(r[kJyMul]) * r[kJyMul + 1];
        switch (address & 0x5C03u) {
            case 0x5800u: out = static_cast<std::uint8_t>(product & 0xFFu); break;
            case 0x5801u: out = static_cast<std::uint8_t>(product >> 8); break;
            case 0x5803u: out = r[kJyRegie]; break;
            default: out = 0; break;  // tekker (dip switch) at power-on
        }
        return true;
    }
    if (buffers.cart.bank_kind == kBankingM83 && address >= 0x5000u) {
        if (address == 0x5000u) {  // UNLYOKOReadDip: (X.DB & $FC) | dip 0; X.DB is the address high byte
            out = 0x50;
            return true;
        }
        if (address >= 0x5100u && address <= 0x5103u) {
            out = regs(buffers, env)[kM83Low + (address & 3u)];
            return true;
        }
    }
    return false;
}

// MMC3 scanline counter clock (FCEUmm ClockMMC3Counter, revision B). RAMBO-1 (64): RAMBO1HBHook.
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE void mmc3_clock(BatchBuffers& buffers, std::uint32_t env) {
    std::uint8_t* r = regs(buffers, env);
    if (buffers.cart.bank_kind == kBankingJy90) {  // SLWrap (GameHBIRQHook2): IRQ source 1, 8 clocks per line
        if ((r[kJyIrqMode] & 3u) == 1u) jy_clock(buffers, env, 8);
        return;
    }
    if (buffers.cart.bank_kind == kBankingM222) {  // M222IRQ: level IRQ once the counter reaches 238
        if (r[11] != 0) {
            ++r[12];
            if (r[12] >= 238u) buffers.cpu.irq_pending[env] = 1;
        }
        return;
    }
    if (buffers.cart.bank_kind == kBankingM117) {  // M117IRQHook
        if (r[13] == 3u && r[14] != 0 && --r[14] == 0) {
            r[13] = static_cast<std::uint8_t>(r[13] & 1u);
            buffers.cpu.irq_pending[env] = 1;
        }
        return;
    }
    if (buffers.cart.bank_kind == kBankingTaito48) {  // M48IRQ: fires when the 16-bit count reaches $100
        if (r[9] != 0) {
            const std::uint32_t count = ((r[11] | (static_cast<std::uint32_t>(r[12]) << 8)) + 1u) & 0xFFFFu;
            r[11] = static_cast<std::uint8_t>(count & 0xFFu);
            r[12] = static_cast<std::uint8_t>(count >> 8);
            if (count == 0x100u) {
                buffers.cpu.irq_pending[env] = 1;
                r[9] = 0;
            }
        }
        return;
    }
    if (buffers.cart.bank_kind == kBankingM91) {  // M91IRQHook: IRQ after 8 scanlines
        if (r[6] != 0 && r[7] < 8u && ++r[7] >= 8u) buffers.cpu.irq_pending[env] = 1;
        return;
    }
    if (buffers.cart.mapper == 64 || buffers.cart.mapper == 158) {
        if (r[kRamboIrqMode] == 0) rambo_clock(buffers, env, true);
        return;
    }
    const std::uint8_t before = r[kMmc3IrqCounter];
    if (r[kMmc3IrqCounter] == 0 || r[kMmc3IrqReload] != 0) {
        r[kMmc3IrqCounter] = r[kMmc3IrqLatch];
        r[kMmc3IrqReload] = 0;
    } else {
        --r[kMmc3IrqCounter];
    }
    // 114 (and 182) set isRevB = 0: a counter reloaded from 0 does not fire.
    const bool rev_a = buffers.cart.mapper == 114 || buffers.cart.mapper == 182;
    if (r[kMmc3IrqCounter] == 0 && r[kMmc3IrqEnabled] != 0 && (before != 0 || !rev_a)) {
        buffers.cpu.irq_pending[env] = 1;
    }
}

// MMC2 CHR latches: a PPU fetch from $0FD0-$0FDF/$0FE0-$0FEF (latch 0) or
// $1FD0-$1FDF/$1FE0-$1FEF (latch 1) selects the $FD or $FE bank of that half
// (FCEUmm's PPU_hook ranges). The CPU drives addresses through $2006/$2007 here;
// the frame's own tile and sprite fetches are replayed from the frame log
// (batch_frame.cuh), which draws with them and hands the final latches back to
// the board at vblank.
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE void mmc2_ppu_address(const BatchBuffers& buffers,
                                                                    std::uint32_t env,
                                                                    std::uint16_t address) {
    const std::uint32_t high = (address >> 8) & 0x3Fu;
    if (high >= 0x20u || (high & 0x0Fu) != 0x0Fu) {
        return;
    }
    const std::uint32_t low = address & 0xF0u;
    if (low != 0xD0u && low != 0xE0u) {
        return;
    }
    std::uint8_t* r = regs(buffers, env);
    const std::uint8_t latch = low == 0xE0u ? 1u : 0u;
    if (high < 0x10u) {
        r[kMmc2Latch0] = latch;
        set_chr4(buffers, env, 0, r[latch]);
    } else {
        r[kMmc2Latch1] = latch;
        set_chr4(buffers, env, 1, r[2u + latch]);
    }
}

NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE bool prg_ram_enabled(const BatchBuffers& buffers,
                                                                   std::uint32_t env) {
    // MMC1: PRG register bit 4 disables WRAM (FCEUmm MMC1WRAMEnabled).
    if (buffers.cart.bank_kind == kBankingFme7) return (regs(buffers, env)[kFmePrg6] & 0xC0u) == 0xC0u;
    if (buffers.cart.bank_kind == kBankingJy90 || buffers.cart.bank_kind == kBankingSa72008 ||
        buffers.cart.bank_kind == kBankingM243 || buffers.cart.bank_kind == kBankingCprom ||
        buffers.cart.bank_kind == kBankingBandai16 ||  // 16: prg_ram holds the EEPROM, not CPU-visible
        buffers.cart.bank_kind == kBankingMmc5) {      // 5: WRAM is in ext_ram
        return false;  // no RAM: ROM (read_6000_rom) or open bus
    }
    if (buffers.cart.mapper == 187 || buffers.cart.mapper == 119 || buffers.cart.mapper == 114 ||
        buffers.cart.mapper == 182) {
        return false;  // GenMMC3_Init(.., wram 0): no RAM
    }
    return buffers.cart.bank_kind != kBankingMmc1 || (regs(buffers, env)[kMmc1Prg] & 0x10u) == 0;
}

}  // namespace paged

// Hooks the bus calls; compiled out for the NROM and window instantiations. (The MMC2
// $2006/$2007 address hook is bus_ppu_address in batch_bus.cuh, which also logs it.)

// False while the board has its $6000-$7FFF RAM disabled.
template <BankMode Mode = kBankModeRuntime>
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE bool mapper_prg_ram_enabled(const BatchBuffers& buffers,
                                                                          std::uint32_t env) {
    if constexpr (Mode == kBankModePaged || Mode == kBankModeRuntime) {
        if (cart_is_paged<Mode>(buffers.cart) && buffers.mapper.regs != nullptr) {
            return paged::prg_ram_enabled(buffers, env);
        }
    }
    return true;
}

// True while the board holds the CPU IRQ line low.
template <BankMode Mode = kBankModeRuntime>
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE bool mapper_irq_asserted(const BatchBuffers& buffers,
                                                                       std::uint32_t env) {
    if constexpr (Mode == kBankModePaged || Mode == kBankModeRuntime) {
        return cart_is_paged<Mode>(buffers.cart) && buffers.cpu.irq_pending != nullptr &&
               buffers.cpu.irq_pending[env] != 0;
    } else {
        return false;
    }
}

// CPU cycles for boards that count them (compiled out for NROM and window boards).
template <BankMode Mode = kBankModeRuntime>
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE void mapper_cpu_cycles(BatchBuffers& buffers, std::uint32_t env,
                                                                     std::uint32_t cycles) {
    if constexpr (Mode == kBankModePaged || Mode == kBankModeRuntime) {
        const std::uint8_t kind = buffers.cart.bank_kind;
        if ((paged::kind_is_vrc(kind) || kind == kBankingFme7 || kind == kBankingSmb2j40 ||
             kind == kBankingM42 || kind == kBankingM83 || kind == kBankingJy90 || kind == kBankingVrc3 ||
             kind == kBankingJaleco18 || kind == kBankingBandai16 || kind == kBankingFfe || kind == kBankingN163 || (kind == kBankingMmc1 && buffers.cart.mapper == 105) ||
             (kind == kBankingMmc3 && (buffers.cart.mapper == 64 || buffers.cart.mapper == 158))) &&
            buffers.mapper.regs != nullptr) {
            paged::cpu_cycles(buffers, env, cycles);
        }
    }
}

// ---------------------------------------------------------------- PRG reads

// NROM: the whole 16 or 32 KB image sits at $8000. Kept byte-identical to the
// pre-UxROM read (including the 16 KB special case) so mapper 0 ROMs keep
// their exact behavior and their hot-path cost.
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE std::uint8_t read_nrom_prg(const CartridgeView& cart,
                                                       std::uint16_t address) {
    auto index = static_cast<std::uint32_t>(address - 0x8000);
    if (cart.prg_rom_size == 16u * 1024u) {
        index &= 0x3FFFu;
    } else {
        index &= 0x7FFFu;
    }
    return cart.prg_rom[index];
}

template <BankMode Mode = kBankModeRuntime>
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE std::uint8_t read_prg(const BatchBuffers& buffers,
                                                  std::uint32_t env,
                                                  std::uint16_t address) {
    const CartridgeView& cart = buffers.cart;
    if constexpr (bank_mode_is_nrom(Mode)) {
        return read_nrom_prg(cart, address);
    } else {
        if constexpr (Mode == kBankModePaged || Mode == kBankModeRuntime) {
            if (cart_is_paged<Mode>(cart)) {
                const std::uint32_t page =
                    buffers.mapper.prg_page[static_cast<std::uint64_t>(env) * kPrgPages +
                                            ((address >> 13) & 3u)];
                if ((page & (kPrgPageRam | kPrgPageOpen)) != 0) {  // MMC5 WRAM in $8000-$DFFF, or nothing there
                    if ((page & kPrgPageOpen) != 0) return static_cast<std::uint8_t>(address >> 8);
                    return buffers.mapper.ext_ram[static_cast<std::uint64_t>(env) * cart.ext_ram_bytes + kMmc5ExWram +
                                                  ((page & 0xFFFFFu) | (address & 0x1FFFu))];
                }
                return cart.prg_rom[page + (address & 0x1FFFu)];
            }
        }
        if constexpr (Mode != kBankModeBanked) {
            if (cart.bank_kind == kBankingNone) {
                return read_nrom_prg(cart, address);
            }
        }
        std::uint32_t index;
        // The window sits at the bottom of CPU space. address - $8000 wraps to
        // a huge value for the fixed half, so one unsigned compare covers both.
        const std::uint32_t offset = static_cast<std::uint32_t>(address) - cart.prg_window_start;
        if (offset <= cart.prg_window_mask) {
            const std::uint32_t bank = buffers.mapper.prg_bank[env] & cart.prg_bank_mask;
            index = (bank << cart.prg_window_shift) + offset;
        } else {
            index = cart.prg_fixed_base + (address & cart.prg_fixed_mask);
        }

        // prg_rom_size is the padded (power of two) size, so the mask wraps
        // into the padding rather than reading out of range.
        return cart.prg_rom[index & cart.prg_rom_mask];
    }
}

// ------------------------------------------------------------ mapper writes

// A write anywhere in $8000-$FFFF latches the switchable bank. UxROM boards
// ignore the address.
template <BankMode Mode = kBankModeRuntime>
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE void write_mapper_register(BatchBuffers& buffers,
                                                       std::uint32_t env,
                                                       std::uint16_t address,
                                                       std::uint8_t value) {
    const CartridgeView& cart = buffers.cart;
    if constexpr (bank_mode_is_nrom(Mode)) {
        return;  // mapper 0 has no registers; the write goes nowhere
    } else {
        if constexpr (Mode != kBankModeBanked) {
            if (cart.bank_kind == kBankingNone) {
                return;
            }
        }
        if (address < 0x8000 && !cart_is_paged<Mode>(cart)) {
            // Window boards: a write to $6000-$7FFF reaches only WRAM.
            return;
        }
        if (cart_is_paged<Mode>(cart)) {
            const std::uint8_t rom_byte =
                cart.bus_conflicts != 0 ? read_prg<Mode>(buffers, env, address) : 0;
            paged::write(buffers, env, address, value, rom_byte);
            return;
        }

        std::uint8_t latched = value;
        if (cart.bus_conflicts != 0) {
            // The cartridge's mask ROM drives 0s more strongly than the CPU data
            // bus, so only bits the ROM byte already has set survive the write.
            latched = static_cast<std::uint8_t>(value & read_prg<Mode>(buffers, env, address));
        }

        buffers.mapper.prg_bank[env] =
            static_cast<std::uint8_t>((latched >> cart.prg_bank_shift) & cart.prg_bank_mask);
    }
}

// ------------------------------------------------------------------ CHR I/O

template <BankMode Mode = kBankModeRuntime>
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE std::uint8_t read_chr(const BatchBuffers& buffers,
                                                  std::uint32_t env,
                                                  std::uint16_t address) {
    const CartridgeView& cart = buffers.cart;
    if constexpr (Mode == kBankModePaged || Mode == kBankModeRuntime) {
        if (cart_is_paged<Mode>(cart)) {
            const std::uint32_t offset =
                buffers.mapper.chr_page[static_cast<std::uint64_t>(env) * kChrPages +
                                        ((address >> 10) & 7u)] +
                (address & 0x3FFu);
            if ((offset & kChrPageRam) != 0 && cart.mapper == 74) {  // 74's 2 KB CHR RAM is its ext_ram
                return buffers.mapper.ext_ram[static_cast<std::uint64_t>(env) * cart.ext_ram_bytes + (offset & 0x7FFu)];
            }
            if ((offset & kChrPageRam) != 0) {  // TQROM / CPROM CHR RAM kept in prg_ram
                return buffers.cpu.prg_ram[static_cast<std::uint64_t>(env) * kPrgRamBytes + (offset & 0x1FFFu)];
            }
            if (cart.chr_rom != nullptr) {
                return cart.chr_rom[offset];
            }
            return buffers.ppu.chr_ram[static_cast<std::uint64_t>(env) * kChrRamBytes +
                                       (offset & 0x1FFFu)];
        }
    }
    if (cart.chr_rom == nullptr) {
        // CHR RAM board: reads back what the game uploaded. 8 KB, mirrored
        // through $1FFF, which is what every mapper 2/94 board wires.
        if (buffers.ppu.chr_ram == nullptr) {
            return 0;
        }
        return buffers.ppu.chr_ram[static_cast<std::uint64_t>(env) * kChrRamBytes +
                                   (address & 0x1FFFu)];
    }
    // Fixed CHR wiring. The modulo is the pre-UxROM behavior and handles
    // non-power-of-two CHR ROM images, which some NROM dumps still ship.
    return cart.chr_rom[address % cart.chr_rom_size];
}


template <BankMode Mode = kBankModeRuntime>
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE void write_chr(BatchBuffers& buffers,
                                           std::uint32_t env,
                                           std::uint16_t address,
                                           std::uint8_t value) {
    if (buffers.cart.chr_rom != nullptr) {
        if constexpr (Mode == kBankModePaged || Mode == kBankModeRuntime) {
            if (cart_is_paged<Mode>(buffers.cart)) {  // TQROM: the CHR RAM pages take the write
                const std::uint32_t page =
                    buffers.mapper.chr_page[static_cast<std::uint64_t>(env) * kChrPages + ((address >> 10) & 7u)];
                if ((page & kChrPageRam) != 0 && buffers.cart.mapper == 74) {
                    buffers.mapper.ext_ram[static_cast<std::uint64_t>(env) * buffers.cart.ext_ram_bytes +
                                           ((page + (address & 0x3FFu)) & 0x7FFu)] = value;
                } else if ((page & kChrPageRam) != 0) {
                    buffers.cpu.prg_ram[static_cast<std::uint64_t>(env) * kPrgRamBytes +
                                        ((page + (address & 0x3FFu)) & 0x1FFFu)] = value;
                }
            }
        }
        return;  // CHR ROM board: the write goes nowhere
    }
    if (buffers.ppu.chr_ram == nullptr) {
        return;
    }
    if constexpr (Mode == kBankModePaged || Mode == kBankModeRuntime) {
        if (cart_is_paged<Mode>(buffers.cart)) {
            const std::uint32_t offset =
                buffers.mapper.chr_page[static_cast<std::uint64_t>(env) * kChrPages +
                                        ((address >> 10) & 7u)] +
                (address & 0x3FFu);
            if ((offset & kChrPageRam) != 0) {  // CPROM's second 8 KB
                buffers.cpu.prg_ram[static_cast<std::uint64_t>(env) * kPrgRamBytes + (offset & 0x1FFFu)] = value;
                return;
            }
            buffers.ppu.chr_ram[static_cast<std::uint64_t>(env) * kChrRamBytes + (offset & 0x1FFFu)] =
                value;
            return;
        }
    }
    buffers.ppu.chr_ram[static_cast<std::uint64_t>(env) * kChrRamBytes + (address & 0x1FFFu)] =
        value;
}

// --------------------------------------------------------------- mirroring

// Runtime arrangement for this env. No board here changes mirroring at run
// time, so this is the header's arrangement; kept as a function so the read
// sites stay uniform. The template parameter matches the other accessors so
// call sites can pass their BankMode through unchanged.
template <BankMode Mode = kBankModeRuntime>
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE std::uint8_t env_nametable_arrangement(
    const BatchBuffers& buffers, std::uint32_t) {
    return buffers.cart.nametable_arrangement;
}

// A four-screen cart whose nametables 2-3 live in ext_ram (kFourScreenLive); MMC5 and 74 use ext_ram otherwise.
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE bool four_screen_ram(const BatchBuffers& b) {
    return b.cart.nametable_arrangement == kNametableFourScreen && b.mapper.ext_ram != nullptr &&
           b.cart.bank_kind != kBankingMmc5 && b.cart.mapper != 74;
}

NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE std::uint16_t mirror_nametable_address(std::uint8_t arrangement,
                                                                   std::uint16_t address) {
    const auto index = static_cast<std::uint16_t>((address - 0x2000) & 0x0FFF);
    switch (arrangement) {
        case kNametableFourScreen:
            // Only 2 KB of CIRAM is allocated per env, so a four-screen cart
            // folds to vertical instead of running off the end of its block.
            return static_cast<std::uint16_t>(index & 0x07FF);
        case kNametableHorizontal:
            return static_cast<std::uint16_t>((index & 0x03FF) | ((index & 0x0800) >> 1));
        default:
            return static_cast<std::uint16_t>(index & 0x07FF);
    }
}

// CIRAM offset of a $2000-$3EFF address for this env. Paged boards look the
// logical nametable up in their run-time page map; every other board keeps the
// header's arrangement (and its exact pre-mapper code path).
template <BankMode Mode = kBankModeRuntime>
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE std::uint16_t env_mirror_nametable_address(
    const BatchBuffers& buffers, std::uint32_t env, std::uint16_t address) {
    if constexpr (Mode == kBankModePaged || Mode == kBankModeRuntime) {
        if (cart_is_paged<Mode>(buffers.cart) && buffers.mapper.mirroring != nullptr) {
            const std::uint32_t index = static_cast<std::uint32_t>(address - 0x2000) & 0x0FFFu;
            const std::uint32_t page = (buffers.mapper.mirroring[env] >> (index >> 10)) & 1u;
            return static_cast<std::uint16_t>((page << 10) | (index & 0x03FFu));
        }
    }
    return mirror_nametable_address(env_nametable_arrangement<Mode>(buffers, env), address);
}

// ------------------------------------------------------------------ resets

// Power-on state. UxROM boards latch bank 0 in the switchable window; the fixed
// window is not a register at all (it is the last page of the image). FCEUX
// save states carry no mapper registers, so a snapshot restore starts here too.
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE void reset_mapper_state(BatchBuffers& buffers, std::uint32_t env) {
    const MapperStateSoA& map = buffers.mapper;
    if (map.prg_bank != nullptr) {
        map.prg_bank[env] = 0;
    }
    if (buffers.cart.bank_kind >= kBankingFirstPaged && map.regs != nullptr) {
        paged::power_on(buffers, env);
    }
}

}  // namespace nesle::cuda

#undef NESLE_CUDA_MAPPER_HD
#undef NESLE_CUDA_MAPPER_INLINE
