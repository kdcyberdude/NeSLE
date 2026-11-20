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
                        kMmc3IrqReload = 14, kMmc3Outer = 15;  // [15]: 189's 32 KB PRG register
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

// Recompute the page tables and mirroring from the registers.
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
            set_prg8(buffers, env, 0, r[kMmc2Prg]);
            set_prg8(buffers, env, 1, ~2u);
            set_prg8(buffers, env, 2, ~1u);
            set_prg8(buffers, env, 3, ~0u);
            set_chr4(buffers, env, 0, r[r[kMmc2Latch0] & 1u]);
            set_chr4(buffers, env, 1, r[2u + (r[kMmc2Latch1] & 1u)]);
            set_mirroring(buffers, env, r[kMmc2Mirroring] != 0 ? kPageMapVertical : kPageMapHorizontal);
            break;
        }
        case kBankingMmc3: {
            if (buffers.cart.mapper == 64) {
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
        case kBankingFme7:
            r[kFmePrg + 1] = 1;
            r[kFmePrg + 2] = 2;
            r[kFmeMirror] = buffers.cart.nametable_arrangement == kNametableHorizontal ? 1u : 0u;
            break;
        case kBankingMmc3:
            if (buffers.cart.mapper == 64) {  // RAMBO1Power: every bank register ~0, vertical mirroring
                for (std::uint32_t i = 0; i < 11u; ++i) r[kRamboR + i] = 0xFF;
                r[kRamboMirror] = 1;
                break;
            }
            r[1] = 2;
            r[2] = 4;
            r[3] = 5;
            r[4] = 6;
            r[5] = 7;
            r[7] = 1;
            // M4Power: mirroring from the header.
            r[kMmc3Mirroring] =
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
        } else if (kind == kBankingNina001) {
            if (address < 0x7FFDu) return;
            r[address - 0x7FFDu] = value;
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
            // second write of a read-modify-write). This CPU core issues one write
            // per instruction, so that case cannot arise here.
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
            if (buffers.cart.mapper == 64) {
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
            switch (address & 0xE001u) {
                case 0x8000u:
                    if (buffers.cart.mapper == 245 && ((value ^ r[kMmc3BankSelect]) & 0x80u) != 0) {
                        r[kMmc3Outer] = r[5];  // FixMMC3CHR rewrites every CHR bank, R5 last
                    }
                    r[kMmc3BankSelect] = value;
                    break;
                case 0x8001u:
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

// RAMBO-1 (64) IRQ counter clock (FCEUmm: reload when 0, fire on reaching 0).
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE void rambo_clock(BatchBuffers& buffers, std::uint32_t env) {
    std::uint8_t* r = regs(buffers, env);
    if (r[kRamboCount] == 0) {
        r[kRamboCount] = r[kRamboLatch];
    } else {
        --r[kRamboCount];
    }
    if (r[kRamboCount] == 0 && r[kRamboIrqOn] != 0) {
        buffers.cpu.irq_pending[env] = 1;
    }
}

// CPU cycles elapsed on a board with a cycle-driven IRQ (VRC4, FME-7, RAMBO-1 cycle mode).
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE void cpu_cycles(BatchBuffers& buffers, std::uint32_t env,
                                                              std::uint32_t cycles) {
    std::uint8_t* r = regs(buffers, env);
    const std::uint8_t kind = buffers.cart.bank_kind;
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
        if (buffers.cart.mapper != 64 || r[kRamboIrqMode] == 0) return;
        std::uint32_t pre = r[kRamboPre] + cycles;  // RAMBO1IRQHook: one clock per 4 CPU cycles
        for (; pre >= 4u; pre -= 4u) rambo_clock(buffers, env);
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

// MMC3 scanline counter clock (FCEUmm ClockMMC3Counter, revision B). RAMBO-1 (64): RAMBO1HBHook.
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE void mmc3_clock(BatchBuffers& buffers, std::uint32_t env) {
    std::uint8_t* r = regs(buffers, env);
    if (buffers.cart.bank_kind == kBankingM91) {  // M91IRQHook: IRQ after 8 scanlines
        if (r[6] != 0 && r[7] < 8u && ++r[7] >= 8u) buffers.cpu.irq_pending[env] = 1;
        return;
    }
    if (buffers.cart.mapper == 64) {
        if (r[kRamboIrqMode] == 0) {
            r[kRamboRmode] = 0;
            rambo_clock(buffers, env);
        }
        return;
    }
    if (r[kMmc3IrqCounter] == 0 || r[kMmc3IrqReload] != 0) {
        r[kMmc3IrqCounter] = r[kMmc3IrqLatch];
        r[kMmc3IrqReload] = 0;
    } else {
        --r[kMmc3IrqCounter];
    }
    if (r[kMmc3IrqCounter] == 0 && r[kMmc3IrqEnabled] != 0) {
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
             kind == kBankingM42 || (kind == kBankingMmc3 && buffers.cart.mapper == 64)) &&
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
