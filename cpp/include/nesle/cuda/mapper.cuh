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

constexpr std::uint32_t kMmc1Control = 0, kMmc1Chr0 = 1, kMmc1Chr1 = 2, kMmc1Prg = 3,
                        kMmc1Shift = 4, kMmc1ShiftCount = 5;
constexpr std::uint32_t kMmc2Prg = 4, kMmc2Mirroring = 5, kMmc2Latch0 = 6, kMmc2Latch1 = 7;
constexpr std::uint32_t kMmc3BankSelect = 8, kMmc3Mirroring = 9, kMmc3PrgRam = 10,
                        kMmc3IrqLatch = 11, kMmc3IrqCounter = 12, kMmc3IrqEnabled = 13,
                        kMmc3IrqReload = 14;

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
            set_mirroring(buffers, env,
                          (r[kMmc3Mirroring] & 1u) != 0 ? kPageMapHorizontal : kPageMapVertical);
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
        case kBankingMmc3:
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
    switch (buffers.cart.bank_kind) {
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
            switch (address & 0xE001u) {
                case 0x8000u: r[kMmc3BankSelect] = value; break;
                case 0x8001u: r[r[kMmc3BankSelect] & 7u] = value; break;
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

// MMC3 scanline counter clock (FCEUmm ClockMMC3Counter, revision B).
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE void mmc3_clock(BatchBuffers& buffers, std::uint32_t env) {
    std::uint8_t* r = regs(buffers, env);
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
        return buffers.cart.bank_kind == kBankingMmc3 && buffers.cpu.irq_pending != nullptr &&
               buffers.cpu.irq_pending[env] != 0;
    } else {
        return false;
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
        if (address < 0x8000) {
            // The mapper register is in the CPU space; a write to $6000-$7FFF
            // reaches only WRAM.
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
