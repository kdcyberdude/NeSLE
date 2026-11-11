#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

#include "nesle/rom.hpp"

namespace nesle::fcs {

inline constexpr std::size_t kCpuRamBytes = 2048;
inline constexpr std::size_t kPrgRamBytes = 8 * 1024;
inline constexpr std::size_t kPaletteRamBytes = 32;
inline constexpr std::size_t kOamBytes = 256;
inline constexpr std::size_t kNametableRamBytes = 2048;
inline constexpr std::size_t kChrRamBytes = 8 * 1024;

struct StateSnapshot {
    std::uint16_t pc = 0;
    std::uint8_t a = 0;
    std::uint8_t x = 0;
    std::uint8_t y = 0;
    std::uint8_t sp = 0xFD;
    std::uint8_t p = 0x24;
    std::uint64_t cycles = 0;

    std::array<std::uint8_t, kCpuRamBytes> cpu_ram{};
    std::array<std::uint8_t, kPrgRamBytes> prg_ram{};

    std::uint8_t ppu_ctrl = 0;
    std::uint8_t ppu_mask = 0;
    std::uint8_t ppu_status = 0;
    std::uint8_t ppu_oam_addr = 0;
    std::uint8_t ppu_open_bus = 0;
    std::uint8_t ppu_read_buffer = 0;
    std::uint8_t ppu_x = 0;
    std::uint8_t ppu_w = 0;
    std::uint16_t ppu_v = 0;
    std::uint16_t ppu_t = 0;

    std::array<std::uint8_t, kNametableRamBytes> nametable_ram{};
    std::array<std::uint8_t, kPaletteRamBytes> palette_ram{};
    std::array<std::uint8_t, kOamBytes> oam{};

    // Cartridge section. CHR RAM boards (every UxROM/BNROM cartridge) put their
    // 8 KB of pattern tables here: without it a restored env renders from
    // whatever the previous episode left behind.
    std::array<std::uint8_t, kChrRamBytes> chr_ram{};
    bool has_chr_ram = false;

    // Raw mapper latch as the board saw it: "LATC" for mappers 2 and 94, which
    // still needs the board's shift, and "REGS"[0] for mapper 34, which is the
    // 32 KB page number already. Converted with the cartridge view's shift and
    // mask, so both reduce to the same expression.
    std::uint8_t cart_latch = 0;
    bool has_cart_latch = false;

    // Paged-board registers (section 16) under FCEUmm's chunk names. A state carries
    // only its own board's chunks; board_registers() maps them onto NeSLE's register
    // layout (cuda/mapper.cuh) once the board is known.
    std::array<std::uint8_t, 8> board_regs{};  // MMC1 "DREG" (4), MMC2 "CREG" (4), MMC3 "REGS" (8)
    std::uint8_t board_regs_size = 0;
    std::uint8_t mmc1_shift = 0;        // "BFFR"
    std::uint8_t mmc1_shift_count = 0;  // "BFRS"
    std::uint8_t mmc2_prg = 0;          // "PREG"
    std::uint8_t mmc2_mirroring = 0;    // "MIRR": 1 = vertical, as the board stores it
    std::uint8_t mmc2_latch0 = 1;       // "LAT0": 0 = $FD bank, 1 = $FE bank
    std::uint8_t mmc2_latch1 = 1;       // "LAT1"
    std::uint8_t mmc3_cmd = 0;          // "CMD"
    std::uint8_t mmc3_a000 = 0;         // "A000" (mirroring)
    std::uint8_t mmc3_a001 = 0;         // "A001"
    std::uint8_t mmc3_irq_reload = 0;   // "IRQR"
    std::uint8_t mmc3_irq_count = 0;    // "IRQC"
    std::uint8_t mmc3_irq_latch = 0;    // "IRQL"
    std::uint8_t mmc3_irq_enabled = 0;  // "IRQA"
    std::uint32_t board_chunks = 0;     // kBoard* bits of the chunks seen

    // CPU section 2 (X6502 internals): the IRQ line bits ("IQLB"; bit 0 is the
    // cartridge's line) and the CPU's distance from the end of the frame it was saved
    // at ("ICou", in 1/48 CPU cycles; negative = the last instruction ran past it).
    std::uint32_t cpu_irq_low = 0;
    std::int32_t cpu_count = 0;
    // PPU "KOOK": FCEUmm's odd/even frame toggle for the frame that comes next.
    std::uint8_t ppu_kook = 0;
};

inline constexpr std::uint32_t kBoardDreg = 1u << 0, kBoardBffr = 1u << 1, kBoardBfrs = 1u << 2,
                               kBoardCreg = 1u << 3, kBoardPreg = 1u << 4, kBoardMirr = 1u << 5,
                               kBoardLat0 = 1u << 6, kBoardLat1 = 1u << 7, kBoardRegs = 1u << 8,
                               kBoardCmd = 1u << 9, kBoardA000 = 1u << 10, kBoardIrq = 1u << 11;

namespace detail {

[[nodiscard]] inline std::uint32_t read_le_u32(std::span<const std::uint8_t> bytes) {
    if (bytes.size() < 4) {
        throw std::runtime_error("FCS: truncated u32");
    }
    return static_cast<std::uint32_t>(bytes[0]) |
           (static_cast<std::uint32_t>(bytes[1]) << 8) |
           (static_cast<std::uint32_t>(bytes[2]) << 16) |
           (static_cast<std::uint32_t>(bytes[3]) << 24);
}

[[nodiscard]] inline std::uint64_t read_le_u64(std::span<const std::uint8_t> bytes) {
    if (bytes.size() < 8) {
        throw std::runtime_error("FCS: truncated u64");
    }
    std::uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= static_cast<std::uint64_t>(bytes[i]) << (8 * i);
    }
    return value;
}

[[nodiscard]] inline std::string_view strip_trailing_nuls(std::string_view name) {
    while (!name.empty() && name.back() == '\0') {
        name.remove_suffix(1);
    }
    return name;
}

// Copy a sub-chunk payload into a fixed-size destination. Throws if the chunk is the wrong
// size — FCEUX is consistent enough that any mismatch usually means we're parsing the wrong
// offset.
template <std::size_t N>
inline void copy_fixed(std::span<const std::uint8_t> src, std::array<std::uint8_t, N>& dst,
                       std::string_view label) {
    if (src.size() != N) {
        throw std::runtime_error(std::string("FCS: ") + std::string(label) +
                                 " expected " + std::to_string(N) + " bytes, got " +
                                 std::to_string(src.size()));
    }
    std::memcpy(dst.data(), src.data(), N);
}

inline void apply_cpu_subchunk(StateSnapshot& out, std::string_view name,
                               std::span<const std::uint8_t> payload) {
    if (name == "PC") {
        if (payload.size() != 2) {
            throw std::runtime_error("FCS: CPU.PC must be 2 bytes");
        }
        out.pc = static_cast<std::uint16_t>(payload[0] | (payload[1] << 8));
    } else if (name == "A" && payload.size() == 1) {
        out.a = payload[0];
    } else if (name == "X" && payload.size() == 1) {
        out.x = payload[0];
    } else if (name == "Y" && payload.size() == 1) {
        out.y = payload[0];
    } else if (name == "S" && payload.size() == 1) {
        out.sp = payload[0];
    } else if (name == "P" && payload.size() == 1) {
        out.p = payload[0];
    } else if (name == "DB" && payload.size() == 1) {
        // CPU "data bus" open-bus byte; ignored — we don't expose CPU open bus.
    } else if (name == "RAM") {
        copy_fixed(payload, out.cpu_ram, "CPU.RAM");
    }
    // Other CPU sub-chunks (IQLB, ICoa, ICou, TSBS, JAMM) are emulator-internal IRQ
    // bookkeeping and a CPU timestamp; we don't restore them. TSBS could be used for
    // cycle-counter restore but our cycles field is relative, so 0 is fine.
}

inline void apply_ppu_subchunk(StateSnapshot& out, std::string_view name,
                               std::span<const std::uint8_t> payload) {
    if (name == "NTAR") {
        copy_fixed(payload, out.nametable_ram, "PPU.NTAR");
    } else if (name == "PRAM") {
        copy_fixed(payload, out.palette_ram, "PPU.PRAM");
    } else if (name == "SPRA") {
        copy_fixed(payload, out.oam, "PPU.SPRA");
    } else if (name == "PPUR") {
        // PPUR packs [PPUCTRL, PPUMASK, PPUSTATUS, ?]. Fourth byte is open-bus / unused.
        if (payload.size() < 3) {
            throw std::runtime_error("FCS: PPU.PPUR too short");
        }
        out.ppu_ctrl = payload[0];
        out.ppu_mask = payload[1];
        out.ppu_status = payload[2];
    } else if (name == "PSPL" && payload.size() == 1) {
        out.ppu_oam_addr = payload[0];
    } else if (name == "XOFF" && payload.size() == 1) {
        out.ppu_x = static_cast<std::uint8_t>(payload[0] & 0x07);
    } else if (name == "VTGL" && payload.size() == 1) {
        out.ppu_w = static_cast<std::uint8_t>(payload[0] & 0x01);
    } else if (name == "RADD" && payload.size() == 2) {
        out.ppu_v = static_cast<std::uint16_t>(payload[0] | (payload[1] << 8));
    } else if (name == "TADD" && payload.size() == 2) {
        out.ppu_t = static_cast<std::uint16_t>(payload[0] | (payload[1] << 8));
    } else if (name == "VBUF" && payload.size() == 1) {
        out.ppu_read_buffer = payload[0];
    } else if (name == "PGEN" && payload.size() == 1) {
        out.ppu_open_bus = payload[0];
    } else if (name == "KOOK" && payload.size() == 1) {
        out.ppu_kook = static_cast<std::uint8_t>(payload[0] & 1u);
    }
    // DEAD (power-on frames left) is ignored.
}

inline void apply_cpu_internal_subchunk(StateSnapshot& out, std::string_view name,
                                        std::span<const std::uint8_t> payload) {
    if (name == "IQLB" && payload.size() == 4) {
        out.cpu_irq_low = read_le_u32(payload);
    } else if (name == "ICou" && payload.size() == 4) {
        out.cpu_count = static_cast<std::int32_t>(read_le_u32(payload));
    }
}

inline void apply_cart_subchunk(StateSnapshot& out, std::string_view name,
                                std::span<const std::uint8_t> payload) {
    if (name == "WRAM") {
        // FCEUX always stores 8 KB of WRAM even for cartridges (like SMB / NROM) that don't
        // have it. Copying it through is harmless - our prg_ram lives at $6000-$7FFF and
        // NROM reads return 0 if not wired.
        copy_fixed(payload, out.prg_ram, "CART.WRAM");
    } else if (name == "CHRR") {
        // 8 KB of CHR RAM. Present on every board with CHR RAM (mappers 2/34/94
        // here); absent on CHR ROM carts, which is why has_chr_ram exists instead
        // of treating a missing tag as zeroes.
        copy_fixed(payload, out.chr_ram, "CART.CHRR");
        out.has_chr_ram = true;
    } else if (name == "LATC") {
        // Latch_Init registers the raw value written to the bank register, not
        // the bank number - mapper 94 needs it shifted. Sizes other than one byte
        // mean we are parsing something else, so leave it alone rather than
        // abort: the latch is optional in old saves.
        if (payload.size() == 1) {
            out.cart_latch = payload[0];
            out.has_cart_latch = true;
        }
    } else if (name == "REGS") {
        // Mapper 34 keeps PRG bank in regs[0]; regs[1] and regs[2] are its two
        // 4 KB CHR banks, which only matter on the NINA-001 board that this
        // emulator refuses.
        if (payload.size() >= 1) {
            out.cart_latch = payload[0];
            out.has_cart_latch = true;
        }
        // MMC3 keeps R0-R7 here.
        if (payload.size() == 8) {
            std::memcpy(out.board_regs.data(), payload.data(), 8);
            out.board_regs_size = 8;
            out.board_chunks |= kBoardRegs;
        }
    } else if ((name == "DREG" || name == "CREG") && payload.size() == 4) {
        std::memcpy(out.board_regs.data(), payload.data(), 4);
        out.board_regs_size = 4;
        out.board_chunks |= name == "DREG" ? kBoardDreg : kBoardCreg;
    } else if (payload.size() == 1) {
        const std::uint8_t v = payload[0];
        if (name == "BFFR") { out.mmc1_shift = v; out.board_chunks |= kBoardBffr; }
        else if (name == "BFRS") { out.mmc1_shift_count = v; out.board_chunks |= kBoardBfrs; }
        else if (name == "PREG") { out.mmc2_prg = v; out.board_chunks |= kBoardPreg; }
        else if (name == "MIRR") { out.mmc2_mirroring = v; out.board_chunks |= kBoardMirr; }
        else if (name == "LAT0") { out.mmc2_latch0 = v; out.board_chunks |= kBoardLat0; }
        else if (name == "LAT1") { out.mmc2_latch1 = v; out.board_chunks |= kBoardLat1; }
        else if (name == "CMD") { out.mmc3_cmd = v; out.board_chunks |= kBoardCmd; }
        else if (name == "A000") { out.mmc3_a000 = v; out.board_chunks |= kBoardA000; }
        else if (name == "A001") { out.mmc3_a001 = v; }
        else if (name == "IRQR") { out.mmc3_irq_reload = v; out.board_chunks |= kBoardIrq; }
        else if (name == "IRQC") { out.mmc3_irq_count = v; }
        else if (name == "IRQL") { out.mmc3_irq_latch = v; }
        else if (name == "IRQA") { out.mmc3_irq_enabled = v; }
    }
}

inline void walk_subchunks(std::span<const std::uint8_t> payload,
                           void (*apply)(StateSnapshot&, std::string_view,
                                         std::span<const std::uint8_t>),
                           StateSnapshot& out) {
    std::size_t off = 0;
    while (off + 8 <= payload.size()) {
        const auto raw_name = std::string_view(
            reinterpret_cast<const char*>(payload.data() + off), 4);
        const auto name = strip_trailing_nuls(raw_name);
        const auto size = read_le_u32(payload.subspan(off + 4, 4));
        off += 8;
        if (off + size > payload.size()) {
            throw std::runtime_error("FCS: sub-chunk \"" + std::string(name) +
                                     "\" extends past parent payload");
        }
        apply(out, name, payload.subspan(off, size));
        off += size;
    }
    if (off != payload.size()) {
        throw std::runtime_error("FCS: trailing bytes in sub-chunk stream");
    }
}

// Call after walk_subchunks has checked the sub-chunk sizes. Without these, the env
// would start from zeroed memory with no error and hang or end every episode at once.
inline void require_subchunks(std::span<const std::uint8_t> payload, std::string_view section,
                              std::initializer_list<std::string_view> names) {
    for (const auto wanted : names) {
        bool found = false;
        for (std::size_t off = 0; !found && off + 8 <= payload.size();) {
            const auto raw_name = std::string_view(
                reinterpret_cast<const char*>(payload.data() + off), 4);
            found = strip_trailing_nuls(raw_name) == wanted;
            off += 8 + read_le_u32(payload.subspan(off + 4, 4));
        }
        if (!found) {
            throw std::runtime_error("FCS: " + std::string(section) + " section has no " +
                                     std::string(wanted));
        }
    }
}

}  // namespace detail

[[nodiscard]] inline StateSnapshot parse(std::span<const std::uint8_t> data) {
    if (data.size() < 16) {
        throw std::runtime_error("FCS: header too short");
    }
    if (data[0] != 'F' || data[1] != 'C' || data[2] != 'S' || data[3] != 0xFF) {
        throw std::runtime_error("FCS: bad magic (expected 'FCS\\xff')");
    }
    // Bytes 4..16 are legacy size, version, reserved; we don't enforce them.

    StateSnapshot out;
    bool has_cpu = false;
    bool has_ppu = false;
    std::size_t off = 16;
    while (off + 5 <= data.size()) {
        const auto tag = data[off];
        const auto size = detail::read_le_u32(data.subspan(off + 1, 4));
        off += 5;
        if (off + size > data.size()) {
            throw std::runtime_error("FCS: top-level chunk extends past file");
        }
        const auto payload = data.subspan(off, size);
        switch (tag) {
            case 1:  // CPU + RAM
                detail::walk_subchunks(payload, detail::apply_cpu_subchunk, out);
                detail::require_subchunks(payload, "CPU", {"PC", "RAM"});
                has_cpu = true;
                break;
            case 2:  // X6502 internals: IRQ line, cycle count
                detail::walk_subchunks(payload, detail::apply_cpu_internal_subchunk, out);
                break;
            case 3:  // PPU + VRAM
                detail::walk_subchunks(payload, detail::apply_ppu_subchunk, out);
                detail::require_subchunks(payload, "PPU", {"NTAR", "PRAM", "SPRA", "PPUR"});
                has_ppu = true;
                break;
            case 16:  // Mapper / cartridge state (WRAM lives here)
                detail::walk_subchunks(payload, detail::apply_cart_subchunk, out);
                break;
            default:
                // Tags 4 (input), 5 (APU/sound) are intentionally skipped.
                break;
        }
        off += size;
    }
    if (off != data.size()) {
        throw std::runtime_error("FCS: trailing bytes after last top-level chunk");
    }
    if (!has_cpu || !has_ppu) {
        throw std::runtime_error("FCS: state has no CPU or no PPU section");
    }
    return out;
}

// The 16 board register bytes NeSLE keeps for a paged board (layout in cuda/mapper.cuh)
// and the board's IRQ line, from the chunks FCEUmm saved. `bank_kind` is the cartridge's
// MapperLayout::bank_kind. Returns false when the state lacks the board's registers
// (it was saved for another board, or by an emulator that does not write them).
[[nodiscard]] inline bool board_registers(const StateSnapshot& s, std::uint8_t bank_kind,
                                          std::array<std::uint8_t, 16>& regs,
                                          std::uint8_t& irq_line) {
    regs.fill(0);
    irq_line = 0;
    switch (bank_kind) {
        case kBankKindCnrom:
            if (!s.has_cart_latch) return false;
            regs[0] = s.cart_latch;  // datalatch.c "LATC": the byte the board latched
            return true;
        case kBankKindMmc1:
            if ((s.board_chunks & kBoardDreg) == 0) return false;
            for (int i = 0; i < 4; ++i) regs[i] = s.board_regs[i];  // control, CHR 0, CHR 1, PRG
            regs[4] = s.mmc1_shift;
            regs[5] = s.mmc1_shift_count;
            return true;
        case kBankKindMmc2:
            if ((s.board_chunks & kBoardCreg) == 0) return false;
            for (int i = 0; i < 4; ++i) regs[i] = s.board_regs[i];  // $FD/0, $FE/0, $FD/1, $FE/1
            regs[4] = s.mmc2_prg;
            regs[5] = s.mmc2_mirroring;
            regs[6] = static_cast<std::uint8_t>(s.mmc2_latch0 & 1u);
            regs[7] = static_cast<std::uint8_t>(s.mmc2_latch1 & 1u);
            return true;
        case kBankKindMmc3:
            if ((s.board_chunks & kBoardRegs) == 0 || (s.board_chunks & kBoardCmd) == 0) return false;
            for (int i = 0; i < 8; ++i) regs[i] = s.board_regs[i];  // R0-R7
            regs[8] = s.mmc3_cmd;
            regs[9] = s.mmc3_a000;
            regs[10] = s.mmc3_a001;
            regs[11] = s.mmc3_irq_latch;
            regs[12] = s.mmc3_irq_count;
            regs[13] = s.mmc3_irq_enabled;
            regs[14] = s.mmc3_irq_reload;
            irq_line = static_cast<std::uint8_t>(s.cpu_irq_low & 1u);  // FCEU_IQEXT
            return true;
        default:
            return false;
    }
}

[[nodiscard]] inline StateSnapshot parse(const std::string& bytes) {
    return parse(std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()));
}

}  // namespace nesle::fcs
