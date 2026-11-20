#include "nesle/rom.hpp"

#include <algorithm>

namespace nesle {
namespace {

constexpr std::size_t kHeaderSize = 16;
constexpr std::size_t kTrainerSize = 512;
constexpr std::size_t kPrgBankSize = 16 * 1024;
constexpr std::size_t kChrBankSize = 8 * 1024;

std::size_t next_power_of_two(std::size_t value) noexcept {
    std::size_t result = 1;
    while (result < value) {
        result <<= 1u;
    }
    return result;
}

bool is_power_of_two(std::size_t value) noexcept {
    return value != 0 && (value & (value - 1)) == 0;
}

std::uint8_t page_mask(std::size_t prg_size, std::uint32_t page_bytes) noexcept {
    if (prg_size < page_bytes) {
        return 0;
    }
    // PRG is padded to a power of two before it reaches the device, so the page
    // count is always a power of two and the mask is exact.
    const std::size_t pages = next_power_of_two(prg_size) / page_bytes;
    if (pages < 2) {
        return 0;
    }
    return static_cast<std::uint8_t>(pages - 1);
}

// The device reaches a page with `bank & mask`, which only selects the right
// page when the page count is a power of two; on any other size the mask wraps
// into the zero padding. The host instead wraps with `%` over the real image, so
// the two disagree on such images. Rejecting them keeps one behaviour; there is
// no commercial UxROM/BNROM board with a non-power-of-two page count.
bool page_count_is_power_of_two(std::size_t prg_size, std::uint32_t page_bytes) noexcept {
    if (page_bytes == 0 || prg_size % page_bytes != 0) {
        return false;
    }
    return is_power_of_two(prg_size / page_bytes);
}

// Boards with mirroring wired to solder pads. The iNES four-screen bit is a
// lie on these carts: there is no third and fourth CIRAM page to select, so the
// bit falls back to the horizontal/vertical choice.
bool two_bit_mirroring(std::uint16_t mapper) noexcept {
    return mapper == 2 || mapper == 94;
}

}  // namespace

bool RomMetadata::is_nrom() const noexcept {
    return mapper == 0 && (prg_rom_banks == 1 || prg_rom_banks == 2);
}

bool RomMetadata::is_uxrom() const noexcept {
    return mapper == 2 || mapper == 94;
}


std::string to_string(std::uint16_t mapper) {
    switch (mapper) {
        case 0:
            return "NROM";
        case 2:
            return "UxROM";
        case 34:
            return "BNROM";
        case 94:
            return "UN1ROM";
        case 1:
            return "MMC1";
        case 3:
            return "CNROM";
        case 4:
            return "MMC3";
        case 9:
            return "MMC2";
        default:
            return "mapper " + std::to_string(mapper);
    }
}

std::string to_string(NametableArrangement arrangement) {
    switch (arrangement) {
        case NametableArrangement::Vertical:
            return "vertical";
        case NametableArrangement::Horizontal:
            return "horizontal";
        case NametableArrangement::FourScreen:
            return "four_screen";
    }
    return "unknown";
}

// Window geometry for every board the batched bus knows how to drive. The
// device side never looks at the iNES fields again; it only reads what this
// function computed, which is why adding a board is a host-side change.
MapperLayout describe_mapper(const RomMetadata& metadata) noexcept {
    MapperLayout layout;

    if (metadata.mapper == 0) {
        // NROM: 16 or 32 KB fixed at $8000, no registers. A trainer is 512 bytes
        // in front of the PRG data and parse_ines skips it, so it does not make
        // the image unsupported - main has always accepted those.
        layout.supported = metadata.is_nrom();
        return layout;
    }

    // Bus conflicts are destructive: only bits the mask ROM already has set
    // survive a write. Every commercial UxROM title loads its bank from a
    // table of bytes that already hold the wanted value, so the permissive
    // no-conflict behavior boots them all; conflicts are enabled only when the
    // ROM declares them (NES 2.0 submapper 2), which is what FCEUX does too.
    const bool declared_conflicts = metadata.is_nes2 && metadata.submapper == 2;

    switch (metadata.mapper) {
        case 2:  // UxROM / UNROM / UOROM
        case 94: // UN1ROM: non-consecutive 16 KB pages, value >> 2
            layout.bank_kind = kBankKindUxrom16k;
            layout.window_bytes = 0x4000;
            layout.fixed_bytes = 0x4000;
            layout.bank_shift = metadata.mapper == 94 ? 2 : 0;
            layout.bus_conflicts = declared_conflicts;
            layout.bank_mask = page_mask(metadata.prg_rom_size, layout.window_bytes);
            layout.supported = metadata.prg_rom_size >= 0x8000 &&
                               page_count_is_power_of_two(metadata.prg_rom_size, 0x4000);
            break;

        case 34:
            if (metadata.chr_rom_size != 0) {  // NINA-001 (FCEUmm 34.c): paged, registers at $7FFD-$7FFF
                layout.bank_kind = kBankKindNina001;
                layout.prg_page_mask = static_cast<std::uint16_t>(metadata.prg_rom_size / 0x2000 - 1);
                layout.chr_page_mask = static_cast<std::uint16_t>(metadata.chr_rom_size / 0x400 - 1);
                layout.supported = metadata.prg_rom_size >= 0x8000 && metadata.prg_rom_size <= 512u * 1024u &&
                                   page_count_is_power_of_two(metadata.prg_rom_size, 0x2000) &&
                                   metadata.chr_rom_size <= 256u * 1024u &&
                                   page_count_is_power_of_two(metadata.chr_rom_size, 0x2000);
                break;
            }
            // BNROM: one 32 KB window over the whole CPU space, CHR RAM.
            // NINA-001 is the other board behind mapper 34; it needs its own
            // window layout and is deliberately not supported here.
            layout.bank_kind = kBankKindBnrom32k;
            layout.window_bytes = 0x8000;
            layout.fixed_bytes = 0;
            layout.bus_conflicts = declared_conflicts;
            layout.bank_mask = page_mask(metadata.prg_rom_size, layout.window_bytes);
            layout.supported = metadata.chr_rom_size == 0 &&
                               metadata.prg_rom_size >= 0x8000 &&
                               page_count_is_power_of_two(metadata.prg_rom_size, 0x8000);
            break;

        case 1:   // MMC1 (SxROM)
        case 105: // NWC 1990: MMC1 with a timer IRQ
        case 3:   // CNROM
        case 4:   // MMC3 (TxROM)
        case 118: // MMC3, nametables from CHR bit 7 (TxSROM)
        case 189: // MMC3, 32 KB PRG from $4120-$7FFF
        case 245: // Waixing MMC3 with CHR RAM
        case 64:  // Tengen RAMBO-1
        case 187: // MMC3 with a $5000/$6000 PRG override and CHR A18 on the 2 KB half
        case 119: // TQROM: MMC3 with CHR ROM and 8 KB CHR RAM side by side
        case 47:  // NES-QJ: MMC3 with a 128 KB block register at $6000
        case 114: // MMC3 with scrambled registers and a 16 KB PRG override at $5000-$7FFF
        case 182: // FCEUmm's database runs both 182 ROMs here as 114 (182 itself is deprecated there)
        case 9:   // MMC2 (PxROM)
        case 10: { // MMC4 (FxROM): MMC2's latches, 16 KB PRG
            // Paged boards: no window; the bus goes through per-env page tables.
            layout.bank_kind = metadata.mapper == 1 || metadata.mapper == 105 ? kBankKindMmc1
                               : metadata.mapper == 3 ? kBankKindCnrom
                               : metadata.mapper == 9 || metadata.mapper == 10 ? kBankKindMmc2
                                                      : kBankKindMmc3;
            layout.bus_conflicts = metadata.mapper == 3 && declared_conflicts;
            const std::size_t chr_bytes =
                metadata.chr_rom_size == 0 ? std::size_t{8 * 1024} : metadata.chr_rom_size;
            const bool prg_ok = metadata.prg_rom_size >= 0x4000u &&
                                metadata.prg_rom_size <= (metadata.mapper == 245 ? 1024u : 512u) * 1024u &&
                                page_count_is_power_of_two(metadata.prg_rom_size, 0x2000);
            const bool chr_ok = chr_bytes <= (metadata.mapper == 187 || metadata.mapper == 10 ? 512u : 256u) * 1024u &&
                                page_count_is_power_of_two(chr_bytes, 0x2000);
            // CNROM and MMC2 have no CHR RAM variant that FCEUmm drives this way.
            const bool chr_kind_ok =
                metadata.chr_rom_size != 0 || layout.bank_kind == kBankKindMmc1 || layout.bank_kind == kBankKindMmc3;
            layout.prg_page_mask = static_cast<std::uint16_t>(metadata.prg_rom_size / 0x2000 - 1);
            layout.chr_page_mask = static_cast<std::uint16_t>(chr_bytes / 0x400 - 1);
            layout.supported = prg_ok && chr_ok && chr_kind_ok;
            break;
        }

        case 15: case 32: case 88: case 91: case 42: case 68: case 19: case 17: case 6: case 16: case 222: case 117: case 18: case 48: case 13: case 8: case 227: case 73: case 188: case 243: case 133: case 246: case 75: case 90: case 83: case 228:
        case 24: case 26: case 40: case 232:
        case 21: case 22: case 23: case 25: case 69:
        case 7: case 11: case 66: case 71: case 79: case 113: case 140: case 38: case 87: {
            // Latch boards: paged like CNROM. CHR RAM only where the board has it (7, 71; FCEUmm allows it on all).
            switch (metadata.mapper) {
                case 7: layout.bank_kind = kBankKindAxrom; break;
                case 11: layout.bank_kind = kBankKindColorDreams; break;
                case 66: layout.bank_kind = kBankKindGxrom; break;
                case 71: layout.bank_kind = kBankKindCamerica; break;
                case 79: layout.bank_kind = kBankKindNina0306; break;
                case 113: layout.bank_kind = kBankKindHes; break;
                case 140: layout.bank_kind = kBankKindJaleco140; break;
                case 21: layout.bank_kind = kBankKindVrc21; break;
                case 22: layout.bank_kind = kBankKindVrc22; break;
                case 23: layout.bank_kind = kBankKindVrc23; break;
                case 25: layout.bank_kind = kBankKindVrc25; break;
                case 69: layout.bank_kind = kBankKindFme7; break;
                case 24: layout.bank_kind = kBankKindVrc6a; break;
                case 26: layout.bank_kind = kBankKindVrc6b; break;
                case 40: layout.bank_kind = kBankKindSmb2j40; break;
                case 232: layout.bank_kind = kBankKindQuattro; break;
                case 15: layout.bank_kind = kBankKindM15; break;
                case 32: layout.bank_kind = kBankKindIrem32; break;
                case 88: layout.bank_kind = kBankKindNamco88; break;
                case 91: layout.bank_kind = kBankKindM91; break;
                case 42: layout.bank_kind = kBankKindM42; break;
                case 68: layout.bank_kind = kBankKindSunsoft4; break;
                case 19: layout.bank_kind = kBankKindN163; break;
                case 17: case 6: layout.bank_kind = kBankKindFfe; break;
                case 16: layout.bank_kind = kBankKindBandai16; break;
                case 222: layout.bank_kind = kBankKindM222; break;
                case 117: layout.bank_kind = kBankKindM117; break;
                case 18: layout.bank_kind = kBankKindJaleco18; break;
                case 48: layout.bank_kind = kBankKindTaito48; break;
                case 13: layout.bank_kind = kBankKindCprom; break;
                case 8: layout.bank_kind = kBankKindM8; break;
                case 227: layout.bank_kind = kBankKindM227; break;
                case 73: layout.bank_kind = kBankKindVrc3; break;
                case 188: layout.bank_kind = kBankKindM188; break;
                case 243: layout.bank_kind = kBankKindM243; break;
                case 133: layout.bank_kind = kBankKindSa72008; break;
                case 246: layout.bank_kind = kBankKindM246; break;
                case 75: layout.bank_kind = kBankKindVrc1; break;
                case 90: layout.bank_kind = kBankKindJy90; break;
                case 83: layout.bank_kind = kBankKindM83; break;
                case 228: layout.bank_kind = kBankKindM228; break;
                case 38: layout.bank_kind = kBankKindBitCorp38; break;
                default: layout.bank_kind = kBankKindJaleco87; break;
            }
            const std::size_t chr_bytes =
                metadata.chr_rom_size == 0 ? std::size_t{8 * 1024} : metadata.chr_rom_size;
            layout.prg_page_mask = static_cast<std::uint16_t>(metadata.prg_rom_size / 0x2000 - 1);
            layout.chr_page_mask = static_cast<std::uint16_t>(chr_bytes / 0x400 - 1);
            layout.supported = metadata.prg_rom_size >= 0x4000 && metadata.prg_rom_size <= 512u * 1024u &&
                               page_count_is_power_of_two(metadata.prg_rom_size, 0x2000) &&
                               chr_bytes <= 1024u * 1024u && page_count_is_power_of_two(chr_bytes, 0x2000) &&
                               // A trainer is copied to $7000 (prg_ram + $1000) by the host at power; FFE only.
                               (!metadata.has_trainer || layout.bank_kind == kBankKindFfe);
            break;
        }

        default:
            break;
    }
    return layout;
}

std::string unsupported_mapper_reason(const RomMetadata& metadata) {
    if (metadata.mapper == 0) {
        if (!metadata.is_nrom()) {
            return "expected one or two 16 KB PRG ROM banks for NROM";
        }
        return "";
    }
    const auto layout = describe_mapper(metadata);
    if (layout.supported) {
        return "";
    }
    std::string reason = "unsupported mapper: " + to_string(metadata.mapper) + " (iNES " +
                         std::to_string(metadata.mapper) +
                         "); NeSLE emulates NROM, MMC1, UxROM, CNROM, MMC3, MMC2, BNROM and UN1ROM "
                         "(0, 1, 2, 3, 4, 9, 34, 94)";

    if (metadata.prg_rom_size < 0x8000 || metadata.prg_rom_size % 0x2000 != 0) {
        reason += "; this image also has an unsupported PRG size (" +
                  std::to_string(metadata.prg_rom_size / 1024) + " KB)";
    }
    return reason;
}

bool is_supported_mario_target(const RomMetadata& metadata) noexcept {
    return metadata.mapper == 0 &&
           metadata.submapper == 0 &&
           (metadata.prg_rom_banks == 1 || metadata.prg_rom_banks == 2) &&
           metadata.chr_rom_banks == 1 &&
           !metadata.has_trainer;
}

std::uint32_t reset_vector_base(const MapperLayout& layout,
                                std::size_t real_prg_size) noexcept {
    if (layout.bank_kind == kBankKindBnrom32k) {
        // Bank 0's 32 KB window sits at $8000 at power-on, so the vector is the
        // last four bytes of that window - the first 32 KB of the image - and
        // not of the image.
        return layout.window_bytes - 4u;
    }
    // NROM keeps the whole image at $8000 and UxROM keeps its fixed last 16 KB
    // at $C000, so on both $FFFC is the last four bytes of the image.
    return static_cast<std::uint32_t>(real_prg_size - 4u);
}

std::string unsupported_mario_target_reason(const RomMetadata& metadata) {
    if (metadata.mapper != 0) {
        return "expected mapper 0/NROM for Super Mario Bros.";
    }
    if (metadata.submapper != 0) {
        return "expected submapper 0 for Super Mario Bros.";
    }
    if (metadata.prg_rom_banks != 1 && metadata.prg_rom_banks != 2) {
        return "expected one or two 16 KB PRG ROM banks for NROM";
    }
    if (metadata.chr_rom_banks != 1) {
        return "expected one 8 KB CHR ROM bank for Super Mario Bros.";
    }
    if (metadata.has_trainer) {
        return "ROM trainers are not supported";
    }
    return "";
}

void validate_supported_mario_target(const RomMetadata& metadata) {
    const auto reason = unsupported_mario_target_reason(metadata);
    if (!reason.empty()) {
        throw std::invalid_argument(reason);
    }
}

RomImage parse_ines(std::span<const std::uint8_t> bytes) {
    if (bytes.size() < kHeaderSize) {
        throw std::invalid_argument("iNES data is shorter than the 16-byte header");
    }

    if (bytes[0] != 'N' || bytes[1] != 'E' || bytes[2] != 'S' || bytes[3] != 0x1A) {
        throw std::invalid_argument("iNES header magic must be NES<EOF>");
    }

    const auto prg_banks = bytes[4];
    const auto chr_banks = bytes[5];
    const auto flags6 = bytes[6];
    // Old dumping tools wrote text ("DiskDude!", "Ni03") into bytes 7-15. As FCEUX does, an iNES 1.0 header whose
    // bytes 12-15 are not zero has a garbage flags 7, so its mapper high nibble is ignored.
    const bool nes2_bits = (bytes[7] & 0x0C) == 0x08;
    const bool dirty = !nes2_bits && (bytes[12] | bytes[13] | bytes[14] | bytes[15]) != 0;
    const auto flags7 = dirty ? std::uint8_t{0} : bytes[7];
    const bool is_nes2 = nes2_bits;
    if (is_nes2 && bytes[9] != 0) {
        throw std::invalid_argument("NES 2.0 extended PRG/CHR ROM sizes are not supported yet");
    }

    RomMetadata metadata;
    metadata.prg_rom_banks = prg_banks;
    metadata.chr_rom_banks = chr_banks;
    metadata.mapper = static_cast<std::uint16_t>((flags6 >> 4) | (flags7 & 0xF0));
    metadata.is_nes2 = is_nes2;
    if (metadata.is_nes2) {
        metadata.mapper = static_cast<std::uint16_t>(
            metadata.mapper | (static_cast<std::uint16_t>(bytes[8] & 0x0F) << 8));
        metadata.submapper = static_cast<std::uint8_t>(bytes[8] >> 4);
    }
    metadata.has_trainer = (flags6 & 0x04) != 0;
    metadata.has_battery = (flags6 & 0x02) != 0;
    const bool vertical = (flags6 & 0x01) != 0;
    const bool four_screen = (flags6 & 0x08) != 0;
    if (two_bit_mirroring(metadata.mapper)) {
        // UxROM boards (UNROM/UN1ROM/UOROM) wire mirroring to solder pads:
        // only H and V exist, so a four-screen bit cannot be honored. Some
        // translation patches set it anyway (for instance the uBAH Russian
        // Contra), so fall back to the H/V bit rather than folding the screen
        // to vertical.
        metadata.nametable_arrangement =
            vertical ? NametableArrangement::Vertical : NametableArrangement::Horizontal;
    } else {
        metadata.nametable_arrangement = four_screen
                                             ? NametableArrangement::FourScreen
                                             : (vertical ? NametableArrangement::Vertical
                                                         : NametableArrangement::Horizontal);
    }
    metadata.prg_rom_size = static_cast<std::size_t>(prg_banks) * kPrgBankSize;
    metadata.chr_rom_size = static_cast<std::size_t>(chr_banks) * kChrBankSize;

    std::size_t offset = kHeaderSize;
    if (metadata.has_trainer) {
        offset += kTrainerSize;
    }

    const std::size_t required = offset + metadata.prg_rom_size + metadata.chr_rom_size;
    // A file shorter than its header declares (a short last bank) loads as in FCEUX: the missing bytes read as 0.
    if (bytes.size() < offset + 0x10) {
        throw std::invalid_argument("iNES data is truncated for declared PRG/CHR sizes");
    }
    std::vector<std::uint8_t> padded;
    if (bytes.size() < required) {
        padded.assign(bytes.begin(), bytes.end());
        padded.resize(required, 0);
        bytes = std::span<const std::uint8_t>(padded);
    }

    RomImage image;
    image.metadata = metadata;
    if (metadata.has_trainer) {
        image.trainer.assign(bytes.begin() + static_cast<std::ptrdiff_t>(kHeaderSize),
                             bytes.begin() + static_cast<std::ptrdiff_t>(kHeaderSize + kTrainerSize));
    }
    image.prg_rom.assign(bytes.begin() + static_cast<std::ptrdiff_t>(offset),
                         bytes.begin() + static_cast<std::ptrdiff_t>(offset + metadata.prg_rom_size));
    offset += metadata.prg_rom_size;
    image.chr_rom.assign(bytes.begin() + static_cast<std::ptrdiff_t>(offset),
                         bytes.begin() + static_cast<std::ptrdiff_t>(offset + metadata.chr_rom_size));
    // FCEUmm's ines-correct.h overrides the header by the CRC32 of PRG + CHR. Only entries that no mapper-number rule
    // covers are here.
    {
        std::uint32_t crc = 0xFFFFFFFFu;
        for (const auto* rom : {&image.prg_rom, &image.chr_rom}) {
            for (const std::uint8_t b : *rom) {
                crc ^= b;
                for (int k = 0; k < 8; ++k) crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
            }
        }
        crc = ~crc;
        if (crc == 0xBDE3AE9Bu) {  // Doraemon (World 3 hack): mapper 66, vertical (header says 120, four-screen)
            image.metadata.mapper = 66;
            image.metadata.nametable_arrangement = NametableArrangement::Vertical;
        }
    }
    // Banked boards: as FCEUX does, an image whose PRG or CHR is not a power of two (PlayChoice dumps carry an 8 KB
    // INST-ROM after the CHR, some hacks grew the CHR) is padded with $FF up to one, so bank masks wrap cleanly.
    if (metadata.mapper != 0) {
        auto pow2 = [](std::size_t n) { std::size_t p = 1; while (p < n) p <<= 1; return p; };
        if (!image.prg_rom.empty() && pow2(image.prg_rom.size()) != image.prg_rom.size()) {
            image.prg_rom.resize(pow2(image.prg_rom.size()), 0xFF);
            image.metadata.prg_rom_size = image.prg_rom.size();
        }
        if (!image.chr_rom.empty() && pow2(image.chr_rom.size()) != image.chr_rom.size()) {
            image.chr_rom.resize(pow2(image.chr_rom.size()), 0xFF);
            image.metadata.chr_rom_size = image.chr_rom.size();
        }
    }
    return image;
}

}  // namespace nesle
