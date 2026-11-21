#include <array>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "nesle/fcs.hpp"

namespace {

using Bytes = std::vector<std::uint8_t>;

void put_u32(Bytes& out, std::uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
        out.push_back(static_cast<std::uint8_t>(value >> shift));
    }
}

void add_subchunk(Bytes& section, std::string_view name, std::size_t size) {
    std::string padded(name);
    padded.resize(4, '\0');
    section.insert(section.end(), padded.begin(), padded.end());
    put_u32(section, static_cast<std::uint32_t>(size));
    section.insert(section.end(), size, 0);
}

Bytes section(std::initializer_list<std::pair<std::string_view, std::size_t>> subchunks) {
    Bytes out;
    for (const auto& [name, size] : subchunks) {
        add_subchunk(out, name, size);
    }
    return out;
}

Bytes file(std::initializer_list<std::pair<std::uint8_t, Bytes>> chunks) {
    Bytes out = {'F', 'C', 'S', 0xFF};
    out.resize(16, 0);
    for (const auto& [tag, payload] : chunks) {
        out.push_back(tag);
        put_u32(out, static_cast<std::uint32_t>(payload.size()));
        out.insert(out.end(), payload.begin(), payload.end());
    }
    return out;
}

bool rejected(const Bytes& bytes) {
    try {
        (void)nesle::fcs::parse(std::span<const std::uint8_t>(bytes));
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

const Bytes kCpu = section({{"PC", 2}, {"RAM", 2048}});
const Bytes kPpu = section({{"NTAR", 2048}, {"PRAM", 32}, {"SPRA", 256}, {"PPUR", 4}});

void parses_a_real_fceux_save() {
    // Written by FCEUX 2.6.5 with no compression; the chunks start after its 16-byte header.
    std::ifstream in("docs/data/smb_level1_1_fceux.fc1", std::ios::binary);
    assert(in);
    Bytes bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    assert(bytes.size() > 16 && bytes[3] == 'X');
    bytes[3] = 0xFF;

    const auto snapshot = nesle::fcs::parse(std::span<const std::uint8_t>(bytes));
    assert(snapshot.pc == 0x8057);
    assert(snapshot.cpu_ram[0x0770] == 1);  // SMB's OperMode: in the game, past the title
    assert(snapshot.cpu_ram[0x075F] == 0);  // World 1
    assert(snapshot.palette_ram[0] != 0 || snapshot.palette_ram[1] != 0);
}

void accepts_a_state_with_every_core_section() {
    assert(!rejected(file({{1, kCpu}, {3, kPpu}})));
}

void rejects_states_that_do_not_describe_a_machine() {
    assert(rejected(file({})));
    assert(rejected(file({{1, kCpu}})));
    assert(rejected(file({{3, kPpu}})));
    assert(rejected(file({{2, Bytes(8, 0)}})));
    assert(rejected(file({{1, section({{"PC", 2}})}, {3, kPpu}})));
    assert(rejected(file({{1, section({{"RAM", 2048}})}, {3, kPpu}})));
    for (const auto missing : {"NTAR", "PRAM", "SPRA", "PPUR"}) {
        Bytes ppu;
        for (const auto& [name, size] : std::initializer_list<std::pair<std::string_view, std::size_t>>{
                 {"NTAR", 2048}, {"PRAM", 32}, {"SPRA", 256}, {"PPUR", 4}}) {
            if (name != missing) {
                add_subchunk(ppu, name, size);
            }
        }
        assert(rejected(file({{1, kCpu}, {3, ppu}})));
    }
}

Bytes one(std::string_view name, std::initializer_list<std::uint8_t> payload) {
    Bytes out;
    std::string padded(name);
    padded.resize(4, '\0');
    out.insert(out.end(), padded.begin(), padded.end());
    put_u32(out, static_cast<std::uint32_t>(payload.size()));
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}

Bytes cat(std::initializer_list<Bytes> parts) {
    Bytes out;
    for (const auto& part : parts) out.insert(out.end(), part.begin(), part.end());
    return out;
}

// The paged boards' registers come out of FCEUmm's board chunks (as stable_retro's
// fceumm writes them) and map onto NeSLE's register layout per board.
void reads_paged_board_registers() {
    using nesle::fcs::board_registers;
    std::array<std::uint8_t, 16> regs{};
    std::uint8_t irq = 0;

    // MMC3: REGS (R0-R7), CMD, A000, A001, IRQ*, and the cartridge IRQ line in IQLB.
    const auto mmc3 = nesle::fcs::parse(std::span<const std::uint8_t>(file(
        {{1, kCpu},
         {2, cat({one("IQLB", {0x01, 0x02, 0, 0}), one("ICou", {0xA0, 0xFF, 0xFF, 0xFF})})},
         {3, cat({kPpu, one("KOOK", {1})})},
         {16, cat({one("REGS", {0, 0x3A, 0x4A, 0x5D, 0x6E, 0x61, 0x0A, 0x04}), one("CMD", {0x46}),
                   one("A000", {1}), one("A001", {0x80}), one("IRQR", {1}), one("IRQC", {7}),
                   one("IRQL", {9}), one("IRQA", {1})})}})));
    assert(mmc3.cpu_count == -96 && mmc3.ppu_kook == 1);
    assert(board_registers(mmc3, nesle::kBankKindMmc3, regs, irq));
    assert(regs[1] == 0x3A && regs[7] == 0x04 && regs[8] == 0x46 && regs[9] == 1 && regs[10] == 0x80);
    assert(regs[11] == 9 && regs[12] == 7 && regs[13] == 1 && regs[14] == 1 && irq == 1);
    assert(!board_registers(mmc3, nesle::kBankKindMmc1, regs, irq));  // another board's chunks

    // MMC1: DREG (control, CHR 0, CHR 1, PRG) and the half-filled shift register.
    const auto mmc1 = nesle::fcs::parse(std::span<const std::uint8_t>(file(
        {{1, kCpu}, {3, kPpu},
         {16, cat({one("DREG", {0x0F, 0, 0, 5}), one("BFFR", {0x03}), one("BFRS", {2})})}})));
    assert(board_registers(mmc1, nesle::kBankKindMmc1, regs, irq));
    assert(regs[0] == 0x0F && regs[3] == 5 && regs[4] == 3 && regs[5] == 2 && irq == 0);

    // MMC2: CREG, PREG, MIRR, LAT0/LAT1.
    const auto mmc2 = nesle::fcs::parse(std::span<const std::uint8_t>(file(
        {{1, kCpu}, {3, kPpu},
         {16, cat({one("CREG", {5, 0x1D, 0, 1}), one("PREG", {2}), one("MIRR", {1}), one("LAT0", {0}),
                   one("LAT1", {1})})}})));
    assert(board_registers(mmc2, nesle::kBankKindMmc2, regs, irq));
    assert(regs[1] == 0x1D && regs[4] == 2 && regs[5] == 1 && regs[6] == 0 && regs[7] == 1);

    // CNROM: LATC.
    const auto cnrom = nesle::fcs::parse(std::span<const std::uint8_t>(
        file({{1, kCpu}, {3, kPpu}, {16, cat({one("LATC", {0x31}), one("BUSC", {1})})}})));
    assert(board_registers(cnrom, nesle::kBankKindCnrom, regs, irq) && regs[0] == 0x31);
    assert(!board_registers(nesle::fcs::parse(std::span<const std::uint8_t>(file({{1, kCpu}, {3, kPpu}}))),
                            nesle::kBankKindCnrom, regs, irq));
}

}  // namespace

int main() {
    reads_paged_board_registers();
    parses_a_real_fceux_save();
    accepts_a_state_with_every_core_section();
    rejects_states_that_do_not_describe_a_machine();
}
