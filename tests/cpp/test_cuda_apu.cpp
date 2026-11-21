// DMC sample channel: $4015 status and the CPU cycles its fetches steal (batch_apu.cuh).
#include <cassert>
#include <cstdint>
#include <vector>

#include "nesle/cuda/batch_bus.cuh"

namespace {

using namespace nesle::cuda;

constexpr std::uint64_t kByte = 8 * 54;  // rate 15: 54 CPU cycles per bit

struct Machine {
    std::uint64_t event = 0;
    DmcState dmc{};
    BatchBuffers buffers{};
    std::uint64_t now = 1000;  // CPU cycles
    std::uint64_t stolen = 0;

    Machine() {
        buffers.apu.dmc_event = &event;
        buffers.apu.dmc = &dmc;
        reset_dmc_env(buffers, 0, now);
    }

    void write(std::uint16_t address, std::uint8_t value) {
        batch_cpu_write<kBankModeNrom>(buffers, 0, address, value);
    }

    std::uint8_t status() { return batch_cpu_read<kBankModeNrom>(buffers, 0, 0x4015); }

    // Runs `cycles` worth of 4-cycle instructions, servicing the DMC after each
    // one the way the console step does.
    void run(std::uint64_t cycles) {
        const auto end = now + cycles;
        while (now < end) {
            now += 4;
            if (now >= event) {
                const auto s = service_dmc_env(buffers, 0, now, true);
                now += s;
                stolen += s;
            }
        }
    }
};

}  // namespace

int main() {
    {
        // No APU buffers: $4015 reads 0 and the writes are ignored, as before.
        BatchBuffers buffers{};
        batch_cpu_write<kBankModeNrom>(buffers, 0, 0x4015, 0x1F);
        assert(batch_cpu_read<kBankModeNrom>(buffers, 0, 0x4015) == 0x00);
    }

    {
        // Power-on is silent, and an idle DMC never steals.
        Machine m;
        assert(m.status() == 0x00);
        m.run(100 * kByte);
        assert(m.stolen == 0);
        assert(m.event == kDmcIdle);
    }

    {
        // A 17-byte sample at rate 15: $4015 bit 4 stays set until the last
        // byte is fetched, and each of the 17 fetches takes 4 cycles.
        Machine m;
        m.run(3 * kByte + 100);
        m.write(0x4010, 0x0F);
        m.write(0x4013, 0x01);
        m.write(0x4015, 0x1F);
        assert(m.status() == 0x10);
        m.run(4);  // the first byte is fetched after the starting instruction
        assert(m.stolen == 4);
        m.run(15 * kByte - 2 * 4);
        assert(m.status() == 0x10);
        m.run(2 * kByte);
        assert(m.status() == 0x00);
        assert(m.stolen == 17 * 4);
        m.run(10 * kByte);
        assert(m.stolen == 17 * 4);
        // Writing bit 4 again restarts a finished sample.
        m.write(0x4015, 0x10);
        assert(m.status() == 0x10);
    }

    {
        // Bit 4 set again while playing does not restart; bit 4 clear stops at
        // once (a byte already buffered still drains without a fetch).
        Machine m;
        m.write(0x4010, 0x0F);
        m.write(0x4013, 0x01);
        m.write(0x4015, 0x10);
        m.run(4 * kByte);
        const auto fetched = m.stolen / 4;
        assert(fetched >= 4 && fetched <= 5);
        m.write(0x4015, 0x10);
        m.run(4 * kByte);
        assert(m.stolen / 4 >= 8 && m.stolen / 4 <= 9);
        m.write(0x4015, 0x00);
        assert(m.status() == 0x00);
        const auto before = m.stolen;
        m.run(4 * kByte);
        assert(m.stolen == before);
    }

    {
        // Loop flag: the sample restarts, bit 4 never clears, one fetch per byte.
        Machine m;
        m.write(0x4010, 0x4F);
        m.write(0x4013, 0x00);  // 1 byte, looped
        m.write(0x4015, 0x10);
        m.run(100 * kByte);
        assert(m.status() == 0x10);
        assert(m.stolen >= 100 * 4 && m.stolen <= 101 * 4);
    }

    {
        // A rate change applies from the next bit clock: at rate 0 (428 cycles
        // a bit) a byte lasts 3424 cycles.
        Machine m;
        m.write(0x4010, 0x4F);
        m.write(0x4015, 0x10);
        m.run(10 * kByte);
        m.write(0x4010, 0x40);
        m.run(8 * 428);  // drains the byte already counting at the old rate
        const auto before = m.stolen;
        m.run(20 * 8 * 428);
        assert(m.stolen - before >= 19 * 4 && m.stolen - before <= 21 * 4);
    }

    {
        // Fast-forwarded idle loop (steal = false): bytes are still consumed on
        // time, but the CPU loses no cycles.
        Machine m;
        m.write(0x4010, 0x0F);
        m.write(0x4013, 0x01);
        m.write(0x4015, 0x10);
        const auto start = m.now;
        assert(service_dmc_env(m.buffers, 0, start, false) == 0);
        assert(service_dmc_env(m.buffers, 0, start + 10 * kByte, false) == 0);
        assert(m.status() == 0x10);
        assert(service_dmc_env(m.buffers, 0, start + 17 * kByte, false) == 0);
        assert(m.status() == 0x00);
    }

    return 0;
}
