#include <array>
#include <cassert>
#include <cstdint>
#include <vector>

#include "nesle/cuda/batch_render.cuh"

namespace {

constexpr int kW = nesle::cuda::kFrameWidth;
constexpr int kH = nesle::cuda::kFrameHeight;
using Frame = std::array<std::uint8_t, kW * kH * nesle::cuda::kRgbChannels>;

struct Env {
    std::vector<std::uint8_t> chr_rom = std::vector<std::uint8_t>(8 * 1024, 0);
    std::vector<std::uint8_t> nametable = std::vector<std::uint8_t>(nesle::cuda::kNametableRamBytes, 0);
    std::vector<std::uint8_t> palette = std::vector<std::uint8_t>(nesle::cuda::kPaletteRamBytes, 0);
    std::vector<std::uint8_t> oam = std::vector<std::uint8_t>(nesle::cuda::kOamBytes, 0xFF);
    std::array<std::uint8_t, 1> ctrl{0}, mask{0}, x{0}, bank{0};
    std::array<std::uint16_t, 1> v{0}, t{0}, count{0};
    std::vector<std::uint32_t> frame_log = std::vector<std::uint32_t>(nesle::cuda::kFrameLogWords, 0);
    Frame frame{};

    // No frame log: the live state, drawn as a frame without writes.
    nesle::cuda::BatchBuffers live() {
        nesle::cuda::BatchBuffers b{};
        b.ppu.ctrl = ctrl.data();
        b.ppu.mask = mask.data();
        b.ppu.nametable_ram = nametable.data();
        b.ppu.palette_ram = palette.data();
        b.ppu.oam = oam.data();
        b.cart.chr_rom = chr_rom.data();
        b.cart.chr_rom_size = static_cast<std::uint32_t>(chr_rom.size());
        b.cart.nametable_arrangement = nesle::cuda::kNametableVertical;
        b.frames_rgb = frame.data();
        return b;
    }

    // With a frame log: bank 0 presented, seeded from the live state.
    nesle::cuda::BatchBuffers logged() {
        auto b = live();
        b.ppu.x = x.data();
        b.ppu.v = v.data();
        b.ppu.t = t.data();
        b.ppu.frame_log = frame_log.data();
        b.ppu.log_count = count.data();
        b.ppu.log_bank = bank.data();
        nesle::cuda::frame_log_reset(b, 0);
        return b;
    }

    // The RGB stable_retro shows for palette value `value` on a line without emphasis.
    static std::array<std::uint8_t, 3> rgb(std::uint8_t value) {
        std::array<std::uint8_t, 3> out{};
        nesle::cuda::fceumm_rgb(value, 0, 0, out.data());
        return out;
    }

    bool pixel_is(int px, int py, std::uint8_t value) const {
        const auto want = rgb(value);
        const auto* p = frame.data() + (py * kW + px) * 3;
        return p[0] == want[0] && p[1] == want[1] && p[2] == want[2];
    }
};

}  // namespace

int main() {
    {
        // FCEUmm's palette, as RGB565 widened back to 8 bits.
        assert((Env::rgb(0x00) == std::array<std::uint8_t, 3>{112, 116, 112}));
        assert((Env::rgb(0x21) == std::array<std::uint8_t, 3>{56, 188, 248}));
    }

    {
        // Background tile 1 at the top left, left column shown.
        Env e;
        e.chr_rom[0x0010] = 0x80;  // tile 1, row 0: leftmost pixel colour 1
        e.nametable[0] = 0x01;
        e.palette[1] = 0x21;
        e.mask[0] = 0x0A;
        auto b = e.live();
        nesle::cuda::render_batch_rgb_frame_env(b, 0);
        assert(e.pixel_is(0, 0, 0x21));
        assert(e.pixel_is(1, 0, 0x00));
        assert(e.pixel_is(0, 1, 0x00));
        assert(e.pixel_is(8, 0, 0x00));

        // Left column hidden: the backdrop instead.
        e.mask[0] = 0x08;
        nesle::cuda::render_batch_rgb_frame_env(b, 0);
        assert(e.pixel_is(0, 0, 0x00));
    }

    {
        // A sprite at OAM y 9 is drawn from line 10, over a disabled background.
        Env e;
        e.chr_rom[0x0020] = 0x80;  // tile 2, row 0: leftmost pixel colour 1
        e.palette[0x11] = 0x16;
        e.oam[0] = 9;
        e.oam[1] = 2;
        e.oam[2] = 0;
        e.oam[3] = 12;
        e.mask[0] = 0x14;
        auto b = e.live();
        nesle::cuda::render_batch_rgb_frame_env(b, 0);
        assert(e.pixel_is(12, 10, 0x16));
        assert(e.pixel_is(12, 9, 0x00));
        assert(e.pixel_is(13, 10, 0x00));
    }

    {
        // A mid-frame $2005 write from the log, late on line 100 (after its fetches):
        // the lines after it start 8 pixels to the right (FCEUmm copies the coarse x
        // from t at dot 262 of each line).
        Env e;
        e.chr_rom[0x0010] = 0xFF;  // tile 1: row 0 all colour 1
        for (int row = 0; row < 8; ++row) e.chr_rom[0x0010 + row] = 0xFF;
        for (int col = 0; col < 32; col += 2) {
            for (int tile_row = 0; tile_row < 30; ++tile_row) e.nametable[tile_row * 32 + col] = 0x01;
        }
        e.palette[1] = 0x21;
        e.mask[0] = 0x0A;
        auto b = e.logged();
        std::uint32_t* h = nesle::cuda::frame_log_bank(b, 0, 0);
        // $2005 = 8 (coarse x 1) by a 4-cycle store starting at NeSLE dot 100 * 341 + 240.
        h[nesle::cuda::kLogHeaderWords] = (100u * 341u + 240u) | (4u << 17) |
                                          (static_cast<std::uint32_t>(nesle::cuda::kEvScroll) << 24);
        h[nesle::cuda::kLogHeaderWords + 1] = 8u;
        h[0] = 1;
        nesle::cuda::render_batch_rgb_frame_env(b, 0);
        assert(e.pixel_is(0, 50, 0x21) && e.pixel_is(8, 50, 0x00));
        assert(e.pixel_is(0, 100, 0x21) && e.pixel_is(8, 100, 0x00));
        assert(e.pixel_is(0, 101, 0x00) && e.pixel_is(8, 101, 0x21));
        assert(e.pixel_is(0, 200, 0x00) && e.pixel_is(8, 200, 0x21));
    }
}
