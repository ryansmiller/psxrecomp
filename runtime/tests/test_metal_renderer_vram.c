/* Metal surface backend (gpu_metal_renderer.m), VRAM operations: fills, VRAM
 * copies (overlapping and not, wrapping, every mask mode), CPU transfers,
 * pokes and readbacks, against the software renderer at the same scale.
 *
 * One seeded script runs twice: through the Metal backend's table, then
 * through the software renderer at the same internal scale. Native VRAM (read
 * back as the guest would) and the display decoded at internal resolution
 * must be identical, at 1x, 2x and 4x. The Metal backend keeps its CPU
 * mirror with the software renderer at 1x, so the runs are sequential. */
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpu_metal_renderer.h"
#include "gpu_sw_renderer.h"

enum { W = 1024, H = 512, VIEW_W = 320, VIEW_H = 240 };

int g_ws_bd_stretch_on, g_ws_bd_stretch_pct;
int psx_ws_prim_in_backdrop(void) { return 0; }
int g_psx_vram_dirty_tracking = 1;
void gpu_vram_dirty_mark_row_impl(uint32_t y) { (void)y; }
void gpu_vram_dirty_mark_rect(int x, int y, int w, int h) { (void)x; (void)y; (void)w; (void)h; }
void gpu_vram_dirty_mark_all(void) {}
void psx_fatal_halt(const char *reason) {
    fprintf(stderr, "FAIL fatal halt: %s\n", reason);
    exit(1);
}

static _Alignas(16384) uint16_t vram[W * H];
static uint16_t native[2][W * H];
static uint32_t *display[2];

typedef struct {
    void (*set_mask)(int set, int check);
    void (*fill)(int x, int y, int w, int h, uint16_t c);
    void (*copy)(int sx, int sy, int dx, int dy, int w, int h);
    void (*write)(int x, int y, uint16_t p);
    uint16_t (*read)(int x, int y);
    void (*in)(int x, int y, int w, int h, const uint16_t *d);
    void (*out)(int x, int y, int w, int h, uint16_t *d);
    int (*hires)(uint32_t *o, int pitch, int x, int y, int w, int h);
} Ops;

static uint32_t rng;
static uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

/* gpu.c's CPU->VRAM path: mask test against VRAM, then hand over a copy. */
static void upload(const Ops *o, int set, int check, int x, int y, int w, int h) {
    static uint16_t data[W * H];
    for (int i = 0; i < w * h; ++i) {
        const int vx = (x + i % w) & (W - 1), vy = (y + i / w) & (H - 1);
        uint16_t p = (uint16_t)rnd();
        if (check && (o->read(vx, vy) & 0x8000)) p = vram[vy * W + vx];
        else if (set) p |= 0x8000;
        vram[vy * W + vx] = p;
        data[i] = p;
    }
    o->in(x, y, w, h, data);
}

/* PSX_METAL_VRAM_TEST_TRACE=1: digest native VRAM after every step, to name
 * the first step where the two runs part. */
static uint64_t trace[2][400];
static int tracing;
static uint32_t last_kind[2][400];
static uint64_t fnv(const void *p, size_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; ++i) { h ^= ((const uint8_t *)p)[i]; h *= 0x100000001b3ull; }
    return h;
}

static void script(const Ops *o, int scale, int slot, uint32_t seed) {
    rng = seed;
    int set = 0, check = 0;
    const int steps = getenv("PSX_METAL_VRAM_TEST_STEPS") ? atoi(getenv("PSX_METAL_VRAM_TEST_STEPS")) : 400;
    for (int step = 0; step < steps; ++step) {
        if (tracing && step) {
            static uint16_t snap[W * H];
            const uint32_t keep = rng;
            o->out(0, 0, W, H, snap);
            rng = keep;
            trace[slot][step - 1] = fnv(snap, sizeof snap);
        }
        const uint32_t k = rnd() % 10;
        last_kind[slot][step] = k;
        const int x = (int)(rnd() % W), y = (int)(rnd() % H);
        if (k == 0) {
            set = (int)(rnd() & 1);
            check = (int)(rnd() & 1);
            o->set_mask(set, check);
        } else if (k <= 2) {
            const int fw = (int)(rnd() % 200) + 1, fh = (int)(rnd() % 200) + 1;
            const uint16_t fc = (uint16_t)(rnd() & 0xFFFF);
            o->fill(x, y, fw, fh, fc);
        } else if (k <= 5) {
            /* Overlapping copies (small offsets) as often as distant ones. */
            const int near = rnd() & 1;
            const int sx = near ? x + (int)(rnd() % 9) - 4 : (int)(rnd() % W);
            const int sy = near ? y + (int)(rnd() % 9) - 4 : (int)(rnd() % H);
            const int cw = (int)(rnd() % 160) + 1, ch = (int)(rnd() % 160) + 1;
            o->copy(sx, sy, x, y, cw, ch);
        } else if (k <= 7) {
            const int uw = (int)(rnd() % 96) + 1, uh = (int)(rnd() % 96) + 1;
            upload(o, set, check, x, y, uw, uh);
        } else if (k == 8) {
            o->write(x, y, (uint16_t)rnd());
            vram[y * W + x] = o->read(x, y);
        } else {
            (void)o->read(x, y);
        }
    }
    o->out(0, 0, W, H, native[slot]);
    const int n = o->hires(display[slot], VIEW_W * scale * 4, 352, 16, VIEW_W, VIEW_H);
    assert(n == VIEW_W * scale * VIEW_H * scale);
}

static void sw_mask(int set, int check) { sw_set_mask_bits(set, check); }

int main(void) {
    const GpuRenderBackend *metal = metal_backend_get();
    tracing = getenv("PSX_METAL_VRAM_TEST_TRACE") != NULL;
    if (!metal) {
        printf("SKIP no Metal device with read-write textures\n");
        return 77;
    }
    /* The software reference stops at SW_MAX_INTERNAL_SCALE (4). */
    static const int scales[] = {1, 2, 4};
    for (size_t si = 0; si < sizeof scales / sizeof scales[0]; ++si) {
        const int scale = scales[si];
        for (int i = 0; i < 2; ++i)
            display[i] = malloc((size_t)VIEW_W * scale * VIEW_H * scale * 4);
        for (uint32_t seed = 1; seed <= 6; ++seed) {
            /* Same opening VRAM for both runs. */
            rng = seed * 2654435761u;
            for (int i = 0; i < W * H; ++i) vram[i] = (uint16_t)rnd();
            static uint16_t opening[W * H];
            memcpy(opening, vram, sizeof vram);

            metal->set_scale(scale);
            metal->init(vram);
            metal->set_draw_area(0, 0, W - 1, H - 1);
            const Ops m = {metal->set_mask_bits, metal->fill_rect, metal->copy_rect,
                           metal->vram_write, metal->vram_read, metal->vram_transfer_in,
                           metal->vram_transfer_out, metal->render_display_hires};
            script(&m, scale, 0, seed);

            memcpy(vram, opening, sizeof vram);
            sw_renderer_init(vram);
            sw_renderer_set_scale(scale);
            /* A new mirror starts blank; load it from VRAM as the Metal
             * backend's init loads its surfaces. */
            sw_vram_transfer_in(0, 0, W, H, opening);
            const Ops s = {sw_mask, sw_fill_rect, sw_copy_rect, sw_vram_write, sw_vram_read,
                           sw_vram_transfer_in, sw_vram_transfer_out, sw_render_display_hires};
            script(&s, scale, 1, seed);
            sw_renderer_set_scale(1);

            if (memcmp(native[0], native[1], sizeof native[0]) != 0) {
                for (int i = 0; tracing && i < 399; ++i)
                    if (trace[0][i] != trace[1][i]) {
                        printf("first differing step %d (kind %u)\n", i, last_kind[0][i]);
                        break;
                    }
                for (int i = 0; i < W * H; ++i)
                    if (native[0][i] != native[1][i]) {
                        printf("FAIL %dx seed %u: native VRAM differs first at (%d,%d): "
                               "metal %04x software %04x\n", scale, seed, i % W, i / W,
                               native[0][i], native[1][i]);
                        break;
                    }
                return 1;
            }
            const size_t px = (size_t)VIEW_W * scale * VIEW_H * scale;
            for (size_t i = 0; i < px; ++i)
                if (display[0][i] != display[1][i]) {
                    printf("FAIL %dx seed %u: display differs at pixel %zu: metal %08x "
                           "software %08x\n", scale, seed, i, display[0][i], display[1][i]);
                    return 1;
                }
        }
        printf("%dx: 6 scripts, native VRAM and display identical to software\n", scale);
        for (int i = 0; i < 2; ++i) free(display[i]);
    }
    return 0;
}
