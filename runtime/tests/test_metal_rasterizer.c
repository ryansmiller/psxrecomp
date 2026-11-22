/* macOS only. Build with the software renderer and gpu_metal_rasterizer.m. */
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpu_sw_renderer.h"
#include "../src/gpu_metal_rasterizer.h"

enum { width = 1024, height = 512, view_width = 320, view_height = 240 };

int g_ws_bd_stretch_on, g_ws_bd_stretch_pct;
int psx_ws_prim_in_backdrop(void) { return 0; }
int g_psx_vram_dirty_tracking = 1;
static unsigned char dirty[height];
void gpu_vram_dirty_mark_row_impl(uint32_t y) { dirty[y] = 1; }
void gpu_vram_dirty_mark_rect(int x, int y, int w, int h) {
    (void)x; (void)y; (void)w; (void)h;
}
void gpu_vram_dirty_mark_all(void) {}
/* gpu_metal_renderer.m links into every Metal target; this test never
 * selects that backend, so a halt there is a failure. */
void psx_fatal_halt(const char *reason) {
    fprintf(stderr, "FAIL fatal halt: %s\n", reason);
    exit(1);
}

static uint16_t vram[width * height], expected[width * height];
static uint32_t display[view_width * view_height * 16];
static uint32_t reference[view_width * view_height * 16];

/* batch queues the flat primitives between begin and end. The copy reads
 * pixels the earlier primitives wrote, so it must apply the queue first. */
static void draw(int metal, int batch, int scale, int blend, int check, int set) {
    assert(setenv("PSX_METAL_RASTERIZER", metal ? "1" : "0", 1) == 0);
    assert(setenv("PSX_METAL_BATCH_MIN_PIXELS", "0", 1) == 0);
    sw_renderer_init(vram);
    sw_renderer_set_scale(1);
    memset(vram, 0, sizeof(vram));
    memset(dirty, 0, sizeof(dirty));
    sw_renderer_set_scale(scale);
    sw_set_draw_area(3, 2, view_width - 1, view_height - 1);
    if (batch) sw_metal_batch_begin();
    sw_draw_flat_rect(4, 16, 259, 191, 0x2a11);
    sw_set_mask_bits(1, 0);
    sw_draw_flat_rect(3, 5, 31, 226, 0x4b21);
    sw_set_mask_bits(set, check);
    sw_set_semi_transparency(blend >= 0, blend);
    sw_draw_flat_triangle(346, 169, -40, 193, 154, -15, 0xed3a);
    sw_draw_flat_triangle(20, 4, 271, 4, 177, 280, 0x0725);
    sw_copy_rect(12, 20, 150, 60, 90, 70);
    sw_draw_flat_rect(-10, 39, 213, 193, 0x617c);
    sw_draw_flat_rect(6, 129, 10, 65, 0x7d65);
    sw_draw_flat_rect(4, 5, 0, 6, 0xffff);
    if (batch) {
        /* Each frame pixel read must observe queued writes. */
        assert(sw_vram_read(8, 140) == vram[140 * width + 8]);
        sw_metal_batch_end();
    }
}

int main(void) {
    for (int scale = 1; scale <= 4; scale *= 2) {
        uint64_t before = psx_metal_completed_draws();
        for (int blend = -1; blend < 4; ++blend) {
            for (int mask = 0; mask < 4; ++mask) {
                unsigned char expected_dirty[height];
                int pitch = view_width * scale * sizeof(uint32_t);
                size_t size = (size_t)pitch * view_height * scale;
                draw(0, 0, scale, blend, mask & 1, mask & 2);
                memcpy(expected, vram, sizeof(vram));
                memcpy(expected_dirty, dirty, sizeof(dirty));
                sw_render_display_hires(reference, pitch, 0, 0, view_width, view_height);

                for (int batch = 0; batch < 2; ++batch) {
                    uint64_t draw_start = psx_metal_completed_draws();
                    draw(1, batch, scale, blend, mask & 1, mask & 2);
                    assert(psx_metal_completed_draws() - draw_start == (scale == 1 ? 6 : 12));
                    sw_render_display_hires(display, pitch, 0, 0, view_width, view_height);
                    assert(memcmp(vram, expected, sizeof(vram)) == 0);
                    assert(memcmp(display, reference, size) == 0);
                    assert(memcmp(dirty, expected_dirty, sizeof(dirty)) == 0);
                }
            }
        }
        uint64_t zero_copy, copied;
        psx_metal_flush_counts(&zero_copy, &copied);
        printf("%dx: 20 cases, each exact for immediate and batched Metal VRAM, dirty rows and display (%llu GPU primitives; batch flushes so far %llu in place, %llu copied)\n",
               scale, (unsigned long long)(psx_metal_completed_draws() - before),
               (unsigned long long)zero_copy, (unsigned long long)copied);
    }
    return 0;
}
