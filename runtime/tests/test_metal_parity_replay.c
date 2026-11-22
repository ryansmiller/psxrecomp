/* Replays a renderer recording (gpu_render_record.h) through the Metal surface
 * backend's table and captures every displayed frame at internal resolution,
 * in the same frames.tsv format as test_gpu_parity_replay.c (OpenGL), so the
 * two frame sets can be compared frame by frame.
 *
 * Usage: metal_parity_replay RECORDING OUT_DIR [IMAGE_EVERY] [SCALE]
 * Writes OUT_DIR/frames.tsv (frame, op, display rect, size, FNV-1a digest of
 * the ARGB frame) and OUT_DIR/frame-NNNNN.ppm for every IMAGE_EVERY-th frame
 * (default 60; 0 = none). SCALE replays at another internal scale than the
 * recording's (no OpenGL reference exists for it; the images show the
 * result). The opening snapshot goes in as test_gpu_parity_replay.c applies
 * it, with the same two limits: its high-resolution mirror is not loaded,
 * and 24-bit display is not modelled. Wide compositor ops are not replayed:
 * the Metal backend has no wide surfaces yet. With PARITY_BACKEND=software
 * in the environment the software renderer replays instead, as a faithful
 * reference at 1x, 2x and 4x. PARITY_DIRECT=1 captures each Metal frame
 * through the direct-present surface (metal_renderer_present_direct) instead
 * of the readback; the digests must not change. */
#include <CoreVideo/CoreVideo.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpu_metal_renderer.h"
#include "gpu_render.h"
#include "gpu_render_record.h"
#include "gpu_sw_renderer.h"

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

static const GpuRenderBackend k_software = {
    .name = "software", .init = sw_renderer_init, .set_scale = sw_renderer_set_scale,
    .scale = sw_renderer_scale, .set_texture_filter = sw_set_texture_filter,
    .set_semi_transparency = sw_set_semi_transparency, .set_mask_bits = sw_set_mask_bits,
    .set_texture_window = sw_set_texture_window,
    .set_color_modulation = sw_set_color_modulation,
    .fill_rect = sw_fill_rect, .copy_rect = sw_copy_rect,
    .draw_flat_triangle = sw_draw_flat_triangle,
    .draw_gouraud_triangle = sw_draw_gouraud_triangle,
    .draw_textured_triangle = sw_draw_textured_triangle,
    .draw_shaded_textured_triangle = sw_draw_shaded_textured_triangle,
    .draw_flat_rect = sw_draw_flat_rect, .draw_textured_rect = sw_draw_textured_rect,
    .draw_textured_rect_scaled = sw_draw_textured_rect_scaled,
    .draw_line = sw_draw_line, .draw_shaded_line = sw_draw_shaded_line,
    .render_display_hires = sw_render_display_hires,
    .vram_write = sw_vram_write, .vram_read = sw_vram_read,
    .vram_transfer_in = sw_vram_transfer_in, .vram_transfer_out = sw_vram_transfer_out,
    .set_draw_area = sw_set_draw_area, .get_draw_area = sw_get_draw_area,
    .set_draw_offset = sw_set_draw_offset,
};

static _Alignas(16384) uint16_t vram[1024 * 512];
static uint16_t staging[1024 * 512];

typedef struct {
    const uint8_t *p, *end;
    int bad;
} GrrReader;

static void grr_take(GrrReader *r, void *out, size_t size) {
    if ((size_t)(r->end - r->p) < size) {
        r->bad = 1;
        memset(out, 0, size);
        return;
    }
    memcpy(out, r->p, size);
    r->p += size;
}

static int32_t grr_i32(GrrReader *r) { int32_t v; grr_take(r, &v, sizeof v); return v; }
static float grr_f32(GrrReader *r) { float v; grr_take(r, &v, sizeof v); return v; }

/* gpu_sw_renderer.c SwStateHeader (version 1 snapshots). */
typedef struct {
    char magic[8];
    int32_t scale, clip[4], offset[2];
    int32_t semi_enabled, semi_mode, mask_set, mask_check;
    uint8_t tw[4], mod[3], pad;
    int32_t raw_texture, texture_filter, texture_filter_req, faithful;
    int32_t precise_valid, precise_x16[3], precise_y16[3];
    int32_t perspective_valid;
    float perspective_q[3];
    int32_t uv_lim[4];
    int32_t view[4];
} GrrState;

static uint8_t *read_file(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *data = n > 0 ? (uint8_t *)malloc((size_t)n) : NULL;
    if (data && fread(data, 1, (size_t)n, f) != (size_t)n) { free(data); data = NULL; }
    fclose(f);
    *size = data ? (size_t)n : 0;
    return data;
}

static uint64_t fnv(const void *p, size_t n, uint64_t h) {
    const uint8_t *b = (const uint8_t *)p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 0x100000001b3ull; }
    return h;
}

static void write_ppm(const char *path, const uint32_t *argb, int w, int h) {
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    uint8_t *row = (uint8_t *)malloc((size_t)w * 3);
    for (int y = 0; y < h && row; y++) {
        for (int x = 0; x < w; x++) {
            const uint32_t c = argb[(size_t)y * w + x];
            row[x * 3] = (uint8_t)(c >> 16);
            row[x * 3 + 1] = (uint8_t)(c >> 8);
            row[x * 3 + 2] = (uint8_t)c;
        }
        fwrite(row, 3, (size_t)w, f);
    }
    free(row);
    fclose(f);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s RECORDING OUT_DIR [IMAGE_EVERY] [SCALE]\n", argv[0]);
        return 2;
    }
    const int image_every = argc > 3 ? atoi(argv[3]) : 60;
    size_t size = 0;
    uint8_t *data = read_file(argv[1], &size);
    GrrReader r = {data, data + size, 0};
    char magic[8];
    grr_take(&r, magic, sizeof magic);
    const uint32_t version = (uint32_t)grr_i32(&r);
    const uint32_t state_size = (uint32_t)grr_i32(&r);
    GrrState h;
    if (!data || r.bad || memcmp(magic, GRR_MAGIC, 8) != 0 || version != GRR_VERSION ||
        (size_t)(r.end - r.p) < state_size || state_size < sizeof h + sizeof vram) {
        fprintf(stderr, "FAIL %s is not a version %u renderer recording\n", argv[1], GRR_VERSION);
        return 2;
    }
    memcpy(&h, r.p, sizeof h);
    if (memcmp(h.magic, "PSXSWS1", 7) != 0 || h.scale < 1) {
        fprintf(stderr, "FAIL opening snapshot\n");
        return 2;
    }
    const int scale = argc > 4 ? atoi(argv[4]) : h.scale;
    const char *which = getenv("PARITY_BACKEND");
    const int software = which && strcmp(which, "software") == 0;
    const char *dflag = getenv("PARITY_DIRECT");
    const int direct = !software && dflag && strcmp(dflag, "1") == 0;
    const GpuRenderBackend *k = software ? &k_software : metal_backend_get();
    if (!k) { fprintf(stderr, "SKIP no Metal device\n"); return 77; }

    memcpy(vram, r.p + sizeof h, sizeof vram);
    k->init(vram);
    k->set_scale(scale);
    if (k->scale() != scale) { fprintf(stderr, "FAIL scale %d not applied\n", scale); return 1; }
    memcpy(staging, vram, sizeof vram);
    k->vram_transfer_in(0, 0, 1024, 512, staging);
    k->set_draw_area(h.clip[0], h.clip[1], h.clip[2], h.clip[3]);
    k->set_draw_offset(h.offset[0], h.offset[1]);
    k->set_semi_transparency(h.semi_enabled, h.semi_mode);
    k->set_mask_bits(h.mask_set, h.mask_check);
    k->set_texture_window((uint32_t)h.tw[0] | (uint32_t)h.tw[1] << 5 |
                          (uint32_t)h.tw[2] << 10 | (uint32_t)h.tw[3] << 15);
    k->set_color_modulation(h.mod[0] << 3, h.mod[1] << 3, h.mod[2] << 3, h.raw_texture);
    k->set_texture_filter(h.texture_filter_req);
    r.p += state_size;

    char path[4096];
    snprintf(path, sizeof path, "%s/frames.tsv", argv[2]);
    FILE *index = fopen(path, "w");
    if (!index) { fprintf(stderr, "FAIL cannot write %s\n", path); return 2; }
    fprintf(index, "frame\top\tx\ty\tw\th\twidth\theight\tdigest\n");
    const size_t frame_cap = (size_t)1024 * scale * 512 * scale;
    uint32_t *frame = (uint32_t *)malloc(frame_cap * 4);
    static uint16_t pixels[1024 * 512];
    uint64_t ops = 0, frames = 0, skipped = 0, all = 0xcbf29ce484222325ull;
    int ok = frame != NULL;
    while (ok) {
        uint8_t code;
        grr_take(&r, &code, 1);
        if (r.bad || code >= GRR_OP_COUNT) { fprintf(stderr, "FAIL truncated at op %llu\n",
                                                     (unsigned long long)ops); ok = 0; break; }
        if (code == GRR_END) break;
        if (code == GRR_UNSUPPORTED) { fprintf(stderr, "FAIL unsupported op\n"); ok = 0; break; }
        ++ops;
        int32_t a[20];
        switch (code) {
        case GRR_SET_SCALE: (void)grr_i32(&r); break;
        case GRR_SET_TEXTURE_FILTER: k->set_texture_filter(grr_i32(&r)); break;
        case GRR_SET_SEMI: a[0] = grr_i32(&r); a[1] = grr_i32(&r);
            k->set_semi_transparency(a[0], a[1]); break;
        case GRR_SET_MASK: a[0] = grr_i32(&r); a[1] = grr_i32(&r); k->set_mask_bits(a[0], a[1]);
            break;
        case GRR_SET_TEXWIN: k->set_texture_window((uint32_t)grr_i32(&r)); break;
        case GRR_SET_COLOR_MOD:
            for (int i = 0; i < 4; ++i) a[i] = grr_i32(&r);
            k->set_color_modulation(a[0], a[1], a[2], a[3]); break;
        case GRR_SET_PRECISE:   /* PGXP: the Metal backend has no sub-pixel path yet */
            for (int i = 0; i < 7; ++i) a[i] = grr_i32(&r);
            if (a[0]) ++skipped;
            break;
        case GRR_SET_PERSPECTIVE:
            a[0] = grr_i32(&r);
            for (int i = 0; i < 3; ++i) (void)grr_f32(&r);
            if (a[0]) ++skipped;
            break;
        case GRR_FILL:
            for (int i = 0; i < 5; ++i) a[i] = grr_i32(&r);
            k->fill_rect(a[0], a[1], a[2], a[3], (uint16_t)a[4]); break;
        case GRR_COPY:
            for (int i = 0; i < 6; ++i) a[i] = grr_i32(&r);
            k->copy_rect(a[0], a[1], a[2], a[3], a[4], a[5]); break;
        case GRR_FLAT_TRI:
            for (int i = 0; i < 7; ++i) a[i] = grr_i32(&r);
            k->draw_flat_triangle(a[0], a[1], a[2], a[3], a[4], a[5], (uint16_t)a[6]); break;
        case GRR_GOURAUD_TRI:
            for (int i = 0; i < 9; ++i) a[i] = grr_i32(&r);
            k->draw_gouraud_triangle(a[0], a[1], (uint16_t)a[2], a[3], a[4], (uint16_t)a[5],
                                     a[6], a[7], (uint16_t)a[8]); break;
        case GRR_TEX_TRI:
            for (int i = 0; i < 15; ++i) a[i] = grr_i32(&r);
            k->draw_textured_triangle(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                                      a[9], a[10], a[11], (uint16_t)a[12], (uint16_t)a[13],
                                      (uint16_t)a[14]); break;
        case GRR_SHADED_TEX_TRI:
            for (int i = 0; i < 19; ++i) a[i] = grr_i32(&r);
            k->draw_shaded_textured_triangle(a[0], a[1], a[2], a[3], (uint32_t)a[4],
                                             a[5], a[6], a[7], a[8], (uint32_t)a[9],
                                             a[10], a[11], a[12], a[13], (uint32_t)a[14],
                                             (uint16_t)a[15], (uint16_t)a[16],
                                             (uint16_t)a[17], a[18]); break;
        case GRR_FLAT_RECT:
            for (int i = 0; i < 5; ++i) a[i] = grr_i32(&r);
            k->draw_flat_rect(a[0], a[1], a[2], a[3], (uint16_t)a[4]); break;
        case GRR_TEX_RECT:
            for (int i = 0; i < 9; ++i) a[i] = grr_i32(&r);
            k->draw_textured_rect(a[0], a[1], a[2], a[3], a[4], a[5], (uint16_t)a[6],
                                  (uint16_t)a[7], (uint16_t)a[8]); break;
        case GRR_TEX_RECT_SCALED:
            for (int i = 0; i < 11; ++i) a[i] = grr_i32(&r);
            k->draw_textured_rect_scaled(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],
                                         (uint16_t)a[8], (uint16_t)a[9], (uint16_t)a[10]);
            break;
        case GRR_LINE:
            for (int i = 0; i < 5; ++i) a[i] = grr_i32(&r);
            k->draw_line(a[0], a[1], a[2], a[3], (uint16_t)a[4]); break;
        case GRR_SHADED_LINE:
            for (int i = 0; i < 6; ++i) a[i] = grr_i32(&r);
            k->draw_shaded_line(a[0], a[1], (uint16_t)a[2], a[3], a[4], (uint16_t)a[5]); break;
        case GRR_RENDER_DISPLAY:
            for (int i = 0; i < 6; ++i) a[i] = grr_i32(&r);
            break;
        case GRR_RENDER_DISPLAY_HIRES: {
            for (int i = 0; i < 6; ++i) a[i] = grr_i32(&r);
            const int w = a[3] * scale, hh = a[4] * scale;
            if (w <= 0 || hh <= 0 || (size_t)w * hh > frame_cap) break;
            memset(frame, 0, (size_t)w * hh * 4);
            if (direct) {
                /* Alternate the two surfaces, as the window does. */
                const int slot = (int)(frames & 1);
                CVPixelBufferRef pb = (CVPixelBufferRef)
                    metal_renderer_present_pixel_buffer(slot, 1024 * scale, 512 * scale + 1);
                int got_w = 0, got_h = 0;
                if (!pb || !metal_renderer_present_direct(slot, a[1], a[2], a[3], a[4], 0, 0, 0,
                                                          &got_w, &got_h) ||
                    got_w != w || got_h != hh) {
                    fprintf(stderr, "FAIL direct present unavailable\n");
                    ok = 0;
                    break;
                }
                CVPixelBufferLockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
                const uint8_t *base = (const uint8_t *)CVPixelBufferGetBaseAddress(pb);
                const size_t pitch = CVPixelBufferGetBytesPerRow(pb);
                for (int y = 0; y < hh; ++y)
                    memcpy(frame + (size_t)y * w, base + (size_t)y * pitch, (size_t)w * 4);
                CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
            } else if (!k->render_display_hires(frame, w * 4, a[1], a[2], a[3], a[4])) {
                fprintf(stderr, "FAIL frame %llu not captured\n", (unsigned long long)frames);
                ok = 0;
                break;
            }
            const uint64_t digest = fnv(frame, (size_t)w * hh * 4, 0xcbf29ce484222325ull);
            all = fnv(&digest, sizeof digest, all);
            fprintf(index, "%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%016llx\n",
                    (unsigned long long)frames, (unsigned long long)ops, a[1], a[2], a[3],
                    a[4], w, hh, (unsigned long long)digest);
            if (image_every > 0 && frames % (uint64_t)image_every == 0) {
                snprintf(path, sizeof path, "%s/frame-%05llu.ppm", argv[2],
                         (unsigned long long)frames);
                write_ppm(path, frame, w, hh);
            }
            ++frames;
            break;
        }
        case GRR_VRAM_WRITE:
            for (int i = 0; i < 3; ++i) a[i] = grr_i32(&r);
            vram[(a[1] & 511) * 1024 + (a[0] & 1023)] = (uint16_t)a[2];
            k->vram_write(a[0], a[1], (uint16_t)a[2]); break;
        case GRR_VRAM_READ:
            for (int i = 0; i < 3; ++i) a[i] = grr_i32(&r);
            (void)k->vram_read(a[0], a[1]); break;
        case GRR_TRANSFER_IN:
        case GRR_TRANSFER_OUT: {
            for (int i = 0; i < 4; ++i) a[i] = grr_i32(&r);
            const size_t count = a[2] > 0 && a[3] > 0 ? (size_t)a[2] * (size_t)a[3] : 0;
            if (count > (size_t)1024 * 512) { ok = 0; break; }
            grr_take(&r, pixels, count * sizeof *pixels);
            if (code == GRR_TRANSFER_IN) {
                for (int y = 0; y < a[3]; ++y)
                    for (int x = 0; x < a[2]; ++x)
                        vram[((a[1] + y) & 511) * 1024 + ((a[0] + x) & 1023)] =
                            pixels[(size_t)y * a[2] + x];
                memcpy(staging, pixels, count * sizeof *pixels);
                k->vram_transfer_in(a[0], a[1], a[2], a[3], staging);
            } else {
                k->vram_transfer_out(a[0], a[1], a[2], a[3], staging);
            }
            break;
        }
        case GRR_SET_DRAW_AREA:
            for (int i = 0; i < 4; ++i) a[i] = grr_i32(&r);
            k->set_draw_area(a[0], a[1], a[2], a[3]); break;
        case GRR_GET_DRAW_AREA:
            for (int i = 0; i < 4; ++i) a[i] = grr_i32(&r);
            break;
        case GRR_SET_DRAW_OFFSET: a[0] = grr_i32(&r); a[1] = grr_i32(&r);
            k->set_draw_offset(a[0], a[1]); break;
        case GRR_BATCH_BEGIN: case GRR_BATCH_END: case GRR_BATCH_SYNC: break;
        case GRR_WIDE_CONFIGURE: (void)grr_i32(&r); (void)grr_i32(&r); break;
        case GRR_WIDE_SET_VIEW:
            for (int i = 0; i < 4; ++i) a[i] = grr_i32(&r);
            if (a[0]) ++skipped;
            break;
        case GRR_WIDE_SET_TARGET: (void)grr_i32(&r); ++skipped; break;
        case GRR_WIDE_DISABLE_TARGET: break;
        default:
            fprintf(stderr, "FAIL op %u has no replay\n", code);
            ok = 0;
        }
        if (r.bad) { fprintf(stderr, "FAIL truncated at op %llu\n", (unsigned long long)ops);
                     ok = 0; }
    }
    if (!software) metal_renderer_finish();
    fclose(index);
    free(frame);
    free(data);
    printf("ops=%llu frames=%llu skipped_state=%llu\n", (unsigned long long)ops,
           (unsigned long long)frames, (unsigned long long)skipped);
    printf("frames_digest=%016llx\n", (unsigned long long)all);
    return ok && frames ? 0 : 1;
}
