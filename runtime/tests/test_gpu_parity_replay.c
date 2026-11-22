/* Replays a renderer recording (gpu_render_record.h) through the renderer
 * facade on a real, hidden OpenGL context and captures every displayed frame
 * at internal resolution. A later hardware backend (Metal) replays the same
 * recording the same way; comparing the two frame sets is the parity check.
 *
 * Usage (run_gl_render_thread.py --single --probe-args):
 *   probe RECORDING OUT_DIR [IMAGE_EVERY]
 * Writes OUT_DIR/frames.tsv (frame, op, display rect, size, FNV-1a digest of
 * the ARGB frame) and OUT_DIR/frame-NNNNN.ppm for every IMAGE_EVERY-th frame
 * (default 60; 0 = none).
 *
 * The opening snapshot's native VRAM and draw state go in through the facade.
 * Its high-resolution mirror cannot: the backend upscales native VRAM, so
 * areas the recording never redraws differ from the software capture. Every
 * backend starts the same way, so this does not affect a backend-to-backend
 * comparison. 24-bit display (FMV) is not modelled here
 * (gpu_display_is_depth24 is 0 in the fixture), also for every backend.
 * Frames are captured where the software renderer was asked to decode the
 * display (RENDER_DISPLAY_HIRES ops); RENDER_DISPLAY ops are skipped. */
#define main rt_fixture_main
#include "test_gl_render_thread.c"
#undef main

#include "gpu_render_record.h"

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

/* Applies the opening snapshot. Returns the snapshot's scale, or 0. */
static int apply_opening(const uint8_t *state, size_t size) {
    GrrState h;
    if (size < sizeof h) return 0;
    memcpy(&h, state, sizeof h);
    if (memcmp(h.magic, "PSXSWS1", 7) != 0 || h.scale < 1 ||
        size < sizeof h + (size_t)1024 * 512 * 2)
        return 0;
    memcpy(vram, state + sizeof h, sizeof vram);
    memcpy(staging, vram, sizeof vram);
    gr_vram_transfer_in(0, 0, 1024, 512, staging);
    gr_set_draw_area(h.clip[0], h.clip[1], h.clip[2], h.clip[3]);
    gr_set_draw_offset(h.offset[0], h.offset[1]);
    gr_set_semi_transparency(h.semi_enabled, h.semi_mode);
    mask_bits(h.mask_set, h.mask_check);
    gr_set_texture_window((uint32_t)h.tw[0] | (uint32_t)h.tw[1] << 5 |
                          (uint32_t)h.tw[2] << 10 | (uint32_t)h.tw[3] << 15);
    /* The snapshot keeps the 5-bit factors (8-bit colour >> 3). */
    gr_set_color_modulation(h.mod[0] << 3, h.mod[1] << 3, h.mod[2] << 3, h.raw_texture);
    gr_set_texture_filter(h.texture_filter_req);
    gr_wide_set_view(h.view[0], h.view[1], h.view[2], h.view[3]);
    return h.scale;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s RECORDING OUT_DIR [IMAGE_EVERY]\n", argv[0]);
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
    if (!data || r.bad || memcmp(magic, GRR_MAGIC, 8) != 0 || version != GRR_VERSION ||
        (size_t)(r.end - r.p) < state_size) {
        fprintf(stderr, "FAIL %s is not a version %u renderer recording\n", argv[1], GRR_VERSION);
        return 2;
    }
    GrrState opening;
    memcpy(&opening, r.p, state_size >= sizeof opening ? sizeof opening : 0);
    const int scale = state_size >= sizeof opening ? opening.scale : 0;
    if (scale < 1) { fprintf(stderr, "FAIL opening snapshot\n"); return 2; }

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "SKIP no video (%s)\n", SDL_GetError());
        return 77;
    }
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
#if defined(__APPLE__)
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG);
#endif
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
    SDL_Window *win = SDL_CreateWindow("Parity replay hidden test", 0, 0, 320, 240,
                                       SDL_WINDOW_OPENGL | SDL_WINDOW_HIDDEN);
    if (!win) { fprintf(stderr, "SKIP no window (%s)\n", SDL_GetError()); return 77; }
    gr_set_backend(GR_BACKEND_OPENGL);
    gr_init(vram);
    gr_set_scale(scale);
    gl_renderer_set_swap_interval(0);
    if (!gl_renderer_init_context(win)) { fprintf(stderr, "FAIL context\n"); return 2; }
    check(gr_backend() == GR_BACKEND_OPENGL, "OpenGL backend");
    check(gr_scale() == scale, "recording scale applied");
    check(apply_opening(r.p, state_size) == scale, "opening snapshot");
    r.p += state_size;

    char path[4096];
    snprintf(path, sizeof path, "%s/frames.tsv", argv[2]);
    FILE *index = fopen(path, "w");
    if (!index) { fprintf(stderr, "FAIL cannot write %s\n", path); return 2; }
    fprintf(index, "frame\top\tx\ty\tw\th\twidth\theight\tdigest\n");
    const size_t frame_cap = (size_t)1024 * scale * 512 * scale;
    uint32_t *frame = (uint32_t *)malloc(frame_cap * 4);
    static uint16_t pixels[1024 * 512];
    uint64_t ops = 0, frames = 0, all = 0xcbf29ce484222325ull;
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
        case GRR_SET_SCALE: a[0] = grr_i32(&r);
            check(a[0] == scale, "recording keeps one scale"); break;
        case GRR_SET_TEXTURE_FILTER: gr_set_texture_filter(grr_i32(&r)); break;
        case GRR_SET_SEMI: a[0] = grr_i32(&r); a[1] = grr_i32(&r);
            gr_set_semi_transparency(a[0], a[1]); break;
        case GRR_SET_MASK: a[0] = grr_i32(&r); a[1] = grr_i32(&r); mask_bits(a[0], a[1]); break;
        case GRR_SET_TEXWIN: gr_set_texture_window((uint32_t)grr_i32(&r)); break;
        case GRR_SET_COLOR_MOD:
            for (int i = 0; i < 4; ++i) a[i] = grr_i32(&r);
            gr_set_color_modulation(a[0], a[1], a[2], a[3]); break;
        case GRR_SET_PRECISE:
            for (int i = 0; i < 7; ++i) a[i] = grr_i32(&r);
            gr_set_precise_triangle(a[0], a[1], a[2], a[3], a[4], a[5], a[6]); break;
        case GRR_SET_PERSPECTIVE: {
            a[0] = grr_i32(&r);
            float q[3];
            for (int i = 0; i < 3; ++i) q[i] = grr_f32(&r);
            gr_set_perspective_triangle(a[0], q[0], q[1], q[2]);
            break;
        }
        case GRR_FILL:
            for (int i = 0; i < 5; ++i) a[i] = grr_i32(&r);
            gr_fill_rect(a[0], a[1], a[2], a[3], (uint16_t)a[4]); break;
        case GRR_COPY:
            for (int i = 0; i < 6; ++i) a[i] = grr_i32(&r);
            gr_copy_rect(a[0], a[1], a[2], a[3], a[4], a[5]); break;
        case GRR_FLAT_TRI:
            for (int i = 0; i < 7; ++i) a[i] = grr_i32(&r);
            gr_draw_flat_triangle(a[0], a[1], a[2], a[3], a[4], a[5], (uint16_t)a[6]); break;
        case GRR_GOURAUD_TRI:
            for (int i = 0; i < 9; ++i) a[i] = grr_i32(&r);
            gr_draw_gouraud_triangle(a[0], a[1], (uint16_t)a[2], a[3], a[4], (uint16_t)a[5],
                                     a[6], a[7], (uint16_t)a[8]); break;
        case GRR_TEX_TRI:
            for (int i = 0; i < 15; ++i) a[i] = grr_i32(&r);
            gr_draw_textured_triangle(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                                      a[9], a[10], a[11], (uint16_t)a[12], (uint16_t)a[13],
                                      (uint16_t)a[14]); break;
        case GRR_SHADED_TEX_TRI:
            for (int i = 0; i < 19; ++i) a[i] = grr_i32(&r);
            gr_draw_shaded_textured_triangle(a[0], a[1], a[2], a[3], (uint32_t)a[4],
                                             a[5], a[6], a[7], a[8], (uint32_t)a[9],
                                             a[10], a[11], a[12], a[13], (uint32_t)a[14],
                                             (uint16_t)a[15], (uint16_t)a[16],
                                             (uint16_t)a[17], a[18]); break;
        case GRR_FLAT_RECT:
            for (int i = 0; i < 5; ++i) a[i] = grr_i32(&r);
            gr_draw_flat_rect(a[0], a[1], a[2], a[3], (uint16_t)a[4]); break;
        case GRR_TEX_RECT:
            for (int i = 0; i < 9; ++i) a[i] = grr_i32(&r);
            gr_draw_textured_rect(a[0], a[1], a[2], a[3], a[4], a[5], (uint16_t)a[6],
                                  (uint16_t)a[7], (uint16_t)a[8]); break;
        case GRR_TEX_RECT_SCALED:
            for (int i = 0; i < 11; ++i) a[i] = grr_i32(&r);
            gr_draw_textured_rect_scaled(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],
                                         (uint16_t)a[8], (uint16_t)a[9], (uint16_t)a[10]);
            break;
        case GRR_LINE:
            for (int i = 0; i < 5; ++i) a[i] = grr_i32(&r);
            gr_draw_line(a[0], a[1], a[2], a[3], (uint16_t)a[4]); break;
        case GRR_SHADED_LINE:
            for (int i = 0; i < 6; ++i) a[i] = grr_i32(&r);
            gr_draw_shaded_line(a[0], a[1], (uint16_t)a[2], a[3], a[4], (uint16_t)a[5]); break;
        case GRR_RENDER_DISPLAY:
            for (int i = 0; i < 6; ++i) a[i] = grr_i32(&r);
            break;
        case GRR_RENDER_DISPLAY_HIRES: {
            for (int i = 0; i < 6; ++i) a[i] = grr_i32(&r);
            const int w = a[3] * scale, h = a[4] * scale;
            if (w <= 0 || h <= 0 || (size_t)w * h > frame_cap) break;
            memset(frame, 0, (size_t)w * h * 4);
            const int got = gl_renderer_capture_display_hires(frame, w * 4, a[1], a[2],
                                                               a[3], a[4]);
            check(got != 0, "frame captured");
            const uint64_t digest = fnv(frame, (size_t)w * h * 4, 0xcbf29ce484222325ull);
            all = fnv(&digest, sizeof digest, all);
            fprintf(index, "%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%016llx\n",
                    (unsigned long long)frames, (unsigned long long)ops, a[1], a[2], a[3],
                    a[4], w, h, (unsigned long long)digest);
            if (image_every > 0 && frames % (uint64_t)image_every == 0) {
                snprintf(path, sizeof path, "%s/frame-%05llu.ppm", argv[2],
                         (unsigned long long)frames);
                write_ppm(path, frame, w, h);
            }
            ++frames;
            gl_renderer_render_thread_frame_boundary();
            break;
        }
        case GRR_VRAM_WRITE:
            for (int i = 0; i < 3; ++i) a[i] = grr_i32(&r);
            vram[(a[1] & 511) * 1024 + (a[0] & 1023)] = (uint16_t)a[2];
            gr_vram_write(a[0], a[1], (uint16_t)a[2]); break;
        case GRR_VRAM_READ:
            for (int i = 0; i < 3; ++i) a[i] = grr_i32(&r);
            (void)gr_vram_read(a[0], a[1]); break;
        case GRR_TRANSFER_IN:
        case GRR_TRANSFER_OUT: {
            for (int i = 0; i < 4; ++i) a[i] = grr_i32(&r);
            const size_t count = a[2] > 0 && a[3] > 0 ? (size_t)a[2] * (size_t)a[3] : 0;
            if (count > (size_t)1024 * 512) { ok = 0; break; }
            grr_take(&r, pixels, count * sizeof *pixels);
            if (code == GRR_TRANSFER_IN) {
                /* gpu.c writes its array (already masked) and hands over a copy. */
                for (int y = 0; y < a[3]; ++y)
                    for (int x = 0; x < a[2]; ++x)
                        vram[((a[1] + y) & 511) * 1024 + ((a[0] + x) & 1023)] =
                            pixels[(size_t)y * a[2] + x];
                memcpy(staging, pixels, count * sizeof *pixels);
                gr_vram_transfer_in(a[0], a[1], a[2], a[3], staging);
            } else {
                gr_vram_transfer_out(a[0], a[1], a[2], a[3], staging);
            }
            break;
        }
        case GRR_SET_DRAW_AREA:
            for (int i = 0; i < 4; ++i) a[i] = grr_i32(&r);
            gr_set_draw_area(a[0], a[1], a[2], a[3]); break;
        case GRR_GET_DRAW_AREA:
            for (int i = 0; i < 4; ++i) a[i] = grr_i32(&r);
            break;
        case GRR_SET_DRAW_OFFSET: a[0] = grr_i32(&r); a[1] = grr_i32(&r);
            gr_set_draw_offset(a[0], a[1]); break;
        case GRR_BATCH_BEGIN: gr_batch_begin(); break;
        case GRR_BATCH_END: gr_batch_end(); break;
        case GRR_BATCH_SYNC: gr_batch_sync(); break;
        case GRR_WIDE_CONFIGURE: a[0] = grr_i32(&r); a[1] = grr_i32(&r);
            gr_wide_configure(a[0], a[1]); break;
        case GRR_WIDE_SET_VIEW:
            for (int i = 0; i < 4; ++i) a[i] = grr_i32(&r);
            gr_wide_set_view(a[0], a[1], a[2], a[3]); break;
        case GRR_WIDE_SET_TARGET: gr_wide_set_target(grr_i32(&r)); break;
        case GRR_WIDE_DISABLE_TARGET: gr_wide_disable_target(); break;
        default:
            fprintf(stderr, "FAIL op %u has no replay\n", code);
            ok = 0;
        }
        if (r.bad) { fprintf(stderr, "FAIL truncated at op %llu\n", (unsigned long long)ops);
                     ok = 0; }
    }
    fclose(index);
    free(frame);
    free(data);
    check(ok, "recording replayed");
    check(frames > 0, "frames captured");
    check(glGetError() == GL_NO_ERROR, "GL error");
    printf("ops=%llu frames=%llu\n", (unsigned long long)ops, (unsigned long long)frames);
    printf("frames_digest=%016llx\n", (unsigned long long)all);
    printf("checks=%d failures=%d\n", checks, failures);
    gl_renderer_shutdown();
    SDL_DestroyWindow(win);
    SDL_Quit();
    return failures ? 1 : 0;
}
