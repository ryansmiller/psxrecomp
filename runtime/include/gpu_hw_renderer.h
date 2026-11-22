#ifndef PSX_GPU_HW_RENDERER_H
#define PSX_GPU_HW_RENDERER_H

/* Presentation and host-side controls of a GPU-surface backend.
 *
 * The gr_* facade (gpu_render.h) carries the PS1 GPU commands. A backend that
 * draws into its own GPU surfaces (OpenGL today, Metal next) also owns
 * presentation, VRAM coherence with the CPU copy, internal-resolution changes
 * and image settings. main.cpp reaches those through this table, so a second
 * surface backend implements the same operations instead of the OpenGL
 * renderer's names. Each member has the signature and meaning of the
 * gl_renderer_* function it is named after (gpu_gl_renderer.h).
 *
 * OpenGL-only features (render passes, stereo/OpenXR, frame generation,
 * interpolation, HD texture packs, PGXP, diagnostics) keep their
 * gl_renderer_* entry points; they stay inert while OpenGL is not the active
 * surface backend. */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SDL_Window;

typedef struct HwRenderer {
    const char *name;
    int  (*init_context)(struct SDL_Window *win);
    void (*shutdown)(void);

    /* Presentation */
    void (*present)(const uint32_t *pixels, int src_w, int src_h, int linear,
                    int force_4_3, int content_w, uint64_t guest_cycle,
                    uint64_t guest_frame, uint64_t guest_epoch,
                    uint64_t stream_generation, int stream_lba, int delivered_lba,
                    uint32_t decode_serial);
    void (*present_vram)(int disp_x, int disp_y, int w, int h, int linear, int force_4_3);
    int  (*present_wide_fbo)(int disp_x, int disp_y, int disp_h, int linear);
    void (*present_blank)(void);
    int  (*present_hold_last)(void);
    int  (*present_rect_dirty)(int disp_x, int disp_y, int w, int h);
    void (*present_probe_reset)(void);
    void (*present_probe_take)(uint64_t *skip_delta, uint64_t *swap_delta,
                               uint64_t *dirty_mark_delta, int *force_remaining);
    void (*invalidate_present)(void);
    void (*note_present_depth24)(int d24);
    int  (*capture_display_hires)(uint32_t *out, int pitch_bytes,
                                  int x, int y, int w, int h);
    void (*set_swap_interval)(int interval);
    void (*swap_timing_json)(char *out, int cap);
    void (*set_display_rate_timing)(uint64_t midpoint_ns, uint64_t native_ns,
                                    uint64_t display_interval_ns);
    void (*invalidate_display_rate)(void);

    /* VRAM coherence with the CPU copy */
    void (*sync_cpu)(void);
    void (*flush_cpu_uploads)(void);
    void (*restage_vram_after_savestate)(void);
    void (*set_cpu_auth_dual)(int on);
    int  (*cpu_auth_dual)(void);

    /* Internal resolution */
    int  (*set_internal_scale_live)(int scale);
    int  (*internal_scale_live_capable)(void);
    int  (*dynamic_resolution_ceiling)(void);
    int  (*step_internal_scale_now)(int scale);

    /* Image settings */
    void (*set_display_aspect)(int num, int den);
    int  (*fit_wide_aspect)(int disp_w, int *num, int *den);
    void (*set_scanlines)(int on, float strength);
    void (*set_fmv_filter)(int cfg_value);
    void (*set_dithering)(int mode);
    int  (*set_post_aa)(int mode);
    void (*set_bloom)(float strength);
    void (*set_post_shader)(const char *path);
} HwRenderer;

/* The surface backend main.cpp presents through. OpenGL until another
 * surface backend is selected. Never NULL. */
const HwRenderer *hw_renderer(void);
/* Selects the surface backend; NULL restores OpenGL. */
void hw_renderer_select(const HwRenderer *renderer);
const HwRenderer *hw_renderer_opengl(void);

#ifdef __cplusplus
}
#endif

#endif
