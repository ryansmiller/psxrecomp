/* Surface-backend table (gpu_hw_renderer.h): the OpenGL renderer's
 * presentation and host controls, and the active-backend selector. */
#include "gpu_hw_renderer.h"
#include "gpu_gl_renderer.h"

static const HwRenderer k_opengl = {
    .name = "opengl",
    .init_context = gl_renderer_init_context,
    .shutdown = gl_renderer_shutdown,
    .present = gl_renderer_present,
    .present_vram = gl_renderer_present_vram,
    .present_wide_fbo = gl_renderer_present_wide_fbo,
    .present_blank = gl_renderer_present_blank,
    .present_hold_last = gl_renderer_present_hold_last,
    .present_rect_dirty = gl_renderer_present_rect_dirty,
    .present_probe_reset = gl_renderer_present_probe_reset,
    .present_probe_take = gl_renderer_present_probe_take,
    .invalidate_present = gl_renderer_invalidate_present,
    .note_present_depth24 = gl_renderer_note_present_depth24,
    .capture_display_hires = gl_renderer_capture_display_hires,
    .set_swap_interval = gl_renderer_set_swap_interval,
    .swap_timing_json = gl_renderer_swap_timing_json,
    .set_display_rate_timing = gl_renderer_set_display_rate_timing,
    .invalidate_display_rate = gl_renderer_invalidate_display_rate,
    .sync_cpu = gl_renderer_sync_cpu,
    .flush_cpu_uploads = gl_renderer_flush_cpu_uploads,
    .restage_vram_after_savestate = gl_renderer_restage_vram_after_savestate,
    .set_cpu_auth_dual = gl_renderer_set_cpu_auth_dual,
    .cpu_auth_dual = gl_renderer_cpu_auth_dual,
    .set_internal_scale_live = gl_renderer_set_internal_scale_live,
    .internal_scale_live_capable = gl_renderer_internal_scale_live_capable,
    .dynamic_resolution_ceiling = gl_renderer_dynamic_resolution_ceiling,
    .step_internal_scale_now = gl_renderer_step_internal_scale_now,
    .set_display_aspect = gl_renderer_set_display_aspect,
    .fit_wide_aspect = gl_renderer_fit_wide_aspect,
    .set_scanlines = gl_renderer_set_scanlines,
    .set_fmv_filter = gl_renderer_set_fmv_filter,
    .set_dithering = gl_renderer_set_dithering,
    .set_post_aa = gl_renderer_set_post_aa,
    .set_bloom = gl_renderer_set_bloom,
    .set_post_shader = gl_renderer_set_post_shader,
};

static const HwRenderer *g_active = &k_opengl;

const HwRenderer *hw_renderer(void) { return g_active; }
void hw_renderer_select(const HwRenderer *renderer) { g_active = renderer ? renderer : &k_opengl; }
const HwRenderer *hw_renderer_opengl(void) { return &k_opengl; }
