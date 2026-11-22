/*
 * gpu_sw_renderer.c — PS1 software rasterizer
 *
 * Operates on a 1024x512 uint16_t VRAM array (15-bit color + mask bit).
 * All drawing is clipped to the current draw area.
 *
 * Texture formats: 4-bit CLUT, 8-bit CLUT, 15-bit direct.
 * Supports: semi-transparency (4 modes), mask bit, texture window,
 *           color modulation for textured primitives.
 *
 * Internal-resolution supersampling (SSAA)
 * ----------------------------------------
 * When sw_renderer_set_scale(S>1) is active, the renderer maintains a
 * second buffer `g_hr` that is an S*-scaled mirror of VRAM. Every drawing
 * primitive is rasterized twice: once into native VRAM (byte-identical to
 * the S==1 path — VRAM stays the authoritative copy that the game reads
 * back for framebuffer effects, render-to-texture, transfers, etc.) and
 * once into g_hr at S* the linear resolution. Block operations (fill, copy,
 * CPU->VRAM transfer, single-pixel write) replicate/scale into g_hr so the
 * mirror stays coherent. Textures are ALWAYS sampled from native VRAM, so
 * texels keep their native resolution (point-sampled) while geometry edges
 * and shading are evaluated at the higher resolution. The display reads g_hr
 * (sw_render_display_hires) and the present path downsamples to the window,
 * which is true ordered-grid supersampling / anti-aliasing.
 *
 * When scale==1 (default) g_hr is NULL and the renderer behaves exactly as
 * it did before this feature existed.
 */

#include "gpu_sw_renderer.h"
#include "gpu_vram_dirty.h"
#include "gpu_sw_edges.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

#ifdef PSX_MACOS_METAL_RASTERIZER
#include "gpu_metal_rasterizer.h"
static int g_metal_rasterizer;
/* PSX_METAL_BATCH_CPU=1 applies batches with the CPU executor instead of
 * Metal, so tests can check the fallback against the reference. */
static int g_metal_batch_cpu;
/* Batches covering fewer pixels than this run on the CPU executor: a GPU
 * round trip costs more than drawing them. PSX_METAL_BATCH_MIN_PIXELS
 * overrides it; 0 sends every batch to Metal. */
static uint32_t g_metal_batch_min_pixels = 32768;
/* Applies queued Metal primitives before VRAM is read or otherwise written. */
#define METAL_BATCH_SYNC() sw_metal_batch_sync()
#else
#define METAL_BATCH_SYNC() ((void)0)
void sw_metal_batch_begin(void) {}
void sw_metal_batch_end(void) {}
void sw_metal_batch_sync(void) {}
void sw_metal_batch_stats(uint64_t *hazard_flushes, uint64_t *self_hazards) {
    *hazard_flushes = *self_hazards = 0;
}
#endif

/* ------------------------------------------------------------------ */
/* Constants                                                          */
/* ------------------------------------------------------------------ */

#define VRAM_WIDTH  1024
#define VRAM_HEIGHT 512

/* ------------------------------------------------------------------ */
/* State                                                              */
/* ------------------------------------------------------------------ */

static uint16_t *g_vram;

/* Hi-res supersampling mirror (see file header). g_scale==1 => disabled. */
static uint16_t *g_hr      = NULL;
static int       g_scale   = 1;
static int       g_hr_w    = VRAM_WIDTH;
static int       g_hr_h    = VRAM_HEIGHT;
static int       g_precise_valid = 0;
static int32_t   g_precise_x16[3], g_precise_y16[3];
static int       g_perspective_valid = 0;
static float     g_perspective_q[3];
static uint32_t  g_perspective_triangles = 0;

/* ---- Native-wide compositor (separate present surfaces) ------------------
 * Canonical VRAM stays faithful. For an opted-in wide game we ADDITIONALLY
 * mirror each framebuffer-targeting primitive into an independent wide surface
 * keyed by the back buffer's VRAM x-origin (base_x), at local x = vram_x -
 * base_x + OFFSET. Independent surfaces ⇒ no cross-buffer bleed; margins clear
 * cleanly; present reads the surface for the displayed buffer. Each surface is
 * (g_wide_w * scale) × (VRAM_HEIGHT * scale) 16-bit; y is unshifted so the
 * buffer's native VRAM y-band is reused directly. */
#define WIDE_MAX_SURF 4
static uint16_t *g_wide_surf[WIDE_MAX_SURF];   /* lazily-allocated surfaces */
static int       g_wide_base[WIDE_MAX_SURF];   /* base_x per surface (-1 = free) */
static int       g_wide_w        = 0;          /* wide width (native px); 0 = disabled */
static int       g_wide_off      = 0;          /* centering OFFSET (native px) */
static uint16_t *g_wide_cur      = NULL;       /* active mirror surface (NULL = no mirror) */
static int       g_wide_cur_base = 0;          /* base_x of g_wide_cur */
static void      wide_free_all(void);          /* defined with the surface helpers below */

/* Draw area clipping rectangle (native coordinates) */
static int g_clip_x1, g_clip_y1, g_clip_x2, g_clip_y2;

/* Draw offset */
static int g_offset_x, g_offset_y;

/* Semi-transparency state */
static int g_semi_trans_enabled;
static int g_semi_trans_mode; /* 0=B/2+F/2, 1=B+F, 2=B-F, 3=B+F/4 */

/* Mask bit state */
static int g_mask_set_bit;   /* force bit 15 on written pixels */
static int g_mask_check_bit; /* skip write if dest bit 15 is set */

/* Texture window (from GP0(E2h)) */
static uint8_t g_tw_mask_x, g_tw_mask_y;
static uint8_t g_tw_off_x, g_tw_off_y;

/* Color modulation for textured primitives */
static uint8_t g_mod_r, g_mod_g, g_mod_b;
static int g_raw_texture; /* 1 = skip modulation */

/* Texture filtering: 0 = nearest (native PSX), 1 = bilinear. */
static int g_texture_filter = 0;
/* The player's filter request, applied unless faithful authority is on. */
static int g_texture_filter_req = 0;
static int g_faithful_authority = 0;   /* see sw_set_faithful_authority */

/* ------------------------------------------------------------------ */
/* Helpers                                                            */
/* ------------------------------------------------------------------ */

static inline int clamp_i(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static inline int min_i(int a, int b) { return a < b ? a : b; }
static inline int max_i(int a, int b) { return a > b ? a : b; }

static inline uint16_t vram_get(int x, int y) {
    x &= (VRAM_WIDTH - 1);
    y &= (VRAM_HEIGHT - 1);
    return g_vram[y * VRAM_WIDTH + x];
}

/* ------------------------------------------------------------------ */
/* Render target — selects native VRAM or the hi-res mirror.          */
/*                                                                    */
/* `s` is the coordinate scale of this target (1 for native, S for    */
/* hi-res). Clip rectangle is expressed in the target's own pixel     */
/* space. Texture fetches never use the target: texels always come    */
/* from native VRAM via vram_get/texel_fetch.                         */
/* ------------------------------------------------------------------ */

typedef struct {
    uint16_t *buf;
    int       w, h;            /* buffer dimensions */
    int       s;               /* coordinate scale */
    int       cx1, cy1, cx2, cy2; /* clip rect (inclusive) in buffer space */
} RTarget;

static inline RTarget rt_native(void) {
    RTarget t;
    t.buf = g_vram; t.w = VRAM_WIDTH; t.h = VRAM_HEIGHT; t.s = 1;
    t.cx1 = g_clip_x1; t.cy1 = g_clip_y1; t.cx2 = g_clip_x2; t.cy2 = g_clip_y2;
    return t;
}

static inline RTarget rt_hires(void) {
    int s = g_scale;
    RTarget t;
    t.buf = g_hr; t.w = g_hr_w; t.h = g_hr_h; t.s = s;
    t.cx1 = g_clip_x1 * s;
    t.cy1 = g_clip_y1 * s;
    t.cx2 = g_clip_x2 * s + (s - 1);
    t.cy2 = g_clip_y2 * s + (s - 1);
    return t;
}

/* Native-wide mirror target: the active wide surface (g_wide_cur). X spans the
 * full wide width (that is where the revealed margins live), but Y is clipped to
 * the canonical DRAW AREA — exactly like rt_native — so a primitive that bleeds
 * past the current framebuffer's vertical band is confined to it. This matters
 * for VERTICALLY double-buffered games (e.g. MMX6: front y0..239 / back
 * y240..479, same base_x): the background tile loop starts a partial row at
 * y = buffer_top - (scrollY & 0xf), which without the band clip would draw up
 * into the OTHER buffer's band in the shared wide surface and corrupt it
 * (top/bottom edge flicker as the buffers flip). Canonical VRAM never showed
 * this because its draw-area clip already confines it. Scale matches SSAA. */
static int view_enabled, view_shift, view_pad_left, view_pad_right;
static inline RTarget rt_wide(void) {
    int s = g_scale;
    RTarget t;
    t.buf = g_wide_cur;
    t.w = g_wide_w * s; t.h = VRAM_HEIGHT * s; t.s = s;
    t.cx1 = view_pad_left * s; t.cy1 = g_clip_y1 * s;
    t.cx2 = (g_wide_w - view_pad_right) * s - 1;
    t.cy2 = g_clip_y2 * s + (s - 1);
    return t;
}

/* X-translation (native px) from canonical VRAM space into the active wide
 * surface: local_x = vram_x - base_x + OFFSET. */
static inline int wide_dx(void) { return g_wide_off + view_shift - g_wide_cur_base; }

/* Native-wide 2D-backdrop X-stretch (SW renderer), mirroring the GL
 * wide_set_bd_scale exactly. The far 2D backdrop (sprite-tagged tile grid: sky
 * band + flower field) bypasses the GTE, so in native-wide it stays 4:3-width and
 * leaves a void; this scales its VRAM x about the 4:3 screen centre by
 * g_wide_w/native_w so it fills the widened frame. Gated per prim by
 * psx_ws_prim_in_backdrop() (native-wide + sprite-tagged + background-phase) so
 * Tomba/HUD/3D are untouched. wide_bd_get() resolves the per-prim scale once. */
typedef struct { int on; float scale, center; } WideBd;
static inline WideBd wide_bd_get(void) {
    WideBd b; b.on = 0; b.scale = 1.0f; b.center = 0.0f;
    extern int psx_ws_prim_in_backdrop(void);
    extern int g_ws_bd_stretch_on, g_ws_bd_stretch_pct;
    if (g_ws_bd_stretch_on && g_wide_w > 0 && g_wide_cur && psx_ws_prim_in_backdrop()) {
        int native_w = g_wide_w - 2 * g_wide_off;
        if (native_w > 0) {
            b.on = 1;
            b.scale = g_ws_bd_stretch_pct > 0 ? (float)g_ws_bd_stretch_pct / 100.0f
                                              : (float)g_wide_w / (float)native_w;
            b.center = (float)g_wide_cur_base + (float)native_w / 2.0f;
        }
    }
    return b;
}
static inline int wide_bd_x(const WideBd *b, int x) {
    if (!b->on) return x;
    float v = ((float)x - b->center) * b->scale + b->center;
    return (int)(v < 0.0f ? v - 0.5f : v + 0.5f);   /* round half away from zero */
}
/* Fwd decl: the backdrop-stretch path in sw_draw_textured_rect routes through the
 * scaled rasterizer (defined later) so a widened sprite stretches, not tiles. */
static void raster_textured_rect_scaled(const RTarget *t, int x, int y, int w, int h,
                                        int u0, int v0, int u1, int v1,
                                        uint16_t clut_x, uint16_t clut_y, uint16_t texpage);

/* ------------------------------------------------------------------ */
/* Semi-transparency blending                                         */
/* ------------------------------------------------------------------ */

static inline uint16_t blend_pixels(uint16_t back, uint16_t front, int mode) {
    int br = (back >>  0) & 0x1F;
    int bg = (back >>  5) & 0x1F;
    int bb = (back >> 10) & 0x1F;
    int fr = (front >>  0) & 0x1F;
    int fg = (front >>  5) & 0x1F;
    int fb = (front >> 10) & 0x1F;
    int r, g, b;

    switch (mode) {
    case 0: /* B/2+F/2 */
        r = (br + fr) >> 1;
        g = (bg + fg) >> 1;
        b = (bb + fb) >> 1;
        break;
    case 1: /* B+F */
        r = br + fr; if (r > 31) r = 31;
        g = bg + fg; if (g > 31) g = 31;
        b = bb + fb; if (b > 31) b = 31;
        break;
    case 2: /* B-F */
        r = br - fr; if (r < 0) r = 0;
        g = bg - fg; if (g < 0) g = 0;
        b = bb - fb; if (b < 0) b = 0;
        break;
    case 3: /* B+F/4 */
        r = br + (fr >> 2); if (r > 31) r = 31;
        g = bg + (fg >> 2); if (g > 31) g = 31;
        b = bb + (fb >> 2); if (b > 31) b = 31;
        break;
    default:
        r = fr; g = fg; b = fb;
        break;
    }
    return (uint16_t)(r | (g << 5) | (b << 10));
}

/* ------------------------------------------------------------------ */
/* Pixel write — central functions with mask + semi-trans             */
/* ------------------------------------------------------------------ */

/* Write an opaque (untextured) pixel with semi-transparency if enabled */
static inline void put_opaque(const RTarget *t, int x, int y, uint16_t color) {
    if (x < 0 || x >= t->w || y < 0 || y >= t->h) return;
    if (x < t->cx1 || x > t->cx2 || y < t->cy1 || y > t->cy2) return;

    int idx = y * t->w + x;

    /* Mask bit check: don't overwrite if dest has bit 15 set */
    if (g_mask_check_bit && (t->buf[idx] & 0x8000)) return;

    /* Semi-transparency: for untextured primitives, always blend when flag set */
    if (g_semi_trans_enabled) {
        color = blend_pixels(t->buf[idx], color, g_semi_trans_mode);
    }

    /* Mask bit set: force bit 15 on written pixel */
    if (g_mask_set_bit) color |= 0x8000;

    t->buf[idx] = color;
    /* Canonical VRAM only (hi-res / wide mirrors are present-side). */
    if (t->buf == g_vram)
        gpu_vram_dirty_mark_row((uint32_t)y);
}

/* Write a textured pixel — semi-trans only if texel bit 15 is set */
static inline void put_textured(const RTarget *t, int x, int y, uint16_t texel,
                                int mod_r, int mod_g, int mod_b,
                                int raw_texture) {
    if (x < 0 || x >= t->w || y < 0 || y >= t->h) return;
    if (x < t->cx1 || x > t->cx2 || y < t->cy1 || y > t->cy2) return;

    /* Transparent texel (0x0000) is always skipped */
    if (texel == 0x0000) return;

    int idx = y * t->w + x;

    /* Mask bit check */
    if (g_mask_check_bit && (t->buf[idx] & 0x8000)) return;

    /* Color modulation: multiply texel by vertex color unless raw texture */
    uint16_t color;
    if (!raw_texture) {
        int tr = (texel >>  0) & 0x1F;
        int tg = (texel >>  5) & 0x1F;
        int tb = (texel >> 10) & 0x1F;
        /* PS1 formula: (texel * color * 2) / 256, with 5-bit texel scaled to 8-bit.
         * Simplified: (tr * mod_r) >> 4, clamped to 31 */
        int r = (tr * mod_r) >> 4; if (r > 31) r = 31;
        int g = (tg * mod_g) >> 4; if (g > 31) g = 31;
        int b = (tb * mod_b) >> 4; if (b > 31) b = 31;
        color = (uint16_t)(r | (g << 5) | (b << 10));
    } else {
        color = texel & 0x7FFF;
    }

    /* Semi-transparency: for textured primitives, only blend if texel has bit 15 */
    if (g_semi_trans_enabled && (texel & 0x8000)) {
        color = blend_pixels(t->buf[idx], color, g_semi_trans_mode);
    }

    /* The texel's bit 15 reaches the framebuffer, forced on by mask set
     * (canonical VRAM shows it; framebuffer-feedback copies depend on it). */
    color |= (uint16_t)(texel & 0x8000);
    if (g_mask_set_bit) color |= 0x8000;

    t->buf[idx] = color;
    if (t->buf == g_vram)
        gpu_vram_dirty_mark_row((uint32_t)y);
}

/* ------------------------------------------------------------------ */
/* Texture lookup with texture window                                 */
/* ------------------------------------------------------------------ */

static uint16_t texel_fetch(int u, int v, uint16_t texpage,
                            uint16_t clut_x, uint16_t clut_y) {
    /* Apply texture window:
     * texcoord = (texcoord AND NOT(Mask*8)) OR ((Offset AND Mask)*8)
     * Mask and Offset are in 8-pixel steps. */
    if (g_tw_mask_x | g_tw_mask_y) {
        u = (u & ~(g_tw_mask_x * 8)) | ((g_tw_off_x & g_tw_mask_x) * 8);
        v = (v & ~(g_tw_mask_y * 8)) | ((g_tw_off_y & g_tw_mask_y) * 8);
    }

    int tpx = (texpage & 0xF) * 64;
    int tpy = ((texpage >> 4) & 1) * 256;
    int depth = (texpage >> 7) & 3;

    switch (depth) {
    case 0: { /* 4-bit CLUT */
        int vram_x = tpx + (u / 4);
        int vram_y = tpy + v;
        uint16_t texel_word = vram_get(vram_x, vram_y);
        int shift = (u & 3) * 4;
        int index = (texel_word >> shift) & 0xF;
        return vram_get(clut_x + index, clut_y);
    }
    case 1: { /* 8-bit CLUT */
        int vram_x = tpx + (u / 2);
        int vram_y = tpy + v;
        uint16_t texel_word = vram_get(vram_x, vram_y);
        int shift = (u & 1) * 8;
        int index = (texel_word >> shift) & 0xFF;
        return vram_get(clut_x + index, clut_y);
    }
    case 2:   /* 15-bit direct */
    case 3: {
        return vram_get(tpx + u, tpy + v);
    }
    default:
        return 0;
    }
}

/* Per-prim uv sampling bounds (inclusive), Beetle-PSX model: bilinear
 * neighbours clamp to the prim's own texture rect so they never blend in
 * texels from a neighbouring sprite/tile or empty VRAM. Set by every
 * textured prim entry point; ignored while a texture window is active
 * (texel_fetch applies the window instead, which wraps by design). */
static int g_uv_lim[4] = { 0, 0, 255, 255 };

/* (u, v) is clamped in the 256-texel page of the base texel (bu, bv) before
 * it wraps: a neighbour past u/v=255 must not wrap to 0 and blend the far
 * side of the prim, while a prim that runs past 255 keeps its wrapped texels. */
static uint16_t bl_fetch(int bu, int bv, int u, int v, uint16_t texpage,
                         uint16_t clut_x, uint16_t clut_y) {
    if (!(g_tw_mask_x | g_tw_mask_y)) {
        const int pu = bu & ~0xFF, pv = bv & ~0xFF;
        if (u < pu + g_uv_lim[0]) u = pu + g_uv_lim[0]; else if (u > pu + g_uv_lim[2]) u = pu + g_uv_lim[2];
        if (v < pv + g_uv_lim[1]) v = pv + g_uv_lim[1]; else if (v > pv + g_uv_lim[3]) v = pv + g_uv_lim[3];
    }
    return texel_fetch(u & 0xFF, v & 0xFF, texpage, clut_x, clut_y);
}

/* Bilinear texel sample, in RGB space (after the CLUT lookup — never
 * interpolate palette indices). fu/fv are texel-space coordinates.
 *
 * Beetle-PSX formulation: the NEAREST texel is the base (cutout + STP
 * authority), the neighbours lie toward the sub-texel offset and clamp to
 * g_uv_lim, and each texel's weight is gated by its opacity with the colour
 * renormalised — so prim edges and cutout borders keep their colour instead
 * of dissolving into the transparent (black) neighbour and dropping whole
 * edge columns (the v1 "-0.5 then floor" base sampled one texel OUTSIDE the
 * prim on its top/left edges). Matches the GL TEX shader's bilinear path. */
static uint16_t texel_fetch_bilinear(float fu, float fv, uint16_t texpage,
                                     uint16_t clut_x, uint16_t clut_y) {
    int iu = (int)floorf(fu);
    int iv = (int)floorf(fv);
    int fx = (int)((fu - (float)iu) * 256.0f) - 128;
    int fy = (int)((fv - (float)iv) * 256.0f) - 128;
    int su = fx < 0 ? -1 : 1, sv = fy < 0 ? -1 : 1;
    if (fx < 0) fx = -fx;
    if (fy < 0) fy = -fy;
    if (fx > 128) fx = 128;
    if (fy > 128) fy = 128;

    uint16_t c00 = bl_fetch(iu, iv, iu, iv, texpage, clut_x, clut_y);
    if (c00 == 0) return 0x0000;
    uint16_t c10 = bl_fetch(iu, iv, iu + su, iv, texpage, clut_x, clut_y);
    uint16_t c01 = bl_fetch(iu, iv, iu, iv + sv, texpage, clut_x, clut_y);
    uint16_t c11 = bl_fetch(iu, iv, iu + su, iv + sv, texpage, clut_x, clut_y);

    int w00 = (c00 ? 1 : 0) * (256 - fx) * (256 - fy);
    int w10 = (c10 ? 1 : 0) * fx * (256 - fy);
    int w01 = (c01 ? 1 : 0) * (256 - fx) * fy;
    int w11 = (c11 ? 1 : 0) * fx * fy;
    int opac = w00 + w10 + w01 + w11;

    int r = ((c00 & 0x1F) * w00 + (c10 & 0x1F) * w10
           + (c01 & 0x1F) * w01 + (c11 & 0x1F) * w11) / opac;
    int g = (((c00 >> 5) & 0x1F) * w00 + ((c10 >> 5) & 0x1F) * w10
           + ((c01 >> 5) & 0x1F) * w01 + ((c11 >> 5) & 0x1F) * w11) / opac;
    int b = (((c00 >> 10) & 0x1F) * w00 + ((c10 >> 10) & 0x1F) * w10
           + ((c01 >> 10) & 0x1F) * w01 + ((c11 >> 10) & 0x1F) * w11) / opac;
    int stp = (((c00 >> 15) & 1) * w00 + ((c10 >> 15) & 1) * w10
             + ((c01 >> 15) & 1) * w01 + ((c11 >> 15) & 1) * w11) * 2 >= opac;

    return (uint16_t)(r | (g << 5) | (b << 10) | (stp ? 0x8000 : 0));
}

static inline float bilinear_center_shift_for_target(const RTarget *t) {
    int s = (t && t->s > 0) ? t->s : 1;
    return 0.5f / (float)s - 1.0f / 64.0f;
}

/* uv sampling bounds (see g_uv_lim): the shared PS1 uv model in gpu_uv.h.
 * The software DDA truncates exactly like the PS1, so unlike the GL/VK
 * backends NO mirrored-2D compensation is applied here — these bounds only
 * feed the bilinear-filter neighbour clamp. */
#include "gpu_uv.h"

static void sw_tri_uv_limits(const int *xs, const int *ys,
                             const int *us, const int *vs) {
    if (g_perspective_valid) psx_uv_tri_world_limits(us, vs, g_uv_lim);
    else psx_uv_tri_limits(xs, ys, us, vs, g_uv_lim);
}

/* One scanline of a textured triangle: the unclipped left edge, the span
 * the per-pixel parameter divides by (at least 1) and the edge values of
 * every interpolated quantity. The CPU rasterizers and the Metal batch both
 * start from these float results. Layout matches PsxMetalTexAttr. */
typedef struct {
    int32_t xa, span;
    float ua, ub, va, vb, qa, qb, uqa, uqb, vqa, vqb, ra, rb, ga, gb, ba, bb;
} TexRow;

/* Vertices sorted by y, with uv, 5-bit modulation colours and q. */
typedef struct {
    int x[3], y[3], u[3], v[3], r[3], g[3], b[3];
    float q[3];
} TexVerts;

static void tex_swap(TexVerts *v, int i, int j) {
    int tt;
    tt = v->x[i]; v->x[i] = v->x[j]; v->x[j] = tt;
    tt = v->y[i]; v->y[i] = v->y[j]; v->y[j] = tt;
    tt = v->u[i]; v->u[i] = v->u[j]; v->u[j] = tt;
    tt = v->v[i]; v->v[i] = v->v[j]; v->v[j] = tt;
    tt = v->r[i]; v->r[i] = v->r[j]; v->r[j] = tt;
    tt = v->g[i]; v->g[i] = v->g[j]; v->g[j] = tt;
    tt = v->b[i]; v->b[i] = v->b[j]; v->b[j] = tt;
    float tq = v->q[i]; v->q[i] = v->q[j]; v->q[j] = tq;
}

/* Sorts by y, keeping every attribute in sync. Returns 0 for zero height. */
static int tex_sort(TexVerts *v) {
    if (v->y[0] > v->y[1]) tex_swap(v, 0, 1);
    if (v->y[0] > v->y[2]) tex_swap(v, 0, 2);
    if (v->y[1] > v->y[2]) tex_swap(v, 1, 2);
    return v->y[2] - v->y[0] != 0;
}

/* Row y of a sorted triangle; *xb receives the unclipped right edge. */
static TexRow tex_row(const TexVerts *v, int y, int *xb_out) {
    const int x0 = v->x[0], y0 = v->y[0], x1 = v->x[1], y1 = v->y[1];
    const int x2 = v->x[2], y2 = v->y[2];
    const int u0 = v->u[0], u1 = v->u[1], u2 = v->u[2];
    const int v0 = v->v[0], v1 = v->v[1], v2 = v->v[2];
    const int r0 = v->r[0], r1 = v->r[1], r2 = v->r[2];
    const int g0 = v->g[0], g1 = v->g[1], g2 = v->g[2];
    const int b0 = v->b[0], b1 = v->b[1], b2 = v->b[2];
    const float q0 = v->q[0], q1 = v->q[1], q2 = v->q[2];
    const int dy_total = y2 - y0;
    int second_half = (y >= y1);
    int seg_height = second_half ? (y2 - y1) : (y1 - y0);
    if (seg_height == 0) seg_height = 1;

    float alpha = (float)(y - y0) / (float)dy_total;
    float beta;
    if (second_half)
        beta = (float)(y - y1) / (float)seg_height;
    else
        beta = (float)(y - y0) / (float)seg_height;

    int xa, xb;
    psx_triangle_edges_at_y(x0, y0, x1, y1, x2, y2, y, &xa, &xb);

    float ua = u0 + (float)(u2 - u0) * alpha;
    float va = v0 + (float)(v2 - v0) * alpha;
    float qa = q0 + (q2 - q0) * alpha;
    float uqa = (float)u0 * q0 + ((float)u2 * q2 - (float)u0 * q0) * alpha;
    float vqa = (float)v0 * q0 + ((float)v2 * q2 - (float)v0 * q0) * alpha;
    float ra = r0 + (float)(r2 - r0) * alpha;
    float ga = g0 + (float)(g2 - g0) * alpha;
    float ba = b0 + (float)(b2 - b0) * alpha;

    float ub, vb, qb, uqb, vqb, rb, gb, bb;
    if (second_half) {
        ub = u1 + (float)(u2 - u1) * beta;
        vb = v1 + (float)(v2 - v1) * beta;
        qb = q1 + (q2 - q1) * beta;
        uqb = (float)u1 * q1 + ((float)u2 * q2 - (float)u1 * q1) * beta;
        vqb = (float)v1 * q1 + ((float)v2 * q2 - (float)v1 * q1) * beta;
        rb = r1 + (float)(r2 - r1) * beta;
        gb = g1 + (float)(g2 - g1) * beta;
        bb = b1 + (float)(b2 - b1) * beta;
    } else {
        ub = u0 + (float)(u1 - u0) * beta;
        vb = v0 + (float)(v1 - v0) * beta;
        qb = q0 + (q1 - q0) * beta;
        uqb = (float)u0 * q0 + ((float)u1 * q1 - (float)u0 * q0) * beta;
        vqb = (float)v0 * q0 + ((float)v1 * q1 - (float)v0 * q0) * beta;
        rb = r0 + (float)(r1 - r0) * beta;
        gb = g0 + (float)(g1 - g0) * beta;
        bb = b0 + (float)(b1 - b0) * beta;
    }

    if (xa > xb) {
        int tt = xa; xa = xb; xb = tt;
        float tf;
        tf = ua; ua = ub; ub = tf;
        tf = va; va = vb; vb = tf;
        tf = qa; qa = qb; qb = tf;
        tf = uqa; uqa = uqb; uqb = tf;
        tf = vqa; vqa = vqb; vqb = tf;
        tf = ra; ra = rb; rb = tf;
        tf = ga; ga = gb; gb = tf;
        tf = ba; ba = bb; bb = tf;
    }
    *xb_out = xb;
    const int span = xb - xa;
    return (TexRow){xa, span == 0 ? 1 : span, ua, ub, va, vb, qa, qb, uqa, uqb, vqa, vqb,
                    ra, rb, ga, gb, ba, bb};
}

/* Texture coordinates at pixel x of a row; returns the row parameter. */
static inline float tex_uv(const TexRow *a, int x, int perspective, float *fu, float *fv) {
    float t_val = (float)(x - a->xa) / (float)a->span;
    *fu = a->ua + (a->ub - a->ua) * t_val;
    *fv = a->va + (a->vb - a->va) * t_val;
    if (perspective) {
        float q = a->qa + (a->qb - a->qa) * t_val;
        if (q > 1.0e-12f) {
            *fu = (a->uqa + (a->uqb - a->uqa) * t_val) / q;
            *fv = (a->vqa + (a->vqb - a->vqa) * t_val) / q;
        }
    }
    return t_val;
}

/* The shaded modulation colours at row parameter t_val. */
static inline void tex_shade(const TexRow *a, float t_val, int *mr, int *mg, int *mb) {
    *mr = clamp_i((int)(a->ra + (a->rb - a->ra) * t_val), 0, 31);
    *mg = clamp_i((int)(a->ga + (a->gb - a->ga) * t_val), 0, 31);
    *mb = clamp_i((int)(a->ba + (a->bb - a->ba) * t_val), 0, 31);
}

static void sw_rect_uv_limits(int u0, int v0, int u1, int v1) {
    psx_uv_rect_limits(u0, v0, u1, v1, g_uv_lim);
}

/* ------------------------------------------------------------------ */
/* Public API: init                                                   */
/* ------------------------------------------------------------------ */

void sw_renderer_init(uint16_t *vram) {
    METAL_BATCH_SYNC();
    g_vram = vram;
#ifdef PSX_MACOS_METAL_RASTERIZER
    const char *metal = getenv("PSX_METAL_RASTERIZER");
    g_metal_rasterizer = metal && strcmp(metal, "1") == 0;
    const char *cpu = getenv("PSX_METAL_BATCH_CPU");
    g_metal_batch_cpu = cpu && strcmp(cpu, "1") == 0;
    const char *min_pixels = getenv("PSX_METAL_BATCH_MIN_PIXELS");
    if (min_pixels) g_metal_batch_min_pixels = (uint32_t)strtoul(min_pixels, NULL, 10);
#endif
    g_clip_x1 = 0;
    g_clip_y1 = 0;
    g_clip_x2 = VRAM_WIDTH - 1;
    g_clip_y2 = VRAM_HEIGHT - 1;
    g_offset_x = 0;
    g_offset_y = 0;
    g_semi_trans_enabled = 0;
    g_semi_trans_mode = 0;
    g_mask_set_bit = 0;
    g_mask_check_bit = 0;
    g_tw_mask_x = g_tw_mask_y = 0;
    g_tw_off_x = g_tw_off_y = 0;
    g_mod_r = g_mod_g = g_mod_b = 16; /* neutral = 128/8 = 16 (no modulation) */
    g_raw_texture = 0;
    g_texture_filter = 0;
    g_texture_filter_req = 0;
}

/* Point the rasterizer at another VRAM array without touching any draw
 * state (the OpenGL render thread switches its upload source this way). */
void sw_renderer_rebind_vram(uint16_t *vram) {
    METAL_BATCH_SYNC();
    g_vram = vram;
}

/* ------------------------------------------------------------------ */
/* Supersampling control                                              */
/* ------------------------------------------------------------------ */

void sw_renderer_set_scale(int scale) {
    METAL_BATCH_SYNC();
    if (scale < 1) scale = 1;
    if (scale > SW_MAX_INTERNAL_SCALE) scale = SW_MAX_INTERNAL_SCALE;

    if (g_hr) {
#ifdef PSX_MACOS_METAL_RASTERIZER
        psx_metal_forget_memory(g_hr);
#endif
        free(g_hr);
        g_hr = NULL;
    }
    /* Wide compositor surfaces are scale-sized; drop them so they re-allocate
     * at the new scale on next use (gpu re-configures native-wide after this). */
    wide_free_all();

    g_scale = scale;
    if (scale > 1) {
        g_hr_w = VRAM_WIDTH * scale;
        g_hr_h = VRAM_HEIGHT * scale;
        const size_t hr_bytes = (size_t)g_hr_w * (size_t)g_hr_h * sizeof(uint16_t);
#ifdef __APPLE__
        /* Page-aligned so Metal can use the mirror in place. Every scale
         * makes hr_bytes a multiple of the 16 KiB page. */
        void *hr = NULL;
        g_hr = posix_memalign(&hr, 16384, hr_bytes) == 0 ? (uint16_t *)hr : NULL;
        if (g_hr) memset(g_hr, 0, hr_bytes);
#else
        g_hr = (uint16_t *)calloc(1, hr_bytes);
#endif
        if (!g_hr) {
            /* Allocation failed — fall back to native rendering. */
            g_scale = 1;
            g_hr_w = VRAM_WIDTH;
            g_hr_h = VRAM_HEIGHT;
        }
    } else {
        g_hr_w = VRAM_WIDTH;
        g_hr_h = VRAM_HEIGHT;
    }
}

int sw_renderer_scale(void) { return g_scale; }

/* Faithful authority (netplay): while this rasterizer writes the guest's
 * authoritative VRAM -- the netplay dual-raster SW@1x mirror or the netplay
 * software GPU -- it draws exactly what hardware draws. The PGXP overrides and
 * texture filtering are presentation enhancements; here they would put host-
 * only state (PGXP shadows, which a rollback load drops on one peer only, and
 * each player's Display filter) into pixels the game can read back with GP0
 * C0 (V8:2 samples its own frame), forking guest RAM between peers. */
void sw_set_faithful_authority(int on) {
    METAL_BATCH_SYNC();
    g_faithful_authority = on ? 1 : 0;
    g_texture_filter = g_faithful_authority ? 0 : g_texture_filter_req;
    if (g_faithful_authority) {
        g_precise_valid = 0;
        g_perspective_valid = 0;
    }
}

int sw_faithful_authority(void) { return g_faithful_authority; }

void sw_set_precise_triangle(int enabled,
                             int32_t x0, int32_t y0,
                             int32_t x1, int32_t y1,
                             int32_t x2, int32_t y2) {
    g_precise_valid = (enabled && !g_faithful_authority) ? 1 : 0;
    g_precise_x16[0] = x0; g_precise_y16[0] = y0;
    g_precise_x16[1] = x1; g_precise_y16[1] = y1;
    g_precise_x16[2] = x2; g_precise_y16[2] = y2;
}

void sw_set_perspective_triangle(int enabled,
                                 float q0, float q1, float q2) {
    const int requested = enabled && q0 > 0.0f && q1 > 0.0f && q2 > 0.0f;
    g_perspective_valid = requested && !g_faithful_authority;
    g_perspective_q[0] = q0;
    g_perspective_q[1] = q1;
    g_perspective_q[2] = q2;
    /* Counts requests: under faithful authority the GPU presentation surface
     * still draws them perspective-correct. */
    if (requested) g_perspective_triangles++;
}

uint32_t sw_perspective_triangle_count(void) {
    return g_perspective_triangles;
}

static inline int precise_scaled(int axis, int vertex, int fallback, int scale) {
    if (!g_precise_valid) return fallback * scale;
    int64_t fixed = axis ? g_precise_y16[vertex] : g_precise_x16[vertex];
    int64_t scaled = fixed * scale;
    if (scaled >= 0) return (int)((scaled + 0x8000) >> 16);
    return -(int)((-scaled + 0x8000) >> 16);
}

static inline int precise_wide_x(int vertex, int fallback, int scale,
                                 int dx, const WideBd *bd) {
    if (g_precise_valid)
        return precise_scaled(0, vertex, fallback, scale) + dx * scale;
    return (wide_bd_x(bd, fallback) + dx) * scale;
}

static inline void precise_consumed(void) {
    g_precise_valid = 0;
    g_perspective_valid = 0;
}

void sw_set_texture_filter(int bilinear) {
    g_texture_filter_req = bilinear ? 1 : 0;
    g_texture_filter = g_faithful_authority ? 0 : g_texture_filter_req;
}
int  sw_texture_filter(void) { return g_texture_filter; }

/* ------------------------------------------------------------------ */
/* Draw state setters                                                 */
/* ------------------------------------------------------------------ */

void sw_set_semi_transparency(int enabled, int mode) {
    g_semi_trans_enabled = enabled;
    g_semi_trans_mode = mode & 3;
}

void sw_set_mask_bits(int set_bit, int check_bit) {
    g_mask_set_bit = set_bit;
    g_mask_check_bit = check_bit;
}

void sw_set_texture_window(uint32_t raw) {
    g_tw_mask_x = (uint8_t)(raw & 0x1F);
    g_tw_mask_y = (uint8_t)((raw >> 5) & 0x1F);
    g_tw_off_x  = (uint8_t)((raw >> 10) & 0x1F);
    g_tw_off_y  = (uint8_t)((raw >> 15) & 0x1F);
}

void sw_set_color_modulation(int r, int g, int b, int raw_texture) {
    /* Convert 8-bit vertex color to modulation factor.
     * result_5bit = (texel * (color >> 3)) >> 4
     * At color=128: factor=16, (31*16)>>4 = 31 — neutral.
     * At color=255: factor=31, saturates. At color=0: black. */
    g_mod_r = (uint8_t)(r >> 3);
    g_mod_g = (uint8_t)(g >> 3);
    g_mod_b = (uint8_t)(b >> 3);
    g_raw_texture = raw_texture;
}

/* ------------------------------------------------------------------ */
/* Draw area / offset                                                 */
/* ------------------------------------------------------------------ */

void sw_set_draw_area(int x1, int y1, int x2, int y2) {
    g_clip_x1 = clamp_i(x1, 0, VRAM_WIDTH - 1);
    g_clip_y1 = clamp_i(y1, 0, VRAM_HEIGHT - 1);
    g_clip_x2 = clamp_i(x2, 0, VRAM_WIDTH - 1);
    g_clip_y2 = clamp_i(y2, 0, VRAM_HEIGHT - 1);
}

void sw_get_draw_area(int *x1, int *y1, int *x2, int *y2) {
    *x1 = g_clip_x1; *y1 = g_clip_y1;
    *x2 = g_clip_x2; *y2 = g_clip_y2;
}

void sw_set_draw_offset(int x, int y) {
    g_offset_x = x;
    g_offset_y = y;
}

/* ------------------------------------------------------------------ */
/* Fill rectangle (GP0(02h) — directly in VRAM, no draw offset)       */
/* ------------------------------------------------------------------ */

static void hr_fill_block(int x, int y, int w, int h, uint16_t color) {
    int s = g_scale;
    int x0 = (x & (VRAM_WIDTH - 1)) * s;
    int y0 = (y & (VRAM_HEIGHT - 1)) * s;
    int W = w * s, H = h * s;
    for (int row = 0; row < H; row++) {
        int py = (y0 + row) % g_hr_h;
        uint16_t *dst = g_hr + (size_t)py * g_hr_w;
        for (int col = 0; col < W; col++) {
            int px = (x0 + col) % g_hr_w;
            dst[px] = color;
        }
    }
}

void sw_fill_rect(int x, int y, int w, int h, uint16_t color) {
    METAL_BATCH_SYNC();
    /* Fill rect ignores draw area, mask bits, and semi-transparency.
     * It writes directly to VRAM with coordinates wrapping. */
    int x0 = x & (VRAM_WIDTH - 1);
    int y0 = y & (VRAM_HEIGHT - 1);

    for (int row = 0; row < h; row++) {
        int py = (y0 + row) & (VRAM_HEIGHT - 1);
        for (int col = 0; col < w; col++) {
            int px = (x0 + col) & (VRAM_WIDTH - 1);
            g_vram[py * VRAM_WIDTH + px] = color;
        }
    }
    gpu_vram_dirty_mark_rect(x0, y0, w, h);

    if (g_hr) hr_fill_block(x, y, w, h, color);
}

/* ------------------------------------------------------------------ */
/* Copy rectangle (VRAM -> VRAM)                                      */
/* ------------------------------------------------------------------ */

void sw_copy_rect(int src_x, int src_y, int dst_x, int dst_y, int w, int h) {
    METAL_BATCH_SYNC();
    uint16_t row_buf[VRAM_WIDTH];

    /* Hi-res scratch: one native row's worth of source super-rows. */
    int s = g_scale;
    uint16_t *hr_rows = NULL;
    if (g_hr) hr_rows = (uint16_t *)malloc((size_t)w * s * s * sizeof(uint16_t));

    for (int row = 0; row < h; row++) {
        int sy = (src_y + row) & (VRAM_HEIGHT - 1);
        int dy = (dst_y + row) & (VRAM_HEIGHT - 1);

        for (int col = 0; col < w; col++) {
            int sx = (src_x + col) & (VRAM_WIDTH - 1);
            row_buf[col] = g_vram[sy * VRAM_WIDTH + sx];
        }

        /* Snapshot the hi-res source super-rows for this native row before
         * any destination write (handles overlapping copies per native row,
         * mirroring the native row_buf approach above). */
        if (hr_rows) {
            for (int sr = 0; sr < s; sr++) {
                int hsy = ((sy * s) + sr) % g_hr_h;
                const uint16_t *src = g_hr + (size_t)hsy * g_hr_w;
                for (int col = 0; col < w; col++) {
                    int hsx_base = ((src_x + col) & (VRAM_WIDTH - 1)) * s;
                    for (int sc = 0; sc < s; sc++) {
                        int hsx = (hsx_base + sc) % g_hr_w;
                        hr_rows[(sr * w + col) * s + sc] = src[hsx];
                    }
                }
            }
        }

        for (int col = 0; col < w; col++) {
            int dx = (dst_x + col) & (VRAM_WIDTH - 1);
            uint16_t pix = row_buf[col];
            /* Copy applies mask bit settings */
            if (g_mask_check_bit && (g_vram[dy * VRAM_WIDTH + dx] & 0x8000))
                continue;
            if (g_mask_set_bit) pix |= 0x8000;
            g_vram[dy * VRAM_WIDTH + dx] = pix;

            if (hr_rows) {
                int hdx_base = dx * s;
                for (int sr = 0; sr < s; sr++) {
                    int hdy = ((dy * s) + sr) % g_hr_h;
                    uint16_t *dst = g_hr + (size_t)hdy * g_hr_w;
                    for (int sc = 0; sc < s; sc++) {
                        int hdx = (hdx_base + sc) % g_hr_w;
                        uint16_t hp = hr_rows[(sr * w + col) * s + sc];
                        if (g_mask_set_bit) hp |= 0x8000;
                        dst[hdx] = hp;
                    }
                }
            }
        }
        gpu_vram_dirty_mark_row((uint32_t)dy);
    }

    if (hr_rows) free(hr_rows);
}

/* ------------------------------------------------------------------ */
/* Flat-shaded triangle (scanline rasterization)                      */
/* ------------------------------------------------------------------ */

#ifdef PSX_MACOS_METAL_RASTERIZER
/* Preserve the reference renderer's clipping and dirty-row semantics. */
static PsxMetalSpan metal_span(const RTarget *t, int left, int right) {
    int x = max_i(0, max_i(left, t->cx1));
    int end = min_i(t->w, min_i(right, t->cx2 + 1));
    return end > x ? (PsxMetalSpan){x, end - x} : (PsxMetalSpan){0, 0};
}

static void raster_textured_rect(const RTarget *t, int x, int y, int w, int h,
                                 int u, int v, uint16_t clut_x, uint16_t clut_y,
                                 uint16_t texpage);

/* Queued Metal primitives, one batch per target buffer. */
#define METAL_BATCH_TARGETS 4
#define METAL_BATCH_MAX_ENTRIES (1u << 20)
typedef struct {
    uint16_t *buf;
    int w, h;
    PsxMetalPrimitive *primitives;
    uint32_t primitive_count, primitive_capacity;
    PsxMetalEntry *entries;
    uint32_t entry_count, entry_capacity;
    uint32_t *attrs;               /* row attribute words */
    uint32_t attr_count, attr_capacity;
} MetalBatch;
_Static_assert(sizeof(TexRow) == sizeof(PsxMetalTexAttr), "TexRow matches PsxMetalTexAttr");
static MetalBatch g_metal_batches[METAL_BATCH_TARGETS];
static int g_metal_batch_depth;
static uint64_t g_metal_hazard_flushes, g_metal_self_hazards;

/* Native VRAM tiles (16 wide, 4 tall) the open batches write, and that
 * their textured primitives read. A batch never holds both for one tile:
 * each target is a separate dispatch, so a read could otherwise see a later
 * write. Each 1024-pixel row of tiles is one 64-bit word; four-row tiles
 * keep one-row CLUTs apart from sprites drawn just above or below them. */
#define METAL_TILE_W 16
#define METAL_TILE_H 4
typedef struct { uint64_t bits[VRAM_HEIGHT / METAL_TILE_H]; } MetalTiles;
static MetalTiles g_metal_writes, g_metal_reads;

static void tiles_add_span(MetalTiles *t, int x0, int x1, int y0, int y1) {
    const int first = x0 / METAL_TILE_W, last = x1 / METAL_TILE_W;
    const uint64_t mask = (last == 63 ? ~0ull : (1ull << (last + 1)) - 1) &
                          ~((1ull << first) - 1);
    for (int ty = y0 / METAL_TILE_H; ty <= y1 / METAL_TILE_H; ++ty) t->bits[ty] |= mask;
}

/* Adds native pixels [x, x+w) x [y, y+h), wrapping like vram_get. */
static void tiles_add_rect(MetalTiles *t, int x, int y, int w, int h) {
    if (w <= 0 || h <= 0) return;
    if (w > VRAM_WIDTH) w = VRAM_WIDTH;
    if (h > VRAM_HEIGHT) h = VRAM_HEIGHT;
    x &= VRAM_WIDTH - 1;
    y &= VRAM_HEIGHT - 1;
    const int xw = min_i(w, VRAM_WIDTH - x), yh = min_i(h, VRAM_HEIGHT - y);
    tiles_add_span(t, x, x + xw - 1, y, y + yh - 1);
    if (w > xw) tiles_add_span(t, 0, w - xw - 1, y, y + yh - 1);
    if (h > yh) {
        tiles_add_span(t, x, x + xw - 1, 0, h - yh - 1);
        if (w > xw) tiles_add_span(t, 0, w - xw - 1, 0, h - yh - 1);
    }
}

static int tiles_overlap(const MetalTiles *a, const MetalTiles *b) {
    for (size_t i = 0; i < sizeof a->bits / sizeof a->bits[0]; ++i)
        if (a->bits[i] & b->bits[i]) return 1;
    return 0;
}

static void tiles_merge(MetalTiles *into, const MetalTiles *from) {
    for (size_t i = 0; i < sizeof into->bits / sizeof into->bits[0]; ++i)
        into->bits[i] |= from->bits[i];
}

/* Native VRAM a textured primitive may read: texels u0..u1, v0..v1
 * (inclusive) of its texture page and its CLUT row. A texture window, or a
 * range that wraps past 255, takes the whole page. */
static void tiles_add_texture(MetalTiles *t, uint16_t texpage, uint16_t clut_x,
                              uint16_t clut_y, int u0, int v0, int u1, int v1) {
    const int depth = (texpage >> 7) & 3;
    const int shift = depth == 0 ? 2 : depth == 1 ? 1 : 0;
    const int page_x = (texpage & 0xF) * 64, page_y = ((texpage >> 4) & 1) * 256;
    if (g_tw_mask_x | g_tw_mask_y || u0 < 0 || v0 < 0 || u1 > 255 || v1 > 255 ||
        u1 < u0 || v1 < v0)
        tiles_add_rect(t, page_x, page_y, 256 >> shift, 256);
    else
        tiles_add_rect(t, page_x + (u0 >> shift), page_y + v0,
                       (u1 >> shift) - (u0 >> shift) + 1, v1 - v0 + 1);
    if (depth < 2) tiles_add_rect(t, clut_x, clut_y, depth == 0 ? 16 : 256, 1);
}

/* gpu_sw_renderer's texel_fetch with the primitive's captured window. */
static uint16_t metal_texel(const PsxMetalPrimitive *p, int u, int v) {
    const int mx = p->window & 0xFF, my = (p->window >> 8) & 0xFF;
    if (mx | my) {
        const int ox = (p->window >> 16) & 0xFF, oy = (p->window >> 24) & 0xFF;
        u = (u & ~(mx * 8)) | ((ox & mx) * 8);
        v = (v & ~(my * 8)) | ((oy & my) * 8);
    }
    const int tpx = (p->texpage & 0xF) * 64, tpy = ((p->texpage >> 4) & 1) * 256;
    const int depth = (p->texpage >> 7) & 3;
    const int cx = (int)(p->clut & 0xFFFF), cy = (int)(p->clut >> 16);
    if (depth == 0)
        return vram_get(cx + ((vram_get(tpx + u / 4, tpy + v) >> ((u & 3) * 4)) & 0xF), cy);
    if (depth == 1)
        return vram_get(cx + ((vram_get(tpx + u / 2, tpy + v) >> ((u & 1) * 8)) & 0xFF), cy);
    return vram_get(tpx + u, tpy + v);
}

/* Reference executor for a batch: the same per-pixel order as the kernel.
 * Used when Metal cannot run the batch, or by PSX_METAL_BATCH_CPU=1. */
static void metal_batch_apply_cpu(const MetalBatch *b, unsigned char *dirty) {
    for (uint32_t i = 0; i < b->entry_count; ++i) {
        const PsxMetalEntry e = b->entries[i];
        const PsxMetalPrimitive *p = &b->primitives[e.primitive];
        uint16_t *row = b->buf + (size_t)e.row * b->w;
        for (uint32_t x = e.x; x < e.x + e.width; ++x) {
            const uint16_t back = row[x];
            uint16_t result, texel_bit = 0;
            int semi = 1;
            if (p->kind == PSX_METAL_FLAT) {
                result = (uint16_t)p->color;
            } else if (p->kind == PSX_METAL_GOURAUD) {
                PsxMetalRowAttr attr;
                memcpy(&attr, b->attrs + e.attr, sizeof attr);
                const PsxMetalRowAttr *a = &attr;
                if (a->span > 0) {
                    float tf = (float)((int)x - a->xa) / (float)a->span;
                    int r = a->ra + (int)((float)(a->rb - a->ra) * tf);
                    int g = a->ga + (int)((float)(a->gb - a->ga) * tf);
                    int bl = a->ba + (int)((float)(a->bb - a->ba) * tf);
                    r = clamp_i(r, 0, 31); g = clamp_i(g, 0, 31); bl = clamp_i(bl, 0, 31);
                    result = (uint16_t)(r | (g << 5) | (bl << 10));
                } else {
                    result = (uint16_t)(a->ra | (a->ga << 5) | (a->ba << 10));
                }
            } else {
                const int col = (int)x - p->x, line = (int)e.row - p->y;
                int u, v;
                int mr = (int)(p->color & 0xFF), mg = (int)((p->color >> 8) & 0xFF);
                int mb = (int)((p->color >> 16) & 0xFF);
                if (p->kind == PSX_METAL_TEX_TRIANGLE) {
                    TexRow a;
                    memcpy(&a, b->attrs + e.attr, sizeof a);
                    float fu, fv;
                    const float t_val = tex_uv(&a, (int)x, (p->flags & PSX_METAL_PERSPECTIVE) != 0,
                                               &fu, &fv);
                    u = (int)fu & 0xFF;
                    v = (int)fv & 0xFF;
                    if (p->flags & PSX_METAL_SHADED) tex_shade(&a, t_val, &mr, &mg, &mb);
                } else if (p->kind == PSX_METAL_SPRITE) {
                    u = (p->u + col / p->w) & 0xFF;
                    v = (p->v + line / p->w) & 0xFF;
                } else {
                    u = (p->u + (p->du * col) / p->w) & 0xFF;
                    v = (p->v + (p->dv * line) / p->h) & 0xFF;
                }
                const uint16_t texel = metal_texel(p, u, v);
                if (texel == 0) continue;
                if (p->color >> 24) {
                    result = texel & 0x7FFF;
                } else {
                    const int r = min_i(((texel & 31) * mr) >> 4, 31);
                    const int g = min_i((((texel >> 5) & 31) * mg) >> 4, 31);
                    const int bl = min_i((((texel >> 10) & 31) * mb) >> 4, 31);
                    result = (uint16_t)(r | (g << 5) | (bl << 10));
                }
                semi = (texel & 0x8000) != 0;
                texel_bit = texel & 0x8000;
            }
            if ((p->flags & PSX_METAL_CHECK_MASK) && (back & 0x8000)) continue;
            if (p->blend < PSX_METAL_OPAQUE && semi)
                result = blend_pixels(back, result, (int)p->blend);
            row[x] = result | texel_bit | ((p->flags & PSX_METAL_SET_MASK) ? 0x8000 : 0);
            if (p->flags & PSX_METAL_TRACK_DIRTY) dirty[e.row] = 1;
        }
    }
}

void sw_metal_batch_sync(void) {
    int textured = 0;
    for (int i = 0; i < METAL_BATCH_TARGETS; ++i)
        for (uint32_t k = 0; k < g_metal_batches[i].primitive_count && !textured; ++k)
            textured = g_metal_batches[i].primitives[k].kind != PSX_METAL_FLAT;
    for (int i = 0; i < METAL_BATCH_TARGETS; ++i) {
        MetalBatch *b = &g_metal_batches[i];
        if (!b->entry_count) {
            b->primitive_count = b->attr_count = 0;
            continue;
        }
        unsigned char dirty[VRAM_HEIGHT * SW_MAX_INTERNAL_SCALE];
        memset(dirty, 0, (size_t)b->h);
        uint64_t pixels = 0;
        for (uint32_t k = 0; k < b->entry_count && pixels < g_metal_batch_min_pixels; ++k)
            pixels += b->entries[k].width;
        if (g_metal_batch_cpu || pixels < g_metal_batch_min_pixels ||
            !psx_metal_draw_batch(b->buf, b->w, b->h, textured ? g_vram : NULL,
                                  b->primitives, b->primitive_count,
                                  b->entries, b->entry_count,
                                  b->attrs, b->attr_count, dirty))
            metal_batch_apply_cpu(b, dirty);
        if (b->buf == g_vram)
            for (int y = 0; y < b->h; ++y)
                if (dirty[y]) gpu_vram_dirty_mark_row((uint32_t)y);
        b->primitive_count = b->entry_count = b->attr_count = 0;
    }
    memset(&g_metal_writes, 0, sizeof g_metal_writes);
    memset(&g_metal_reads, 0, sizeof g_metal_reads);
}

void sw_metal_batch_begin(void) {
    if (g_metal_rasterizer) ++g_metal_batch_depth;
}

void sw_metal_batch_end(void) {
    if (g_metal_batch_depth > 0 && --g_metal_batch_depth == 0)
        sw_metal_batch_sync();
}

void sw_metal_batch_stats(uint64_t *hazard_flushes, uint64_t *self_hazards) {
    *hazard_flushes = g_metal_hazard_flushes;
    *self_hazards = g_metal_self_hazards;
}

static int metal_grow(void **items, uint32_t *capacity, uint32_t needed, size_t size) {
    if (needed <= *capacity) return 1;
    uint32_t next = *capacity ? *capacity : 1024u;
    while (next < needed) next *= 2u;
    void *grown = realloc(*items, (size_t)next * size);
    if (!grown) return 0;
    *items = grown;
    *capacity = next;
    return 1;
}

/* Fills the state fields every queued primitive shares. */
static PsxMetalPrimitive metal_primitive(const RTarget *t, uint32_t kind, uint32_t color) {
    PsxMetalPrimitive p;
    memset(&p, 0, sizeof p);
    p.kind = kind;
    p.color = color;
    p.blend = g_semi_trans_enabled ? (uint32_t)g_semi_trans_mode : PSX_METAL_OPAQUE;
    p.flags = (g_mask_check_bit ? PSX_METAL_CHECK_MASK : 0) |
              (g_mask_set_bit ? PSX_METAL_SET_MASK : 0) |
              (t->buf == g_vram && g_psx_vram_dirty_tracking ? PSX_METAL_TRACK_DIRTY : 0);
    return p;
}

/* Queues one primitive's spans on t. reads, when non-NULL, is the native
 * VRAM it samples; attrs, when non-NULL, holds one row attribute of
 * attr_words 32-bit words per span.
 * Returns 0 when it must draw on the CPU instead; any open batch has then
 * been applied. */
static int metal_enqueue(const RTarget *t, const PsxMetalSpan *spans, int top, int height,
                         const PsxMetalPrimitive *primitive, const MetalTiles *reads,
                         const void *attrs, uint32_t attr_words) {
    MetalTiles writes;
    memset(&writes, 0, sizeof writes);
    if (t->buf == g_vram) {
        int left = VRAM_WIDTH, right = 0, first = -1, last = -1;
        for (int y = 0; y < height; ++y)
            if (spans[y].width) {
                if (first < 0) first = y;
                last = y;
                left = min_i(left, (int)spans[y].x);
                right = max_i(right, (int)(spans[y].x + spans[y].width));
            }
        if (first < 0) return 1;   /* clipped away */
        tiles_add_rect(&writes, left, top + first, right - left, last - first + 1);
    }
    if (reads && tiles_overlap(reads, &writes)) {
        /* Its own writes feed its texels: only sequential drawing is exact. */
        ++g_metal_self_hazards;
        sw_metal_batch_sync();
        return 0;
    }
    if ((reads && tiles_overlap(reads, &g_metal_writes)) ||
        tiles_overlap(&writes, &g_metal_reads)) {
        ++g_metal_hazard_flushes;
        sw_metal_batch_sync();
    }

    MetalBatch *b = NULL;
    for (int i = 0; i < METAL_BATCH_TARGETS && !b; ++i)
        if (g_metal_batches[i].buf == t->buf && g_metal_batches[i].w == t->w &&
            g_metal_batches[i].h == t->h)
            b = &g_metal_batches[i];
    for (int i = 0; i < METAL_BATCH_TARGETS && !b; ++i)
        if (!g_metal_batches[i].entry_count && !g_metal_batches[i].primitive_count) {
            b = &g_metal_batches[i];
            b->buf = t->buf;
            b->w = t->w;
            b->h = t->h;
        }
    if (!b || b->entry_count + (uint32_t)height > METAL_BATCH_MAX_ENTRIES ||
        !metal_grow((void **)&b->primitives, &b->primitive_capacity,
                    b->primitive_count + 1u, sizeof *b->primitives) ||
        !metal_grow((void **)&b->entries, &b->entry_capacity,
                    b->entry_count + (uint32_t)height, sizeof *b->entries) ||
        (attrs && !metal_grow((void **)&b->attrs, &b->attr_capacity,
                              b->attr_count + (uint32_t)height * attr_words,
                              sizeof *b->attrs))) {
        sw_metal_batch_sync();
        return 0;
    }
    const uint32_t index = b->primitive_count++;
    b->primitives[index] = *primitive;
    for (int y = 0; y < height; ++y)
        if (spans[y].width) {
            uint32_t attr = 0;
            if (attrs) {
                attr = b->attr_count;
                memcpy(b->attrs + attr, (const uint32_t *)attrs + (size_t)y * attr_words,
                       attr_words * sizeof(uint32_t));
                b->attr_count += attr_words;
            }
            b->entries[b->entry_count++] =
                (PsxMetalEntry){(uint32_t)(top + y), spans[y].x, spans[y].width, index, attr};
        }
    tiles_merge(&g_metal_writes, &writes);
    if (reads) tiles_merge(&g_metal_reads, reads);
    return 1;
}

/* Queues a textured rectangle on t. Returns 0 when it must draw on the CPU. */
static int metal_sprite(const RTarget *t, const PsxMetalPrimitive *p, int x, int y,
                        int w, int h, const MetalTiles *reads) {
    if (w <= 0 || h <= 0) return 1;
    const int top = max_i(0, max_i(y, t->cy1));
    const int bottom = min_i(t->h, min_i(y + h, t->cy2 + 1));
    if (bottom <= top) return 1;
    const PsxMetalSpan span = metal_span(t, x, x + w);
    if (!span.width) return 1;
    PsxMetalSpan spans[VRAM_HEIGHT * SW_MAX_INTERNAL_SCALE];
    for (int row = 0; row < bottom - top; ++row) spans[row] = span;
    PsxMetalPrimitive placed = *p;
    placed.x = x;
    placed.y = y;
    return metal_enqueue(t, spans, top, bottom - top, &placed, reads, NULL, 0);
}

/* Batches a textured rectangle on every target, or returns 0 after applying
 * any open batch so the caller draws it on the CPU. Bilinear filtering and
 * the wide compositor stay on the CPU. */
static int metal_textured_rect(int scaled, int x, int y, int w, int h, int u0, int v0,
                               int u1, int v1, uint16_t clut_x, uint16_t clut_y,
                               uint16_t texpage) {
    if (!g_metal_batch_depth || g_texture_filter || g_wide_cur) return 0;
    if (w <= 0 || h <= 0) return 1;
    MetalTiles reads;
    memset(&reads, 0, sizeof reads);
    if (scaled)   /* u0 + du * col / w stays between u0 and u1 */
        tiles_add_texture(&reads, texpage, clut_x, clut_y, min_i(u0, u1), min_i(v0, v1),
                          max_i(u0, u1), max_i(v0, v1));
    else
        tiles_add_texture(&reads, texpage, clut_x, clut_y, u0, v0, u0 + w - 1, v0 + h - 1);
    RTarget n = rt_native();
    PsxMetalPrimitive p = metal_primitive(&n, scaled ? PSX_METAL_SCALED_SPRITE : PSX_METAL_SPRITE,
        (uint32_t)g_mod_r | (uint32_t)g_mod_g << 8 | (uint32_t)g_mod_b << 16 |
        (uint32_t)(g_raw_texture ? 1 : 0) << 24);
    p.texpage = texpage;
    p.clut = (uint32_t)clut_x | (uint32_t)clut_y << 16;
    p.window = (uint32_t)g_tw_mask_x | (uint32_t)g_tw_mask_y << 8 |
               (uint32_t)g_tw_off_x << 16 | (uint32_t)g_tw_off_y << 24;
    p.u = u0;
    p.v = v0;
    p.w = scaled ? w : 1;
    p.h = h;
    p.du = u1 - u0;
    p.dv = v1 - v0;
    if (!metal_sprite(&n, &p, x, y, w, h, &reads)) return 0;
    if (g_hr) {
        const int s = g_scale;
        RTarget hr = rt_hires();
        PsxMetalPrimitive q = metal_primitive(&hr, p.kind, p.color);
        q.texpage = p.texpage; q.clut = p.clut; q.window = p.window;
        q.u = p.u; q.v = p.v; q.du = p.du; q.dv = p.dv;
        q.w = scaled ? w * s : s;
        q.h = h * s;
        /* The native half is queued; if the mirror cannot be, the native
         * copy is applied and the mirror draws on the CPU, the same order. */
        if (!metal_sprite(&hr, &q, x * s, y * s, w * s, h * s, &reads)) {
            if (scaled)
                raster_textured_rect_scaled(&hr, x * s, y * s, w * s, h * s, u0, v0, u1, v1,
                                            clut_x, clut_y, texpage);
            else
                raster_textured_rect(&hr, x * s, y * s, w * s, h * s, u0, v0,
                                     clut_x, clut_y, texpage);
        }
    }
    return 1;
}

static int metal_draw(const RTarget *t, const PsxMetalSpan *spans,
                      int top, int height, uint16_t color) {
    if (g_metal_batch_depth) {
        const PsxMetalPrimitive p = metal_primitive(t, PSX_METAL_FLAT, color);
        if (metal_enqueue(t, spans, top, height, &p, NULL, NULL, 0)) return 1;
    }
    uint16_t *rows = t->buf + (size_t)top * t->w;
    unsigned char dirty[VRAM_HEIGHT * SW_MAX_INTERNAL_SCALE];
    const int track = t->buf == g_vram && g_psx_vram_dirty_tracking;
    if (track) {
        for (int y = 0; y < height; ++y) {
            const uint16_t *row = rows + (size_t)y * t->w + spans[y].x;
            dirty[y] = 0;
            for (uint32_t x = 0; x < spans[y].width; ++x) {
                if (!g_mask_check_bit || !(row[x] & 0x8000)) {
                    dirty[y] = 1;
                    break;
                }
            }
        }
    }

    int blend = g_semi_trans_enabled ? g_semi_trans_mode : -1;
    if (!psx_metal_draw_spans(rows, t->w, spans, height, color, blend,
                             g_mask_check_bit, g_mask_set_bit))
        return 0;
    if (track)
        for (int y = 0; y < height; ++y)
            if (dirty[y]) gpu_vram_dirty_mark_row((uint32_t)(top + y));
    return 1;
}

static int metal_triangle(const RTarget *t, int x0, int y0, int x1, int y1,
                          int x2, int y2, uint16_t color) {
    int top = max_i(0, max_i(y0, t->cy1));
    int bottom = min_i(t->h, min_i(y2, t->cy2 + 1));
    if (bottom <= top) return 1;
    PsxMetalSpan spans[VRAM_HEIGHT * SW_MAX_INTERNAL_SCALE];
    for (int y = top; y < bottom; ++y) {
        int a, b;
        psx_triangle_edges_at_y(x0, y0, x1, y1, x2, y2, y, &a, &b);
        spans[y - top] = metal_span(t, min_i(a, b), max_i(a, b));
    }
    return metal_draw(t, spans, top, bottom - top, color);
}

static int metal_rect(const RTarget *t, int x, int y, int w, int h, uint16_t color) {
    int top = max_i(0, max_i(y, t->cy1));
    int bottom = min_i(t->h, min_i(y + h, t->cy2 + 1));
    if (bottom <= top) return 1;
    PsxMetalSpan span = metal_span(t, x, x + w);
    if (!span.width) return 1;
    PsxMetalSpan spans[VRAM_HEIGHT * SW_MAX_INTERNAL_SCALE];
    for (int row = 0; row < bottom - top; ++row)
        spans[row] = span;
    return metal_draw(t, spans, top, bottom - top, color);
}
#endif

static void raster_flat_triangle(const RTarget *t,
                                 int x0, int y0, int x1, int y1,
                                 int x2, int y2, uint16_t color) {
    /* Sort vertices by Y coordinate */
    if (y0 > y1) { int tt; tt=x0; x0=x1; x1=tt; tt=y0; y0=y1; y1=tt; }
    if (y0 > y2) { int tt; tt=x0; x0=x2; x2=tt; tt=y0; y0=y2; y2=tt; }
    if (y1 > y2) { int tt; tt=x1; x1=x2; x2=tt; tt=y1; y1=y2; y2=tt; }

    int dy_total = y2 - y0;
    if (dy_total == 0) return;
#ifdef PSX_MACOS_METAL_RASTERIZER
    if (g_metal_rasterizer && metal_triangle(t, x0, y0, x1, y1, x2, y2, color))
        return;
#endif

    for (int y = y0; y < y2; y++) {
        if (y < t->cy1 || y > t->cy2) continue;

        int xa, xb;
        psx_triangle_edges_at_y(x0, y0, x1, y1, x2, y2, y, &xa, &xb);

        if (xa > xb) { int tt = xa; xa = xb; xb = tt; }

        int sx = max_i(xa, t->cx1);
        int ex = min_i(xb, t->cx2 + 1);

        for (int x = sx; x < ex; x++) {
            put_opaque(t, x, y, color);
        }
    }
}

void sw_draw_flat_triangle(int x0, int y0, int x1, int y1,
                           int x2, int y2, uint16_t color) {
    RTarget n = rt_native();
    raster_flat_triangle(&n, x0, y0, x1, y1, x2, y2, color);
    if (g_hr) {
        int s = g_scale;
        RTarget hr = rt_hires();
        raster_flat_triangle(&hr,
            precise_scaled(0,0,x0,s), precise_scaled(1,0,y0,s),
            precise_scaled(0,1,x1,s), precise_scaled(1,1,y1,s),
            precise_scaled(0,2,x2,s), precise_scaled(1,2,y2,s), color);
    }
    if (g_wide_cur) {
        int s = g_scale, dx = wide_dx();
        RTarget wt = rt_wide();
        WideBd bd = wide_bd_get();
        raster_flat_triangle(&wt,
            precise_wide_x(0,x0,s,dx,&bd), precise_scaled(1,0,y0,s),
            precise_wide_x(1,x1,s,dx,&bd), precise_scaled(1,1,y1,s),
            precise_wide_x(2,x2,s,dx,&bd), precise_scaled(1,2,y2,s), color);
    }
    precise_consumed();
}

/* ------------------------------------------------------------------ */
/* Gouraud-shaded triangle (scanline rasterization with color interp) */
/* ------------------------------------------------------------------ */

/* One scanline of a Gouraud triangle with vertices sorted by y: the edge
 * x positions and 5-bit edge colours, ordered so xa <= xb. Shared by the CPU
 * rasterizer and the Metal batch so both start every row from the same
 * float results. */
typedef struct {
    int xa, xb, ra, ga, ba, rb, gb, bb;
} GouraudRow;

static GouraudRow gouraud_row(int x0, int y0, int r0, int g0, int b0,
                              int x1, int y1, int r1, int g1, int b1,
                              int x2, int y2, int r2, int g2, int b2, int y) {
    GouraudRow row;
    int dy_total = y2 - y0;
    int second_half = (y >= y1);
    int seg_height = second_half ? (y2 - y1) : (y1 - y0);
    if (seg_height == 0) seg_height = 1;

    float alpha = (float)(y - y0) / (float)dy_total;
    float beta;
    if (second_half)
        beta = (float)(y - y1) / (float)seg_height;
    else
        beta = (float)(y - y0) / (float)seg_height;

    /* Interpolate X along edges */
    int xa, xb;
    psx_triangle_edges_at_y(x0, y0, x1, y1, x2, y2, y, &xa, &xb);

    /* Interpolate colors along edges (in 5-bit space) */
    int ra = r0 + (int)((float)(r2 - r0) * alpha);
    int ga = g0 + (int)((float)(g2 - g0) * alpha);
    int ba = b0 + (int)((float)(b2 - b0) * alpha);
    int rb, gb, bb;
    if (second_half) {
        rb = r1 + (int)((float)(r2 - r1) * beta);
        gb = g1 + (int)((float)(g2 - g1) * beta);
        bb = b1 + (int)((float)(b2 - b1) * beta);
    } else {
        rb = r0 + (int)((float)(r1 - r0) * beta);
        gb = g0 + (int)((float)(g1 - g0) * beta);
        bb = b0 + (int)((float)(b1 - b0) * beta);
    }

    /* Ensure xa < xb, swap colors too */
    if (xa > xb) {
        int tt;
        tt = xa; xa = xb; xb = tt;
        tt = ra; ra = rb; rb = tt;
        tt = ga; ga = gb; gb = tt;
        tt = ba; ba = bb; bb = tt;
    }
    row.xa = xa; row.xb = xb;
    row.ra = ra; row.ga = ga; row.ba = ba;
    row.rb = rb; row.gb = gb; row.bb = bb;
    return row;
}

/* Sorts vertices by Y, keeping colors in sync, and splits 15-bit colours
 * into 5-bit components. Returns 0 for a zero-height triangle. */
typedef struct {
    int x[3], y[3], r[3], g[3], b[3];
} GouraudVerts;

static int gouraud_sort(GouraudVerts *v, int x0, int y0, uint16_t c0, int x1, int y1,
                        uint16_t c1, int x2, int y2, uint16_t c2) {
    /* Extract 5-bit color components for each vertex */
    int r0 = (c0 >>  0) & 0x1F, g0 = (c0 >>  5) & 0x1F, b0 = (c0 >> 10) & 0x1F;
    int r1 = (c1 >>  0) & 0x1F, g1 = (c1 >>  5) & 0x1F, b1 = (c1 >> 10) & 0x1F;
    int r2 = (c2 >>  0) & 0x1F, g2 = (c2 >>  5) & 0x1F, b2 = (c2 >> 10) & 0x1F;

    /* Sort vertices by Y coordinate, keeping colors in sync */
    if (y0 > y1) {
        int tt; tt=x0; x0=x1; x1=tt; tt=y0; y0=y1; y1=tt;
        tt=r0; r0=r1; r1=tt; tt=g0; g0=g1; g1=tt; tt=b0; b0=b1; b1=tt;
    }
    if (y0 > y2) {
        int tt; tt=x0; x0=x2; x2=tt; tt=y0; y0=y2; y2=tt;
        tt=r0; r0=r2; r2=tt; tt=g0; g0=g2; g2=tt; tt=b0; b0=b2; b2=tt;
    }
    if (y1 > y2) {
        int tt; tt=x1; x1=x2; x2=tt; tt=y1; y1=y2; y2=tt;
        tt=r1; r1=r2; r2=tt; tt=g1; g1=g2; g2=tt; tt=b1; b1=b2; b2=tt;
    }
    v->x[0] = x0; v->y[0] = y0; v->r[0] = r0; v->g[0] = g0; v->b[0] = b0;
    v->x[1] = x1; v->y[1] = y1; v->r[1] = r1; v->g[1] = g1; v->b[1] = b1;
    v->x[2] = x2; v->y[2] = y2; v->r[2] = r2; v->g[2] = g2; v->b[2] = b2;
    return y2 - y0 != 0;
}

static GouraudRow gouraud_row_at(const GouraudVerts *v, int y) {
    return gouraud_row(v->x[0], v->y[0], v->r[0], v->g[0], v->b[0],
                       v->x[1], v->y[1], v->r[1], v->g[1], v->b[1],
                       v->x[2], v->y[2], v->r[2], v->g[2], v->b[2], y);
}

static void raster_gouraud_triangle(const RTarget *t,
                                    int x0, int y0, uint16_t c0,
                                    int x1, int y1, uint16_t c1,
                                    int x2, int y2, uint16_t c2) {
    GouraudVerts v;
    if (!gouraud_sort(&v, x0, y0, c0, x1, y1, c1, x2, y2, c2)) return;

    for (int y = v.y[0]; y < v.y[2]; y++) {
        if (y < t->cy1 || y > t->cy2) continue;
        const GouraudRow row = gouraud_row_at(&v, y);
        const int xa = row.xa, xb = row.xb;
        const int ra = row.ra, ga = row.ga, ba = row.ba;
        const int rb = row.rb, gb = row.gb, bb = row.bb;

        int sx = max_i(xa, t->cx1);
        int ex = min_i(xb, t->cx2 + 1);
        int span = xb - xa;

        for (int x = sx; x < ex; x++) {
            /* Interpolate color across the scanline */
            uint16_t color;
            if (span > 0) {
                float tf = (float)(x - xa) / (float)span;
                int r = ra + (int)((float)(rb - ra) * tf);
                int g = ga + (int)((float)(gb - ga) * tf);
                int b = ba + (int)((float)(bb - ba) * tf);
                if (r < 0) r = 0; if (r > 31) r = 31;
                if (g < 0) g = 0; if (g > 31) g = 31;
                if (b < 0) b = 0; if (b > 31) b = 31;
                color = (uint16_t)(r | (g << 5) | (b << 10));
            } else {
                color = (uint16_t)(ra | (ga << 5) | (ba << 10));
            }
            put_opaque(t, x, y, color);
        }
    }
}

#ifdef PSX_MACOS_METAL_RASTERIZER
/* Queues one Gouraud triangle on t with the CPU rasterizer's rows. */
static int metal_gouraud(const RTarget *t, int x0, int y0, uint16_t c0, int x1, int y1,
                         uint16_t c1, int x2, int y2, uint16_t c2) {
    GouraudVerts v;
    if (!gouraud_sort(&v, x0, y0, c0, x1, y1, c1, x2, y2, c2)) return 1;
    /* raster_gouraud_triangle clips rows to the draw area; put_opaque also
     * drops pixels outside the target. */
    const int top = max_i(v.y[0], max_i(t->cy1, 0));
    const int bottom = min_i(v.y[2], min_i(t->cy2 + 1, t->h));
    if (bottom <= top) return 1;
    static PsxMetalSpan spans[VRAM_HEIGHT * SW_MAX_INTERNAL_SCALE];
    static PsxMetalRowAttr attrs[VRAM_HEIGHT * SW_MAX_INTERNAL_SCALE];
    for (int y = top; y < bottom; ++y) {
        const GouraudRow row = gouraud_row_at(&v, y);
        const int sx = max_i(max_i(row.xa, t->cx1), 0);
        const int ex = min_i(min_i(row.xb, t->cx2 + 1), t->w);
        spans[y - top] = ex > sx ? (PsxMetalSpan){(uint32_t)sx, (uint32_t)(ex - sx)}
                                 : (PsxMetalSpan){0, 0};
        attrs[y - top] = (PsxMetalRowAttr){row.xa, row.xb - row.xa, row.ra, row.ga, row.ba,
                                           row.rb, row.gb, row.bb};
    }
    const PsxMetalPrimitive p = metal_primitive(t, PSX_METAL_GOURAUD, 0);
    return metal_enqueue(t, spans, top, bottom - top, &p, NULL, attrs,
                         sizeof *attrs / sizeof(uint32_t));
}
#endif

void sw_draw_gouraud_triangle(int x0, int y0, uint16_t c0,
                              int x1, int y1, uint16_t c1,
                              int x2, int y2, uint16_t c2) {
#ifdef PSX_MACOS_METAL_RASTERIZER
    if (g_metal_batch_depth && !g_wide_cur) {
        RTarget n = rt_native();
        if (metal_gouraud(&n, x0, y0, c0, x1, y1, c1, x2, y2, c2)) {
            if (g_hr) {
                int s = g_scale;
                RTarget hr = rt_hires();
                const int hx0 = precise_scaled(0,0,x0,s), hy0 = precise_scaled(1,0,y0,s);
                const int hx1 = precise_scaled(0,1,x1,s), hy1 = precise_scaled(1,1,y1,s);
                const int hx2 = precise_scaled(0,2,x2,s), hy2 = precise_scaled(1,2,y2,s);
                /* The native half is queued; if the mirror cannot be, the
                 * batch is applied and the mirror draws on the CPU. */
                if (!metal_gouraud(&hr, hx0, hy0, c0, hx1, hy1, c1, hx2, hy2, c2))
                    raster_gouraud_triangle(&hr, hx0, hy0, c0, hx1, hy1, c1, hx2, hy2, c2);
            }
            precise_consumed();
            return;
        }
    }
#endif
    METAL_BATCH_SYNC();
    RTarget n = rt_native();
    raster_gouraud_triangle(&n, x0, y0, c0, x1, y1, c1, x2, y2, c2);
    if (g_hr) {
        int s = g_scale;
        RTarget hr = rt_hires();
        raster_gouraud_triangle(&hr,
            precise_scaled(0,0,x0,s), precise_scaled(1,0,y0,s), c0,
            precise_scaled(0,1,x1,s), precise_scaled(1,1,y1,s), c1,
            precise_scaled(0,2,x2,s), precise_scaled(1,2,y2,s), c2);
    }
    if (g_wide_cur) {
        int s = g_scale, dx = wide_dx();
        RTarget wt = rt_wide();
        WideBd bd = wide_bd_get();
        raster_gouraud_triangle(&wt,
            precise_wide_x(0,x0,s,dx,&bd), precise_scaled(1,0,y0,s), c0,
            precise_wide_x(1,x1,s,dx,&bd), precise_scaled(1,1,y1,s), c1,
            precise_wide_x(2,x2,s,dx,&bd), precise_scaled(1,2,y2,s), c2);
    }
    precise_consumed();
}

/* ------------------------------------------------------------------ */
/* Textured triangle                                                  */
/* ------------------------------------------------------------------ */

/* Rasterizes a sorted textured triangle. shaded selects per-pixel
 * modulation from the vertex colours; otherwise the current modulation. */
static void raster_tex_triangle(const RTarget *t, const TexVerts *v, int shaded,
                                uint16_t clut_x, uint16_t clut_y, uint16_t texpage,
                                int raw_texture, int perspective) {
    for (int y = v->y[0]; y < v->y[2]; y++) {
        if (y < t->cy1 || y > t->cy2) continue;
        int xb;
        const TexRow row = tex_row(v, y, &xb);
        int sx = max_i(row.xa, t->cx1);
        int ex = min_i(xb, t->cx2 + 1);

        for (int x = sx; x < ex; x++) {
            float fu, fv;
            const float t_val = tex_uv(&row, x, perspective, &fu, &fv);
            int mr = g_mod_r, mg = g_mod_g, mb = g_mod_b;
            if (shaded) tex_shade(&row, t_val, &mr, &mg, &mb);

            uint16_t texel;
            if (g_texture_filter) {
                float sh = bilinear_center_shift_for_target(t);
                texel = texel_fetch_bilinear(fu + sh, fv + sh, texpage, clut_x, clut_y);
            } else {
                texel = texel_fetch((int)fu & 0xFF, (int)fv & 0xFF, texpage, clut_x, clut_y);
            }
            put_textured(t, x, y, texel, mr, mg, mb, raw_texture);
        }
    }
}

static inline void color24_to_mod(uint32_t color, int *r, int *g, int *b) {
    *r = (int)((color >> 0) & 0xFF) >> 3;
    *g = (int)((color >> 8) & 0xFF) >> 3;
    *b = (int)((color >> 16) & 0xFF) >> 3;
}

/* Vertices of a textured triangle on a target; coordinates are already in
 * the target's pixel space. */
static TexVerts tex_verts(const int *xs, const int *ys, const int *us, const int *vs,
                          const uint32_t *colors) {
    TexVerts v;
    for (int i = 0; i < 3; ++i) {
        v.x[i] = xs[i]; v.y[i] = ys[i]; v.u[i] = us[i]; v.v[i] = vs[i];
        v.q[i] = g_perspective_q[i];
        if (colors) color24_to_mod(colors[i], &v.r[i], &v.g[i], &v.b[i]);
        else v.r[i] = v.g[i] = v.b[i] = 0;
    }
    return v;
}

#ifdef PSX_MACOS_METAL_RASTERIZER
/* Queues one sorted textured triangle on t with the CPU rasterizer's rows. */
static int metal_tex_triangle(const RTarget *t, const TexVerts *v, const PsxMetalPrimitive *p,
                              const MetalTiles *reads) {
    const int top = max_i(v->y[0], max_i(t->cy1, 0));
    const int bottom = min_i(v->y[2], min_i(t->cy2 + 1, t->h));
    if (bottom <= top) return 1;
    static PsxMetalSpan spans[VRAM_HEIGHT * SW_MAX_INTERNAL_SCALE];
    static TexRow rows[VRAM_HEIGHT * SW_MAX_INTERNAL_SCALE];
    for (int y = top; y < bottom; ++y) {
        int xb;
        rows[y - top] = tex_row(v, y, &xb);
        const int sx = max_i(max_i(rows[y - top].xa, t->cx1), 0);
        const int ex = min_i(min_i(xb, t->cx2 + 1), t->w);
        spans[y - top] = ex > sx ? (PsxMetalSpan){(uint32_t)sx, (uint32_t)(ex - sx)}
                                 : (PsxMetalSpan){0, 0};
    }
    PsxMetalPrimitive placed = *p;
    PsxMetalPrimitive q = metal_primitive(t, PSX_METAL_TEX_TRIANGLE, p->color);
    placed.flags = q.flags | (p->flags & (PSX_METAL_PERSPECTIVE | PSX_METAL_SHADED));
    return metal_enqueue(t, spans, top, bottom - top, &placed, reads, rows,
                         sizeof *rows / sizeof(uint32_t));
}

/* Batches a textured triangle on the native target and the mirror, or
 * returns 0 after applying any open batch so the caller draws it on the CPU.
 * Bilinear filtering and the wide compositor stay on the CPU. */
static int metal_textured_triangle(const int *xs, const int *ys, const int *us, const int *vs,
                                   const uint32_t *colors, uint16_t clut_x, uint16_t clut_y,
                                   uint16_t texpage, int raw_texture) {
    if (!g_metal_batch_depth || g_texture_filter || g_wide_cur) return 0;
    /* Pixels interpolate uv inside the vertices' range; float rounding can
     * truncate one texel below it. sw_set_perspective_triangle accepts only
     * positive q, so perspective division keeps the same bounds. */
    MetalTiles reads;
    memset(&reads, 0, sizeof reads);
    const int u0 = min_i(us[0], min_i(us[1], us[2])), u1 = max_i(us[0], max_i(us[1], us[2]));
    const int v0 = min_i(vs[0], min_i(vs[1], vs[2])), v1 = max_i(vs[0], max_i(vs[1], vs[2]));
    tiles_add_texture(&reads, texpage, clut_x, clut_y, max_i(u0 - 1, 0), max_i(v0 - 1, 0),
                      min_i(u1 + 1, 255), min_i(v1 + 1, 255));
    RTarget n = rt_native();
    PsxMetalPrimitive p = metal_primitive(&n, PSX_METAL_TEX_TRIANGLE,
        colors ? (uint32_t)(raw_texture ? 1 : 0) << 24
               : (uint32_t)g_mod_r | (uint32_t)g_mod_g << 8 | (uint32_t)g_mod_b << 16 |
                 (uint32_t)(raw_texture ? 1 : 0) << 24);
    p.texpage = texpage;
    p.clut = (uint32_t)clut_x | (uint32_t)clut_y << 16;
    p.window = (uint32_t)g_tw_mask_x | (uint32_t)g_tw_mask_y << 8 |
               (uint32_t)g_tw_off_x << 16 | (uint32_t)g_tw_off_y << 24;
    p.flags = (g_perspective_valid ? PSX_METAL_PERSPECTIVE : 0) |
              (colors ? PSX_METAL_SHADED : 0);
    TexVerts v = tex_verts(xs, ys, us, vs, colors);
    if (tex_sort(&v) && !metal_tex_triangle(&n, &v, &p, &reads)) return 0;
    if (g_hr) {
        const int s = g_scale;
        RTarget hr = rt_hires();
        int hx[3], hy[3];
        for (int i = 0; i < 3; ++i) {
            hx[i] = precise_scaled(0, i, xs[i], s);
            hy[i] = precise_scaled(1, i, ys[i], s);
        }
        TexVerts h = tex_verts(hx, hy, us, vs, colors);
        /* The native half is queued; if the mirror cannot be, the batch is
         * applied and the mirror draws on the CPU, the same order. */
        if (tex_sort(&h) && !metal_tex_triangle(&hr, &h, &p, &reads))
            raster_tex_triangle(&hr, &h, colors != NULL, clut_x, clut_y, texpage, raw_texture,
                                g_perspective_valid);
    }
    return 1;
}
#endif

/* Draws a textured triangle on every target; colors, when non-NULL, are the
 * three 24-bit vertex colours of a shaded triangle. */
static void draw_textured_triangle(const int *xs, const int *ys, const int *us, const int *vs,
                                   const uint32_t *colors, uint16_t clut_x, uint16_t clut_y,
                                   uint16_t texpage, int raw_texture) {
    int64_t area2 = (int64_t)(xs[1] - xs[0]) * (int64_t)(ys[2] - ys[0])
                  - (int64_t)(xs[2] - xs[0]) * (int64_t)(ys[1] - ys[0]);
    if (area2 == 0) { precise_consumed(); return; }

#ifdef PSX_MACOS_METAL_RASTERIZER
    if (metal_textured_triangle(xs, ys, us, vs, colors, clut_x, clut_y, texpage, raw_texture)) {
        precise_consumed();
        return;
    }
#endif
    METAL_BATCH_SYNC();
    if (g_texture_filter) sw_tri_uv_limits(xs, ys, us, vs);

    const int shaded = colors != NULL;
    TexVerts v = tex_verts(xs, ys, us, vs, colors);
    RTarget n = rt_native();
    if (tex_sort(&v))
        raster_tex_triangle(&n, &v, shaded, clut_x, clut_y, texpage, raw_texture,
                            g_perspective_valid);
    if (g_hr) {
        int s = g_scale, hx[3], hy[3];
        for (int i = 0; i < 3; ++i) {
            hx[i] = precise_scaled(0, i, xs[i], s);
            hy[i] = precise_scaled(1, i, ys[i], s);
        }
        RTarget hr = rt_hires();
        TexVerts h = tex_verts(hx, hy, us, vs, colors);
        if (tex_sort(&h))
            raster_tex_triangle(&hr, &h, shaded, clut_x, clut_y, texpage, raw_texture,
                                g_perspective_valid);
    }
    if (g_wide_cur) {
        int s = g_scale, dx = wide_dx(), wx[3], wy[3];
        RTarget wt = rt_wide();
        WideBd bd = wide_bd_get();
        for (int i = 0; i < 3; ++i) {
            wx[i] = precise_wide_x(i, xs[i], s, dx, &bd);
            wy[i] = precise_scaled(1, i, ys[i], s);
        }
        TexVerts w = tex_verts(wx, wy, us, vs, colors);
        if (tex_sort(&w))
            raster_tex_triangle(&wt, &w, shaded, clut_x, clut_y, texpage, raw_texture,
                                g_perspective_valid);
    }
    precise_consumed();
}

void sw_draw_textured_triangle(int x0, int y0, int u0, int v0,
                               int x1, int y1, int u1, int v1,
                               int x2, int y2, int u2, int v2,
                               uint16_t clut_x, uint16_t clut_y,
                               uint16_t texpage) {
    const int xs[3] = {x0, x1, x2}, ys[3] = {y0, y1, y2};
    const int us[3] = {u0, u1, u2}, vs[3] = {v0, v1, v2};
    draw_textured_triangle(xs, ys, us, vs, NULL, clut_x, clut_y, texpage, g_raw_texture);
}

void sw_draw_shaded_textured_triangle(int x0, int y0, int u0, int v0,
                                      uint32_t color0,
                                      int x1, int y1, int u1, int v1,
                                      uint32_t color1,
                                      int x2, int y2, int u2, int v2,
                                      uint32_t color2,
                                      uint16_t clut_x, uint16_t clut_y,
                                      uint16_t texpage, int raw_texture) {
    const int xs[3] = {x0, x1, x2}, ys[3] = {y0, y1, y2};
    const int us[3] = {u0, u1, u2}, vs[3] = {v0, v1, v2};
    const uint32_t colors[3] = {color0, color1, color2};
    draw_textured_triangle(xs, ys, us, vs, colors, clut_x, clut_y, texpage, raw_texture);
}

/* ------------------------------------------------------------------ */
/* Flat rectangle                                                     */
/* ------------------------------------------------------------------ */

static void raster_flat_rect(const RTarget *t, int x, int y, int w, int h,
                             uint16_t color) {
#ifdef PSX_MACOS_METAL_RASTERIZER
    if (g_metal_rasterizer && metal_rect(t, x, y, w, h, color)) return;
#endif
    for (int row = 0; row < h; row++) {
        int py = y + row;
        if (py < t->cy1 || py > t->cy2) continue;

        int sx = max_i(x, t->cx1);
        int ex = min_i(x + w - 1, t->cx2);

        for (int px = sx; px <= ex; px++) {
            put_opaque(t, px, py, color);
        }
    }
}

void sw_draw_flat_rect(int x, int y, int w, int h, uint16_t color) {
    RTarget n = rt_native();
    raster_flat_rect(&n, x, y, w, h, color);
    if (g_hr) {
        int s = g_scale;
        RTarget hr = rt_hires();
        raster_flat_rect(&hr, x*s, y*s, w*s, h*s, color);
    }
    if (g_wide_cur) {
        int s = g_scale, dx = wide_dx();
        RTarget wt = rt_wide();
        /* Full-screen 2D overlay (pause gray-filter / load fade): a flat rect
         * spanning the whole 4:3 framebuffer must cover the whole wide surface
         * too, else the revealed 16:9 margins are left undimmed/unfaded. Detect
         * a rect whose native screen-X span (x - base) covers [0, native_w) and
         * extend its wide pass across the full surface; every other rect mirrors
         * 1:1 as before. native_w = the 4:3 framebuffer width (g_wide_w less the
         * per-side reveal on both sides); g_wide_cur_base is its VRAM left edge.
         * Only runs in native-wide (g_wide_cur != NULL), so 4:3 is unaffected. */
        int native_w = g_wide_w - 2 * g_wide_off;
        int lx = x - g_wide_cur_base, rx = x + w - g_wide_cur_base;
        WideBd bd = wide_bd_get();
        if (native_w > 0 && lx <= 0 && rx >= native_w)
            raster_flat_rect(&wt, 0, y*s, g_wide_w*s, h*s, color);
        else if (bd.on) {
            int xl = wide_bd_x(&bd, x), xr = wide_bd_x(&bd, x + w);
            raster_flat_rect(&wt, (xl+dx)*s, y*s, (xr-xl)*s, h*s, color);
        } else
            raster_flat_rect(&wt, (x+dx)*s, y*s, w*s, h*s, color);
    }
}

/* ------------------------------------------------------------------ */
/* Textured rectangle                                                 */
/*                                                                    */
/* Hi-res variant samples one native texel per native-pixel footprint */
/* (target coord / scale), keeping textures at native resolution while */
/* the rectangle's footprint is rendered at the higher resolution.    */
/* ------------------------------------------------------------------ */

static void raster_textured_rect(const RTarget *t, int x, int y, int w, int h,
                                 int u, int v,
                                 uint16_t clut_x, uint16_t clut_y,
                                 uint16_t texpage) {
    int s = t->s;
    for (int row = 0; row < h; row++) {
        int py = y + row;
        if (py < t->cy1 || py > t->cy2) continue;
        int tv = (v + row / s) & 0xFF;
        float fv = (float)v + (float)row / (float)s;

        for (int col = 0; col < w; col++) {
            int px = x + col;
            if (px < t->cx1 || px > t->cx2) continue;

            uint16_t texel;
            if (g_texture_filter) {
                float fu = (float)u + (float)col / (float)s;
                float sh = bilinear_center_shift_for_target(t);
                texel = texel_fetch_bilinear(fu + sh, fv + sh, texpage, clut_x, clut_y);
            } else {
                int tu = (u + col / s) & 0xFF;
                texel = texel_fetch(tu, tv, texpage, clut_x, clut_y);
            }
            put_textured(t, px, py, texel, g_mod_r, g_mod_g, g_mod_b, g_raw_texture);
        }
    }
}

void sw_draw_textured_rect(int x, int y, int w, int h,
                           int u, int v,
                           uint16_t clut_x, uint16_t clut_y,
                           uint16_t texpage) {
#ifdef PSX_MACOS_METAL_RASTERIZER
    if (metal_textured_rect(0, x, y, w, h, u, v, 0, 0, clut_x, clut_y, texpage)) return;
#endif
    METAL_BATCH_SYNC();
    if (g_texture_filter) sw_rect_uv_limits(u, v, u + w, v + h);
    RTarget n = rt_native();
    raster_textured_rect(&n, x, y, w, h, u, v, clut_x, clut_y, texpage);
    if (g_hr) {
        int s = g_scale;
        RTarget hr = rt_hires();
        raster_textured_rect(&hr, x*s, y*s, w*s, h*s, u, v, clut_x, clut_y, texpage);
    }
    if (g_wide_cur) {
        int s = g_scale, dx = wide_dx();
        RTarget wt = rt_wide();
        WideBd bd = wide_bd_get();
        if (bd.on) {
            /* Stretch about screen centre: widen the destination span and map the
             * native texel footprint across it via the SCALED rasterizer (the
             * 1:1 sampler would TILE a widened rect instead of stretching it). */
            int xl = wide_bd_x(&bd, x), xr = wide_bd_x(&bd, x + w);
            raster_textured_rect_scaled(&wt, (xl+dx)*s, y*s, (xr-xl)*s, h*s,
                                        u, v, u + w, v + h, clut_x, clut_y, texpage);
        } else
            raster_textured_rect(&wt, (x+dx)*s, y*s, w*s, h*s, u, v, clut_x, clut_y, texpage);
    }
}

/* ---- MMX6 native-wide reveal-column tile (host-side, WIDE SURFACE ONLY) --------
 * Rasterize ONE 16x16 textured tile into the active wide surface for a given vertical
 * band, leaving canonical VRAM AND the hi-res mirror byte-identical. This fills the
 * 16:9 reveal margins with background columns the guest BG renderer (kept at its native
 * 21-column window) never draws — so the guest packet buffer / OT / 999-tile cap and the
 * 4:3 image are completely untouched (no guest-RAM mutation, elective, 4:3 identical).
 *
 * `x`,`y` are guest screen coords; `y` already includes the band offset. The clip is the
 * EXPLICIT band [band_y, band_y+240) — not g_clip — because this runs at the per-frame
 * fill/clear, before the guest applies E3/E4/E5 for the new frame. Texture state globals
 * are set transiently; the next real GP0 draw's setup overwrites them. */
void sw_wide_emit_tile(int band_y, int x, int y, int u, int v,
                       uint16_t clut_x, uint16_t clut_y, uint16_t texpage,
                       uint32_t color24, int semi_trans, int semi_mode, int raw_texture) {
    METAL_BATCH_SYNC();
    if (!g_wide_cur) return;
    int s = g_scale, dx = wide_dx();
    RTarget t;
    t.buf = g_wide_cur; t.w = g_wide_w * s; t.h = VRAM_HEIGHT * s; t.s = s;
    t.cx1 = 0;            t.cx2 = g_wide_w * s - 1;
    t.cy1 = band_y * s;   t.cy2 = (band_y + 240) * s - 1;
    g_mod_r = (uint8_t)((color24 & 0xff) >> 3);
    g_mod_g = (uint8_t)(((color24 >> 8) & 0xff) >> 3);
    g_mod_b = (uint8_t)(((color24 >> 16) & 0xff) >> 3);
    g_raw_texture        = raw_texture ? 1 : 0;
    g_semi_trans_enabled = semi_trans ? 1 : 0;
    g_semi_trans_mode    = semi_mode & 3;
    raster_textured_rect(&t, (x + dx) * s, y * s, 16 * s, 16 * s,
                         u, v, clut_x, clut_y, texpage);
}

static void raster_textured_rect_scaled(const RTarget *t, int x, int y,
                                        int w, int h,
                                        int u0, int v0, int u1, int v1,
                                        uint16_t clut_x, uint16_t clut_y,
                                        uint16_t texpage) {
    if (w <= 0 || h <= 0) return;

    int du = u1 - u0;
    int dv = v1 - v0;

    for (int row = 0; row < h; row++) {
        int py = y + row;
        if (py < t->cy1 || py > t->cy2) continue;

        int tv = (int)(v0 + ((int64_t)dv * row) / h) & 0xFF;
        float fv = (float)v0 + (float)dv * (float)row / (float)h;
        for (int col = 0; col < w; col++) {
            int px = x + col;
            if (px < t->cx1 || px > t->cx2) continue;

            uint16_t texel;
            if (g_texture_filter) {
                float fu = (float)u0 + (float)du * (float)col / (float)w;
                float sh = bilinear_center_shift_for_target(t);
                texel = texel_fetch_bilinear(fu + sh, fv + sh, texpage, clut_x, clut_y);
            } else {
                int tu = (int)(u0 + ((int64_t)du * col) / w) & 0xFF;
                texel = texel_fetch(tu, tv, texpage, clut_x, clut_y);
            }
            put_textured(t, px, py, texel, g_mod_r, g_mod_g, g_mod_b, g_raw_texture);
        }
    }
}

void sw_draw_textured_rect_scaled(int x, int y, int w, int h,
                                  int u0, int v0, int u1, int v1,
                                  uint16_t clut_x, uint16_t clut_y,
                                  uint16_t texpage) {
#ifdef PSX_MACOS_METAL_RASTERIZER
    if (metal_textured_rect(1, x, y, w, h, u0, v0, u1, v1, clut_x, clut_y, texpage)) return;
#endif
    METAL_BATCH_SYNC();
    if (w <= 0 || h <= 0) return;
    if (g_texture_filter) sw_rect_uv_limits(u0, v0, u1, v1);
    RTarget n = rt_native();
    raster_textured_rect_scaled(&n, x, y, w, h, u0, v0, u1, v1,
                                clut_x, clut_y, texpage);
    if (g_hr) {
        int s = g_scale;
        RTarget hr = rt_hires();
        /* Footprint scales by s; the destination span widens so the texel
         * step stays per-native-pixel. */
        raster_textured_rect_scaled(&hr, x*s, y*s, w*s, h*s, u0, v0, u1, v1,
                                    clut_x, clut_y, texpage);
    }
    if (g_wide_cur) {
        int s = g_scale, dx = wide_dx();
        RTarget wt = rt_wide();
        WideBd bd = wide_bd_get();
        int xl = wide_bd_x(&bd, x), xr = wide_bd_x(&bd, x + w);
        raster_textured_rect_scaled(&wt, (xl+dx)*s, y*s, (xr-xl)*s, h*s, u0, v0, u1, v1,
                                    clut_x, clut_y, texpage);
    }
}

/* ------------------------------------------------------------------ */
/* Line (Bresenham)                                                   */
/*                                                                    */
/* Lines are 1px on PS1. In the hi-res mirror each visited native     */
/* pixel is replicated as an s*s block so line thickness survives     */
/* downsampling (re-rasterizing a 1-hires-pixel line would vanish).   */
/* ------------------------------------------------------------------ */

static inline void hr_put_block_opaque(int nx, int ny, uint16_t color) {
    int s = g_scale;
    RTarget hr = rt_hires();
    int bx = nx * s, by = ny * s;
    for (int dy = 0; dy < s; dy++)
        for (int dx = 0; dx < s; dx++)
            put_opaque(&hr, bx + dx, by + dy, color);
}

/* Mirror a 1px line pixel into the active wide surface (s*s block, x-translated).
 * Self-gates on g_wide_cur; runs at any scale. */
static inline void wide_put_block_opaque(int nx, int ny, uint16_t color) {
    if (!g_wide_cur) return;
    int s = g_scale, tdx = wide_dx();
    RTarget wt = rt_wide();
    int bx = (nx + tdx) * s, by = ny * s;
    for (int dy = 0; dy < s; dy++)
        for (int dx = 0; dx < s; dx++)
            put_opaque(&wt, bx + dx, by + dy, color);
}

void sw_draw_line(int x0, int y0, int x1, int y1, uint16_t color) {
    METAL_BATCH_SYNC();
    RTarget n = rt_native();
    int dx = abs(x1 - x0);
    int dy = -abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx + dy;

    for (;;) {
        put_opaque(&n, x0, y0, color);
        if (g_hr) hr_put_block_opaque(x0, y0, color);
        wide_put_block_opaque(x0, y0, color);

        if (x0 == x1 && y0 == y1) break;

        int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

void sw_draw_shaded_line(int x0, int y0, uint16_t c0,
                         int x1, int y1, uint16_t c1) {
    METAL_BATCH_SYNC();
    RTarget n = rt_native();
    int dx = abs(x1 - x0);
    int dy = -abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx + dy;

    /* Total steps for interpolation */
    int total = max_i(abs(x1 - x0), abs(y1 - y0));
    if (total == 0) total = 1;

    /* Extract 5-bit color channels */
    int r0 = c0 & 0x1F, g0 = (c0 >> 5) & 0x1F, b0 = (c0 >> 10) & 0x1F;
    int r1 = c1 & 0x1F, g1 = (c1 >> 5) & 0x1F, b1 = (c1 >> 10) & 0x1F;

    int step = 0;
    for (;;) {
        int r = r0 + (r1 - r0) * step / total;
        int g = g0 + (g1 - g0) * step / total;
        int b = b0 + (b1 - b0) * step / total;
        uint16_t color = (uint16_t)(r | (g << 5) | (b << 10));
        put_opaque(&n, x0, y0, color);
        if (g_hr) hr_put_block_opaque(x0, y0, color);
        wide_put_block_opaque(x0, y0, color);

        if (x0 == x1 && y0 == y1) break;

        int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
        step++;
    }
}

/* ------------------------------------------------------------------ */
/* VRAM pixel access                                                  */
/* ------------------------------------------------------------------ */

void sw_vram_write(int x, int y, uint16_t pixel) {
    METAL_BATCH_SYNC();
    x &= (VRAM_WIDTH - 1);
    y &= (VRAM_HEIGHT - 1);
    g_vram[y * VRAM_WIDTH + x] = pixel;
    gpu_vram_dirty_mark_row((uint32_t)y);

    if (g_hr) {
        int s = g_scale;
        int bx = x * s, by = y * s;
        for (int dy = 0; dy < s; dy++) {
            uint16_t *dst = g_hr + (size_t)(by + dy) * g_hr_w + bx;
            for (int dx = 0; dx < s; dx++) dst[dx] = pixel;
        }
    }
}

uint16_t sw_vram_read(int x, int y) {
    METAL_BATCH_SYNC();
    return vram_get(x, y);
}

/* ------------------------------------------------------------------ */
/* Bulk VRAM transfers                                                */
/* ------------------------------------------------------------------ */

/* Rebuild the supersampled mirror from canonical VRAM after a bulk replace.
 * Row-wise replication (memcpy) beats the per-texel nested loop used by the
 * general transfer path — savestate restore hits full 1024×512 often. */
static void hr_rebuild_from_vram(void) {
    if (!g_hr || !g_vram) return;
    const int s = g_scale;
    if (s <= 1) return;
    uint16_t *row = (uint16_t *)malloc((size_t)g_hr_w * sizeof(uint16_t));
    if (!row) {
        /* Fall back to slow per-pixel replication if OOM (should not happen). */
        for (int py = 0; py < VRAM_HEIGHT; py++) {
            for (int px = 0; px < VRAM_WIDTH; px++) {
                uint16_t pixel = g_vram[py * VRAM_WIDTH + px];
                int bx = px * s, by = py * s;
                for (int dy = 0; dy < s; dy++) {
                    uint16_t *dst = g_hr + (size_t)(by + dy) * (size_t)g_hr_w;
                    for (int dx = 0; dx < s; dx++)
                        dst[bx + dx] = pixel;
                }
            }
        }
        return;
    }
    for (int y = 0; y < VRAM_HEIGHT; y++) {
        const uint16_t *src = g_vram + (size_t)y * VRAM_WIDTH;
        for (int x = 0; x < VRAM_WIDTH; x++) {
            uint16_t p = src[x];
            uint16_t *d = row + (size_t)x * (size_t)s;
            for (int dx = 0; dx < s; dx++)
                d[dx] = p;
        }
        for (int dy = 0; dy < s; dy++) {
            memcpy(g_hr + ((size_t)(y * s + dy) * (size_t)g_hr_w),
                   row, (size_t)g_hr_w * sizeof(uint16_t));
        }
    }
    free(row);
}

void sw_vram_transfer_in(int x, int y, int w, int h, const uint16_t *data) {
    METAL_BATCH_SYNC();
    if (!data || !g_vram || w <= 0 || h <= 0) return;

    /* Full-VRAM replace (savestate / boot_state): memcpy + one HR rebuild.
     * The general path is a wrapped per-pixel loop that, with supersampling,
     * does s² stores per guest texel — multi-second hitches on 2×/4×. */
    if (x == 0 && y == 0 && w == VRAM_WIDTH && h == VRAM_HEIGHT) {
        memcpy(g_vram, data, (size_t)VRAM_WIDTH * (size_t)VRAM_HEIGHT * sizeof(uint16_t));
        gpu_vram_dirty_mark_all();
        if (g_hr) hr_rebuild_from_vram();
        return;
    }

    /* Contiguous non-wrapping rect, no HR: memcpy each row. */
    x &= (VRAM_WIDTH - 1);
    y &= (VRAM_HEIGHT - 1);
    if (!g_hr && x + w <= VRAM_WIDTH && y + h <= VRAM_HEIGHT) {
        for (int row = 0; row < h; row++) {
            memcpy(g_vram + (size_t)(y + row) * VRAM_WIDTH + (size_t)x,
                   data + (size_t)row * (size_t)w,
                   (size_t)w * sizeof(uint16_t));
        }
        gpu_vram_dirty_mark_rect(x, y, w, h);
        return;
    }

    int idx = 0;
    int s = g_scale;
    for (int row = 0; row < h; row++) {
        int py = (y + row) & (VRAM_HEIGHT - 1);
        for (int col = 0; col < w; col++) {
            int px = (x + col) & (VRAM_WIDTH - 1);
            uint16_t pixel = data[idx++];
            g_vram[py * VRAM_WIDTH + px] = pixel;

            if (g_hr) {
                int bx = px * s, by = py * s;
                for (int dy = 0; dy < s; dy++) {
                    uint16_t *dst = g_hr + (size_t)((by + dy) % g_hr_h) * g_hr_w;
                    for (int dx = 0; dx < s; dx++)
                        dst[(bx + dx) % g_hr_w] = pixel;
                }
            }
        }
        gpu_vram_dirty_mark_row((uint32_t)py);
    }
}

void sw_vram_transfer_out(int x, int y, int w, int h, uint16_t *data) {
    METAL_BATCH_SYNC();
    if (!data || !g_vram || w <= 0 || h <= 0) return;

    if (x == 0 && y == 0 && w == VRAM_WIDTH && h == VRAM_HEIGHT) {
        memcpy(data, g_vram, (size_t)VRAM_WIDTH * (size_t)VRAM_HEIGHT * sizeof(uint16_t));
        return;
    }

    x &= (VRAM_WIDTH - 1);
    y &= (VRAM_HEIGHT - 1);
    if (x + w <= VRAM_WIDTH && y + h <= VRAM_HEIGHT) {
        for (int row = 0; row < h; row++) {
            memcpy(data + (size_t)row * (size_t)w,
                   g_vram + (size_t)(y + row) * VRAM_WIDTH + (size_t)x,
                   (size_t)w * sizeof(uint16_t));
        }
        return;
    }

    int idx = 0;
    for (int row = 0; row < h; row++) {
        int py = (y + row) & (VRAM_HEIGHT - 1);
        for (int col = 0; col < w; col++) {
            int px = (x + col) & (VRAM_WIDTH - 1);
            data[idx++] = g_vram[py * VRAM_WIDTH + px];
        }
    }
}

/* ------------------------------------------------------------------ */
/* Display output — convert 15-bit VRAM to 32-bit RGBA                */
/* ------------------------------------------------------------------ */

static inline uint32_t rgb555_to_argb(uint16_t pix) {
    int r5 = (pix >>  0) & 0x1F;
    int g5 = (pix >>  5) & 0x1F;
    int b5 = (pix >> 10) & 0x1F;
    uint8_t r = (uint8_t)((r5 << 3) | (r5 >> 2));
    uint8_t g = (uint8_t)((g5 << 3) | (g5 >> 2));
    uint8_t b = (uint8_t)((b5 << 3) | (b5 >> 2));
    /* ARGB8888 (matches gpu_display_pixel_argb): A<<24 | R<<16 | G<<8 | B. */
    return ((uint32_t)r << 16) | ((uint32_t)g << 8) | (uint32_t)b | 0xFF000000u;
}

int sw_render_display(uint32_t *out_pixels, int out_pitch,
                      int disp_x, int disp_y, int disp_w, int disp_h) {
    METAL_BATCH_SYNC();
    int count = 0;

    for (int row = 0; row < disp_h; row++) {
        int vy = (disp_y + row) & (VRAM_HEIGHT - 1);
        uint32_t *dst = (uint32_t *)((uint8_t *)out_pixels + row * out_pitch);

        for (int col = 0; col < disp_w; col++) {
            int vx = (disp_x + col) & (VRAM_WIDTH - 1);
            dst[col] = rgb555_to_argb(g_vram[vy * VRAM_WIDTH + vx]);
            count++;
        }
    }

    return count;
}

int sw_render_display_hires(uint32_t *out_pixels, int out_pitch,
                            int disp_x, int disp_y, int disp_w, int disp_h) {
    METAL_BATCH_SYNC();
    if (!g_hr || g_scale <= 1)
        return sw_render_display(out_pixels, out_pitch, disp_x, disp_y,
                                 disp_w, disp_h);

    int s = g_scale;
    int hx = disp_x * s;
    int hy = disp_y * s;
    int out_w = disp_w * s;
    int out_h = disp_h * s;
    int count = 0;

    for (int row = 0; row < out_h; row++) {
        int vy = (hy + row) % g_hr_h;
        const uint16_t *src = g_hr + (size_t)vy * g_hr_w;
        uint32_t *dst = (uint32_t *)((uint8_t *)out_pixels + row * out_pitch);

        for (int col = 0; col < out_w; col++) {
            int vx = (hx + col) % g_hr_w;
            dst[col] = rgb555_to_argb(src[vx]);
            count++;
        }
    }

    return count;
}

/* ------------------------------------------------------------------ */
/* Native-wide compositor surfaces                                    */
/* ------------------------------------------------------------------ */

static void wide_free_all(void) {
    for (int i = 0; i < WIDE_MAX_SURF; i++) {
        if (g_wide_surf[i]) { free(g_wide_surf[i]); g_wide_surf[i] = NULL; }
        g_wide_base[i] = -1;
    }
    g_wide_cur = NULL;
}

static uint16_t *wide_surf_for(int base_x) {
    if (g_wide_w <= 0) return NULL;
    for (int i = 0; i < WIDE_MAX_SURF; i++)
        if (g_wide_surf[i] && g_wide_base[i] == base_x) return g_wide_surf[i];
    for (int i = 0; i < WIDE_MAX_SURF; i++) {
        if (!g_wide_surf[i]) {
            size_t n = (size_t)(g_wide_w * g_scale) * (size_t)(VRAM_HEIGHT * g_scale);
            g_wide_surf[i] = (uint16_t *)calloc(n, sizeof(uint16_t));
            if (!g_wide_surf[i]) return NULL;
            g_wide_base[i] = base_x;
            return g_wide_surf[i];
        }
    }
    return NULL;  /* more distinct buffers than WIDE_MAX_SURF — shouldn't happen */
}

/* Enable native-wide with a wide width + centering offset (native px), or
 * disable (wide_w <= 0). Re-allocates if the width changed. */
void sw_wide_set_view(int enabled, int shift, int pad_left, int pad_right) {
    METAL_BATCH_SYNC();
    if (!enabled) shift = pad_left = pad_right = 0;
    if (view_enabled == enabled && view_shift == shift &&
        view_pad_left == pad_left && view_pad_right == pad_right) return;
    view_enabled = enabled;
    view_shift = shift;
    view_pad_left = pad_left;
    view_pad_right = pad_right;
}

void sw_wide_configure(int wide_w, int offset) {
    METAL_BATCH_SYNC();
    if (wide_w <= 0) { wide_free_all(); g_wide_w = 0; g_wide_off = 0; return; }
    if (wide_w != g_wide_w) wide_free_all();
    g_wide_w = wide_w;
    g_wide_off = offset;
}

/* Select the wide surface to mirror into for the back buffer at base_x. */
void sw_wide_set_target(int base_x) {
    METAL_BATCH_SYNC();
    g_wide_cur = wide_surf_for(base_x);
    g_wide_cur_base = base_x;
}

/* Stop mirroring (offscreen draws that don't target a framebuffer). */
void sw_wide_disable_target(void) { METAL_BATCH_SYNC(); g_wide_cur = NULL; }

/* Mirror a framebuffer clear: fill the full wide width over [y, y+h) of the
 * surface for base_x, so the revealed margins are clean (not stale). */
void sw_wide_clear(int base_x, int y, int h, uint16_t color) {
    METAL_BATCH_SYNC();
    uint16_t *surf = wide_surf_for(base_x);
    if (!surf) return;
    int s = g_scale;
    int W = g_wide_w * s;
    int H = VRAM_HEIGHT * s;
    int y0 = y * s, y1 = (y + h) * s;
    if (y0 < 0) y0 = 0;
    if (y1 > H) y1 = H;
    for (int row = y0; row < y1; row++) {
        uint16_t *dst = surf + (size_t)row * W;
        for (int col = 0; col < W; col++) dst[col] = color;
    }
}

/* Clear only the synthetic columns outside the centred canonical framebuffer.
 * This provides a defined black void without disturbing guest-owned pixels. */
void sw_wide_clear_margins(int base_x, int y, int h, uint16_t color, int sides) {
    METAL_BATCH_SYNC();
    uint16_t *surf = wide_surf_for(base_x);
    if (!surf) return;
    int s = g_scale;
    int W = g_wide_w * s;
    int H = VRAM_HEIGHT * s;
    int margin = g_wide_off * s;
    int y0 = y * s, y1 = (y + h) * s;
    if (margin <= 0 || margin * 2 >= W) return;
    if (y0 < 0) y0 = 0;
    if (y1 > H) y1 = H;
    for (int row = y0; row < y1; row++) {
        uint16_t *dst = surf + (size_t)row * W;
        if (sides & 1) for (int col = 0; col < margin; col++) dst[col] = color;
        if (sides & 2) for (int col = W - margin; col < W; col++) dst[col] = color;
    }
}

/* An existing wide surface for base_x (never allocates: a source that was not
 * drawn yet has nothing to copy). */
static uint16_t *wide_surf_existing(int base_x) {
    if (g_wide_w <= 0) return NULL;
    for (int i = 0; i < WIDE_MAX_SURF; i++)
        if (g_wide_surf[i] && g_wide_base[i] == base_x) return g_wide_surf[i];
    return NULL;
}

/* Band texels count as bit 15 set: the band columns' own bit 15 is not
 * authoritative (margin fills write 0), and a framebuffer trail keeps the
 * centre at bit 15 once it blends, so the bands blend exactly like it. */
void sw_wide_feedback_rect(int src_base, int src_x, int src_y, int dst_x, int dst_y,
                           int w, int h, int r, int g, int b, int raw, int semi) {
    METAL_BATCH_SYNC();
    const uint16_t *src = wide_surf_existing(src_base);
    if (!src || !g_wide_cur || src == g_wide_cur || w <= 0 || h <= 0) return;
    RTarget t = rt_wide();
    const int s = t.s;
    const int sx0 = (src_x - src_base + g_wide_off + view_shift) * s;
    const int dx0 = (dst_x + wide_dx()) * s;
    const int saved_enabled = g_semi_trans_enabled, saved_mode = g_semi_trans_mode;
    g_semi_trans_enabled = semi >= 0;
    g_semi_trans_mode = semi >= 0 ? semi : saved_mode;
    for (int row = 0; row < h * s; row++) {
        const int sy = src_y * s + row;
        if (sy < 0 || sy >= t.h) continue;
        for (int col = 0; col < w * s; col++) {
            const int sx = sx0 + col;
            if (sx < 0 || sx >= t.w) continue;
            const uint16_t texel = src[(size_t)sy * t.w + sx];
            if (texel == 0x0000) continue;
            /* Bit 15 is forced: see sw_wide_feedback_rect's comment. put_textured
             * takes the factor sw_set_color_modulation stores. */
            put_textured(&t, dx0 + col, dst_y * s + row, (uint16_t)(texel | 0x8000),
                         r >> 3, g >> 3, b >> 3, raw);
        }
    }
    g_semi_trans_enabled = saved_enabled;
    g_semi_trans_mode = saved_mode;
}

/* Present source: convert the wide surface for the displayed buffer (base_x) to
 * ARGB. Output is (g_wide_w*scale) wide × (disp_h*scale) tall. Returns 0 if no
 * surface exists for base_x (caller falls back to the canonical present). */
int sw_render_wide_display(uint32_t *out_pixels, int out_pitch, int base_x,
                           int disp_y, int disp_h) {
    METAL_BATCH_SYNC();
    uint16_t *surf = NULL;
    for (int i = 0; i < WIDE_MAX_SURF; i++)
        if (g_wide_surf[i] && g_wide_base[i] == base_x) { surf = g_wide_surf[i]; break; }
    if (!surf || g_wide_w <= 0) return 0;
    int s = g_scale;
    int W = g_wide_w * s;
    int H = VRAM_HEIGHT * s;
    int out_h = disp_h * s;
    int count = 0;
    for (int row = 0; row < out_h; row++) {
        int vy = disp_y * s + row;
        if (vy < 0) vy = 0;
        if (vy >= H) vy = H - 1;
        const uint16_t *src = surf + (size_t)vy * W;
        uint32_t *dst = (uint32_t *)((uint8_t *)out_pixels + row * out_pitch);
        for (int col = 0; col < W; col++) { dst[col] = rgb555_to_argb(src[col]); count++; }
    }
    return count;
}

int sw_wide_width(void) { return g_wide_w; }

/* Diagnostic: dump the ENTIRE active wide surface for base_x (full g_wide_w*scale
 * × VRAM_HEIGHT*scale — BOTH double-buffer y-bands, all margins) to ARGB. Lets a
 * probe see exactly where the over-draw lands, independent of the per-frame band
 * the present selects. *ow/*oh receive the surface dims. Returns pixels written,
 * or 0 if no surface for base_x / it won't fit in cap_pixels. */
int sw_wide_dump_full(uint32_t *out, int cap_pixels, int *ow, int *oh, int base_x) {
    METAL_BATCH_SYNC();
    uint16_t *surf = NULL;
    for (int i = 0; i < WIDE_MAX_SURF; i++)
        if (g_wide_surf[i] && g_wide_base[i] == base_x) { surf = g_wide_surf[i]; break; }
    if (!surf || g_wide_w <= 0) return 0;
    int s = g_scale;
    int W = g_wide_w * s, H = VRAM_HEIGHT * s;
    if ((long long)W * H > cap_pixels) return 0;
    for (int i = 0; i < W * H; i++) out[i] = rgb555_to_argb(surf[i]);
    if (ow) *ow = W;
    if (oh) *oh = H;
    return W * H;
}

/* ------------------------------------------------------------------ */
/* Renderer state snapshot (recordings and offline replay)            */
/* ------------------------------------------------------------------ */

/* Fixed-width fields so a snapshot means the same thing in every build. */
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
} SwStateHeader;

static const char k_sw_state_magic[8] = {'P','S','X','S','W','S','1','\0'};

size_t sw_state_size(void) {
    size_t pixels = (size_t)VRAM_WIDTH * VRAM_HEIGHT;
    if (g_hr) pixels += (size_t)g_hr_w * g_hr_h;
    return sizeof(SwStateHeader) + pixels * sizeof(uint16_t);
}

int sw_state_save(void *out, size_t capacity) {
    METAL_BATCH_SYNC();
    if (g_wide_w || capacity < sw_state_size()) return 0;
    SwStateHeader h;
    memset(&h, 0, sizeof h);
    memcpy(h.magic, k_sw_state_magic, sizeof h.magic);
    h.scale = g_scale;
    h.clip[0] = g_clip_x1; h.clip[1] = g_clip_y1; h.clip[2] = g_clip_x2; h.clip[3] = g_clip_y2;
    h.offset[0] = g_offset_x; h.offset[1] = g_offset_y;
    h.semi_enabled = g_semi_trans_enabled; h.semi_mode = g_semi_trans_mode;
    h.mask_set = g_mask_set_bit; h.mask_check = g_mask_check_bit;
    h.tw[0] = g_tw_mask_x; h.tw[1] = g_tw_mask_y; h.tw[2] = g_tw_off_x; h.tw[3] = g_tw_off_y;
    h.mod[0] = g_mod_r; h.mod[1] = g_mod_g; h.mod[2] = g_mod_b;
    h.raw_texture = g_raw_texture;
    h.texture_filter = g_texture_filter; h.texture_filter_req = g_texture_filter_req;
    h.faithful = g_faithful_authority;
    h.precise_valid = g_precise_valid;
    memcpy(h.precise_x16, g_precise_x16, sizeof h.precise_x16);
    memcpy(h.precise_y16, g_precise_y16, sizeof h.precise_y16);
    h.perspective_valid = g_perspective_valid;
    memcpy(h.perspective_q, g_perspective_q, sizeof h.perspective_q);
    memcpy(h.uv_lim, g_uv_lim, sizeof h.uv_lim);
    h.view[0] = view_enabled; h.view[1] = view_shift;
    h.view[2] = view_pad_left; h.view[3] = view_pad_right;
    uint8_t *p = (uint8_t *)out;
    memcpy(p, &h, sizeof h);
    p += sizeof h;
    memcpy(p, g_vram, (size_t)VRAM_WIDTH * VRAM_HEIGHT * sizeof(uint16_t));
    p += (size_t)VRAM_WIDTH * VRAM_HEIGHT * sizeof(uint16_t);
    if (g_hr) memcpy(p, g_hr, (size_t)g_hr_w * g_hr_h * sizeof(uint16_t));
    return 1;
}

/* Restores a snapshot into an initialized renderer, including its scale. */
int sw_state_load(const void *in, size_t size) {
    METAL_BATCH_SYNC();
    SwStateHeader h;
    if (size < sizeof h) return 0;
    memcpy(&h, in, sizeof h);
    if (memcmp(h.magic, k_sw_state_magic, sizeof h.magic) != 0) return 0;
    sw_renderer_set_scale(h.scale);
    if (g_scale != h.scale || size != sw_state_size()) return 0;
    g_clip_x1 = h.clip[0]; g_clip_y1 = h.clip[1]; g_clip_x2 = h.clip[2]; g_clip_y2 = h.clip[3];
    g_offset_x = h.offset[0]; g_offset_y = h.offset[1];
    g_semi_trans_enabled = h.semi_enabled; g_semi_trans_mode = h.semi_mode;
    g_mask_set_bit = h.mask_set; g_mask_check_bit = h.mask_check;
    g_tw_mask_x = h.tw[0]; g_tw_mask_y = h.tw[1]; g_tw_off_x = h.tw[2]; g_tw_off_y = h.tw[3];
    g_mod_r = h.mod[0]; g_mod_g = h.mod[1]; g_mod_b = h.mod[2];
    g_raw_texture = h.raw_texture;
    g_texture_filter = h.texture_filter; g_texture_filter_req = h.texture_filter_req;
    g_faithful_authority = h.faithful;
    g_precise_valid = h.precise_valid;
    memcpy(g_precise_x16, h.precise_x16, sizeof h.precise_x16);
    memcpy(g_precise_y16, h.precise_y16, sizeof h.precise_y16);
    g_perspective_valid = h.perspective_valid;
    memcpy(g_perspective_q, h.perspective_q, sizeof h.perspective_q);
    memcpy(g_uv_lim, h.uv_lim, sizeof h.uv_lim);
    view_enabled = h.view[0]; view_shift = h.view[1];
    view_pad_left = h.view[2]; view_pad_right = h.view[3];
    const uint8_t *p = (const uint8_t *)in + sizeof h;
    memcpy(g_vram, p, (size_t)VRAM_WIDTH * VRAM_HEIGHT * sizeof(uint16_t));
    p += (size_t)VRAM_WIDTH * VRAM_HEIGHT * sizeof(uint16_t);
    if (g_hr) memcpy(g_hr, p, (size_t)g_hr_w * g_hr_h * sizeof(uint16_t));
    return 1;
}
