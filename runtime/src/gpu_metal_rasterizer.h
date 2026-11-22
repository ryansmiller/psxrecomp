#ifndef PSX_GPU_METAL_RASTERIZER_H
#define PSX_GPU_METAL_RASTERIZER_H

#include <stdint.h>

typedef struct {
    uint32_t x, width;
} PsxMetalSpan;

/* One clipped span per row. On failure, pixels are untouched. */
int psx_metal_draw_spans(uint16_t *rows, int pitch, const PsxMetalSpan *spans,
                         int height, uint16_t color, int blend,
                         int check_mask, int set_mask);
uint64_t psx_metal_completed_draws(void);

/* Batched primitives for one target buffer. Primitives apply in index order.
 * Each entry is one clipped span of one primitive on one row. */
enum {
    PSX_METAL_OPAQUE = 4,          /* blend value for opaque primitives */
    PSX_METAL_CHECK_MASK = 1u,
    PSX_METAL_SET_MASK = 2u,
    PSX_METAL_TRACK_DIRTY = 4u,    /* writes mark the row dirty */
    PSX_METAL_PERSPECTIVE = 8u,    /* textured triangle: perspective-correct uv */
    PSX_METAL_SHADED = 16u,        /* textured triangle: per-pixel modulation */
};

enum {
    PSX_METAL_FLAT = 0,            /* color: 15-bit color */
    PSX_METAL_SPRITE = 1,          /* texel (u + col / w, v + row / w) */
    PSX_METAL_SCALED_SPRITE = 2,   /* texel (u + du * col / w, v + dv * row / h) */
    PSX_METAL_GOURAUD = 3,         /* colour from the entry's row attributes */
    PSX_METAL_TEX_TRIANGLE = 4,    /* uv (and shading) from the row attributes */
};

/* All fields are 32-bit so the Metal struct has the same layout.
 * Textured primitives: color = mod_r | mod_g << 8 | mod_b << 16 | raw << 24
 * (5-bit modulation factors), clut = x | y << 16, window = mask_x |
 * mask_y << 8 | offset_x << 16 | offset_y << 24, and (x, y) is the
 * rectangle's origin in target pixels. A sprite's w is the target scale. */
typedef struct {
    uint32_t kind, color, blend, flags;
    uint32_t texpage, clut, window;
    int32_t x, y, u, v, w, h, du, dv;
} PsxMetalPrimitive;

/* attr is the word offset of the entry's row attributes, for kinds that
 * have them. */
typedef struct {
    uint32_t row, x, width, primitive, attr;
} PsxMetalEntry;

/* A Gouraud row: unclipped edges xa..xb and 5-bit edge colours, exactly as
 * gpu_sw_renderer.c's gouraud_row computes them. */
typedef struct {
    int32_t xa, span, ra, ga, ba, rb, gb, bb;
} PsxMetalRowAttr;

/* A textured-triangle row, as gpu_sw_renderer.c's tex_row computes it:
 * unclipped left edge, span (at least 1), and the edge values of u, v, q,
 * u*q, v*q and the 5-bit modulation colours. */
typedef struct {
    int32_t xa, span;
    float ua, ub, va, vb, qa, qb, uqa, uqb, vqa, vqb, ra, rb, ga, gb, ba, bb;
} PsxMetalTexAttr;

/* Drops any shared Metal buffer wrapping memory before it is freed. */
void psx_metal_forget_memory(const void *memory);
/* Batch flushes that used the target in place, and ones that copied rows
 * because the target was not page-aligned. */
void psx_metal_flush_counts(uint64_t *zero_copy, uint64_t *copied);

/* Applies the batch to rows of buf. Textured primitives read texels and
 * CLUTs from texture_vram, the 1024x512 native VRAM; the caller guarantees
 * the batch writes no VRAM that it also reads. dirty_rows, when non-NULL,
 * has one byte per buffer row; rows written by a TRACK_DIRTY primitive are
 * set to 1. Returns 0 with buf and dirty_rows untouched when Metal is
 * unavailable. */
int psx_metal_draw_batch(uint16_t *buf, int pitch, int rows,
                         const uint16_t *texture_vram,
                         const PsxMetalPrimitive *primitives,
                         uint32_t primitive_count,
                         const PsxMetalEntry *entries, uint32_t entry_count,
                         const uint32_t *attr_words, uint32_t attr_word_count,
                         unsigned char *dirty_rows);

#endif
