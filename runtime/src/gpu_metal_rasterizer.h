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

/* Batched flat primitives for one target buffer. Primitives apply in index
 * order. Each entry is one clipped span of one primitive on one row. */
enum {
    PSX_METAL_OPAQUE = 4,          /* blend value for opaque primitives */
    PSX_METAL_CHECK_MASK = 1u,
    PSX_METAL_SET_MASK = 2u,
    PSX_METAL_TRACK_DIRTY = 4u,    /* writes mark the row dirty */
};

typedef struct {
    uint16_t color;
    uint8_t blend;                 /* PS1 semi-transparency mode or OPAQUE */
    uint8_t flags;
} PsxMetalPrimitive;

typedef struct {
    uint32_t row, x, width, primitive;
} PsxMetalEntry;

/* Applies the batch to rows of buf. dirty_rows, when non-NULL, has one byte
 * per buffer row; rows written by a TRACK_DIRTY primitive are set to 1.
 * Returns 0 with buf and dirty_rows untouched when Metal is unavailable. */
/* Drops any shared Metal buffer wrapping memory before it is freed. */
void psx_metal_forget_memory(const void *memory);
/* Batch flushes that used the target in place, and ones that copied rows
 * because the target was not page-aligned. */
void psx_metal_flush_counts(uint64_t *zero_copy, uint64_t *copied);

int psx_metal_draw_batch(uint16_t *buf, int pitch, int rows,
                         const PsxMetalPrimitive *primitives,
                         uint32_t primitive_count,
                         const PsxMetalEntry *entries, uint32_t entry_count,
                         unsigned char *dirty_rows);

#endif
