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

#endif
