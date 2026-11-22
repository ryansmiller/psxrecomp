#ifndef PSX_GPU_METAL_RENDERER_H
#define PSX_GPU_METAL_RENDERER_H

/* Metal GPU-surface backend (gpu_metal_renderer.m), modelled on the OpenGL
 * backend: the authoritative VRAM is an RGBA8 texture at the internal scale
 * (alpha carries the PS1 mask bit), with a native 16-bit copy for texture
 * sampling and CPU readback. gpu.c's array is the CPU mirror, written by CPU
 * transfers at once and brought up to date from the GPU before any guest
 * readback.
 *
 * Built only with PSX_METAL_RENDERER (macOS). Implemented so far: VRAM,
 * fills, VRAM copies, CPU transfers, readbacks and display decoding.
 * Primitive drawing halts with a fatal error until it is implemented. */

#include "gpu_render.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The backend table, or NULL when Metal or read-write textures are
 * unavailable on this machine. */
const GpuRenderBackend *metal_backend_get(void);
/* Submits queued GPU work and waits for it. */
void metal_renderer_finish(void);
/* Direct present: two IOSurface-backed CVPixelBufferRefs (index 0 and 1)
 * of at least width x height (BGRA) that the window's SDL textures wrap, or
 * NULL. The backend keeps ownership. present_direct writes display rect
 * (dx, dy, dw, dh) into surface `index` on the GPU and waits for it, with a
 * black row below. With out_w x out_h set and the internal image more than
 * 1.25x larger, it resolves the area to that size, as OpenGL presents; aa
 * 1 (fxaa) or 2 (fxaa_hq) then runs FXAA on the result. *img_w/*img_h get
 * the written size. Returns 0 when it cannot. */
void *metal_renderer_present_pixel_buffer(int index, int width, int height);
int metal_renderer_present_direct(int index, int dx, int dy, int dw, int dh,
                                  int out_w, int out_h, int aa, int *img_w, int *img_h);

#ifdef __cplusplus
}
#endif

#endif
