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
/* Direct present: an IOSurface-backed CVPixelBufferRef of at least
 * width x height (BGRA) that the window's SDL texture wraps, or NULL. The
 * backend keeps ownership. present_direct decodes display rect (dx, dy,
 * dw, dh) at internal resolution into it on the GPU and waits for the GPU;
 * pad_row also clears the row below. Returns 0 when it cannot. */
void *metal_renderer_present_pixel_buffer(int width, int height);
int metal_renderer_present_direct(int dx, int dy, int dw, int dh, int pad_row);

#ifdef __cplusplus
}
#endif

#endif
