/* gpu_metal_renderer.m — Metal GPU-surface backend (gpu_metal_renderer.h).
 *
 * VRAM MODEL (as gpu_gl_renderer.c's v2 model)
 *   s_hr   RGBA8Unorm texture, 1024*S x 512*S: the authoritative image.
 *          Each 5-bit channel c is stored as c << 3 (OpenGL's "accurate
 *          blending" basis), alpha 1.0 when bit 15 (the mask bit) is set.
 *   s_raw  1024 x 512 16-bit words (shared buffer): native VRAM as the GPU
 *          last wrote it, for texture sampling and CPU readback.
 *   gpu.c's array is the CPU mirror. CPU transfers write it at once
 *          (through the software renderer at 1x) and queue the same pixels
 *          to the GPU; GPU fills and copies mark it stale, and every guest
 *          readback first waits for the GPU and copies s_raw's stale rows.
 *
 * Commands encode into one open compute encoder (serial dispatch, so each
 * dispatch sees the previous one's writes) and are submitted at readbacks.
 * Fills and copies follow gpu_sw_renderer.c exactly: coordinates wrap at
 * 1024x512; fills ignore the mask; copies snapshot each source row before
 * writing it, skip masked destinations when checking, and set bit 15 when
 * setting. A copy whose source and destination overlap runs row by row, so
 * a row written earlier is read later just as the software loop reads it. */

#import <Metal/Metal.h>

#include "gpu_metal_renderer.h"
#include "gpu_sw_renderer.h"
#include "crash_trace.h"
#include "gpu_vram_dirty.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { VRAM_W = 1024, VRAM_H = 512, MAX_SCALE = 16, STAGING_BYTES = 8u << 20 };

typedef struct {
    int32_t x, y, w, h;       /* destination rect (native), x/y already wrapped */
    int32_t sx, sy;           /* copy source (native) */
    uint32_t color, scale;
    uint32_t flags;           /* 1 = check mask, 2 = set mask */
    uint32_t offset;          /* staging offset in 16-bit words */
    int32_t row0, rows;       /* copy rows handled by this dispatch */
} MetalOp;

static NSString *const k_source =
    @"#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct Op { int x, y, w, h, sx, sy; uint color, scale, flags, offset; int row0, rows; };\n"
    /* 5-bit channels in the c << 3 basis; alpha is the mask bit. */
    "static float4 rgba(uint c) {\n"
    "    return float4(float((c & 31u) << 3), float(((c >> 5) & 31u) << 3),\n"
    "                  float(((c >> 10) & 31u) << 3), (c & 0x8000u) ? 255.0 : 0.0) / 255.0;\n"
    "}\n"
    "static uint2 hr_at(constant Op &o, uint2 g, int ox, int oy) {\n"
    "    uint S = o.scale;\n"
    "    uint nx = uint(ox + int(g.x / S)) & 1023u, ny = uint(oy + int(g.y / S)) & 511u;\n"
    "    return uint2(nx * S + g.x % S, ny * S + g.y % S);\n"
    "}\n"
    "kernel void fill(texture2d<float, access::write> hr [[texture(0)]],\n"
    "                 device ushort *raw [[buffer(0)]], constant Op &o [[buffer(1)]],\n"
    "                 uint2 g [[thread_position_in_grid]]) {\n"
    "    uint S = o.scale;\n"
    "    if (g.x >= uint(o.w) * S || g.y >= uint(o.h) * S) return;\n"
    "    hr.write(rgba(o.color), hr_at(o, g, o.x, o.y));\n"
    "    if (g.x % S == 0u && g.y % S == 0u) {\n"
    "        uint nx = uint(o.x + int(g.x / S)) & 1023u, ny = uint(o.y + int(g.y / S)) & 511u;\n"
    "        raw[ny * 1024u + nx] = ushort(o.color);\n"
    "    }\n"
    "}\n"
    "kernel void upload(texture2d<float, access::write> hr [[texture(0)]],\n"
    "                   device ushort *raw [[buffer(0)]], constant Op &o [[buffer(1)]],\n"
    "                   device const ushort *src [[buffer(2)]],\n"
    "                   uint2 g [[thread_position_in_grid]]) {\n"
    "    uint S = o.scale;\n"
    "    if (g.x >= uint(o.w) * S || g.y >= uint(o.h) * S) return;\n"
    "    uint c = src[o.offset + (g.y / S) * uint(o.w) + g.x / S];\n"
    "    hr.write(rgba(c), hr_at(o, g, o.x, o.y));\n"
    "    if (g.x % S == 0u && g.y % S == 0u) {\n"
    "        uint nx = uint(o.x + int(g.x / S)) & 1023u, ny = uint(o.y + int(g.y / S)) & 511u;\n"
    "        raw[ny * 1024u + nx] = ushort(c);\n"
    "    }\n"
    "}\n"
    /* Copy, phase 1: snapshot source rows row0..row0+rows-1. */
    "kernel void copy_gather(texture2d<float, access::read> hr [[texture(0)]],\n"
    "                        texture2d<float, access::write> scratch [[texture(1)]],\n"
    "                        device const ushort *raw [[buffer(0)]], constant Op &o [[buffer(1)]],\n"
    "                        device ushort *raw_scratch [[buffer(2)]],\n"
    "                        uint2 g [[thread_position_in_grid]]) {\n"
    "    uint S = o.scale;\n"
    "    if (g.x >= uint(o.w) * S || g.y >= uint(o.rows) * S) return;\n"
    "    scratch.write(hr.read(hr_at(o, g, o.sx, o.sy + o.row0)), g);\n"
    "    if (g.x % S == 0u && g.y % S == 0u) {\n"
    "        uint nx = uint(o.sx + int(g.x / S)) & 1023u;\n"
    "        uint ny = uint(o.sy + o.row0 + int(g.y / S)) & 511u;\n"
    "        raw_scratch[(g.y / S) * 1024u + g.x / S] = raw[ny * 1024u + nx];\n"
    "    }\n"
    "}\n"
    /* Copy, phase 2: write the snapshot with the mask rules. */
    "kernel void copy_scatter(texture2d<float, access::read_write> hr [[texture(0)]],\n"
    "                         texture2d<float, access::read> scratch [[texture(1)]],\n"
    "                         device ushort *raw [[buffer(0)]], constant Op &o [[buffer(1)]],\n"
    "                         device const ushort *raw_scratch [[buffer(2)]],\n"
    "                         uint2 g [[thread_position_in_grid]]) {\n"
    "    uint S = o.scale;\n"
    "    if (g.x >= uint(o.w) * S || g.y >= uint(o.rows) * S) return;\n"
    "    uint2 d = hr_at(o, g, o.x, o.y + o.row0);\n"
    "    if (!((o.flags & 1u) && hr.read(d).a > 0.5)) {\n"
    "        float4 c = scratch.read(g);\n"
    "        if (o.flags & 2u) c.a = 1.0;\n"
    "        hr.write(c, d);\n"
    "    }\n"
    "    if (g.x % S == 0u && g.y % S == 0u) {\n"
    "        uint nx = uint(o.x + int(g.x / S)) & 1023u;\n"
    "        uint ny = uint(o.y + o.row0 + int(g.y / S)) & 511u;\n"
    "        uint back = raw[ny * 1024u + nx];\n"
    "        if (!((o.flags & 1u) && (back & 0x8000u))) {\n"
    "            uint p = raw_scratch[(g.y / S) * 1024u + g.x / S];\n"
    "            raw[ny * 1024u + nx] = ushort((o.flags & 2u) ? (p | 0x8000u) : p);\n"
    "        }\n"
    "    }\n"
    "}\n"
    /* Display decode at internal resolution, as sw_render_display_hires. */
    "kernel void export_hires(texture2d<float, access::read> hr [[texture(0)]],\n"
    "                         device uint *out [[buffer(0)]], constant Op &o [[buffer(1)]],\n"
    "                         uint2 g [[thread_position_in_grid]]) {\n"
    "    uint S = o.scale;\n"
    "    if (g.x >= uint(o.w) * S || g.y >= uint(o.h) * S) return;\n"
    "    uint2 p = uint2((uint(o.x) * S + g.x) % (1024u * S), (uint(o.y) * S + g.y) % (512u * S));\n"
    "    float4 c = hr.read(p);\n"
    "    uint r = uint(rint(c.r * 255.0)) >> 3, gg = uint(rint(c.g * 255.0)) >> 3;\n"
    "    uint b = uint(rint(c.b * 255.0)) >> 3;\n"
    "    r = (r << 3) | (r >> 2); gg = (gg << 3) | (gg >> 2); b = (b << 3) | (b >> 2);\n"
    "    out[g.y * uint(o.w) * S + g.x] = 0xFF000000u | (r << 16) | (gg << 8) | b;\n"
    "}\n";

static id<MTLDevice> s_device;
static id<MTLCommandQueue> s_queue;
static id<MTLComputePipelineState> s_fill, s_upload, s_gather, s_scatter, s_export;
static id<MTLTexture> s_hr, s_scratch;
static id<MTLBuffer> s_raw, s_raw_scratch, s_staging, s_export_buf;
static id<MTLCommandBuffer> s_command;
static id<MTLComputeCommandEncoder> s_encoder;
static uint32_t s_staging_used;   /* bytes of s_staging queued since the last submit */
static uint16_t *s_vram;          /* gpu.c's array: the CPU mirror */
static int s_scale = 1, s_ready;
static int s_mask_set, s_mask_check;
static int s_clip[4] = {0, 0, VRAM_W - 1, VRAM_H - 1};
static int s_cpu_stale;           /* GPU wrote VRAM the CPU mirror lacks */
static int s_stale_y0 = VRAM_H, s_stale_y1 = -1;   /* stale native rows */

static void unimplemented(const char *what) {
    char reason[160];
    snprintf(reason, sizeof reason, "Metal renderer: %s is not implemented yet", what);
    psx_fatal_halt(reason);
}

static id<MTLComputePipelineState> pipeline(id<MTLLibrary> library, NSString *name) {
    NSError *error = nil;
    id<MTLFunction> function = [library newFunctionWithName:name];
    return function ? [s_device newComputePipelineStateWithFunction:function error:&error] : nil;
}

static int device_ready(void) {
    static int attempted, ok;
    if (attempted) return ok;
    attempted = 1;
    s_device = MTLCreateSystemDefaultDevice();
    /* RGBA8 read-write textures need tier 2 (every Apple-silicon GPU). */
    if (!s_device || s_device.readWriteTextureSupport < MTLReadWriteTextureTier2) return 0;
    MTLCompileOptions *options = [MTLCompileOptions new];
    NSError *error = nil;
    id<MTLLibrary> library = [s_device newLibraryWithSource:k_source options:options error:&error];
    if (!library) {
        fprintf(stderr, "psxrecomp: Metal renderer shaders failed: %s\n",
                error.localizedDescription.UTF8String);
        return 0;
    }
    s_fill = pipeline(library, @"fill");
    s_upload = pipeline(library, @"upload");
    s_gather = pipeline(library, @"copy_gather");
    s_scatter = pipeline(library, @"copy_scatter");
    s_export = pipeline(library, @"export_hires");
    s_queue = [s_device newCommandQueue];
    ok = s_fill && s_upload && s_gather && s_scatter && s_export && s_queue;
    return ok;
}

static id<MTLTexture> image(int scale, MTLTextureUsage usage) {
    MTLTextureDescriptor *d =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                           width:(NSUInteger)VRAM_W * scale
                                                          height:(NSUInteger)VRAM_H * scale
                                                       mipmapped:NO];
    d.storageMode = MTLStorageModePrivate;
    d.usage = usage;
    return [s_device newTextureWithDescriptor:d];
}

/* (Re)allocates the surfaces at s_scale and loads them from the CPU mirror. */
static int allocate(void) {
    const MTLTextureUsage rw = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite |
                               MTLTextureUsageRenderTarget;
    s_hr = image(s_scale, rw);
    s_scratch = image(s_scale, rw);
    const NSUInteger words = (NSUInteger)VRAM_W * VRAM_H * sizeof(uint16_t);
    s_raw = [s_device newBufferWithLength:words options:MTLResourceStorageModeShared];
    s_raw_scratch = [s_device newBufferWithLength:words options:MTLResourceStorageModeShared];
    s_staging = [s_device newBufferWithLength:STAGING_BYTES options:MTLResourceStorageModeShared];
    s_ready = s_hr && s_scratch && s_raw && s_raw_scratch && s_staging;
    return s_ready;
}

static id<MTLComputeCommandEncoder> encoder(void) {
    if (!s_encoder) {
        s_command = [s_queue commandBuffer];
        s_encoder = [s_command computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    }
    return s_encoder;
}

/* Submits queued work and waits for it. */
void metal_renderer_finish(void) {
    if (!s_encoder) return;
    [s_encoder endEncoding];
    [s_command commit];
    [s_command waitUntilCompleted];
    if (s_command.status != MTLCommandBufferStatusCompleted)
        psx_fatal_halt("Metal renderer: a command buffer failed");
    s_encoder = nil;
    s_command = nil;
    s_staging_used = 0;
}

static void dispatch(id<MTLComputePipelineState> p, const MetalOp *op, NSUInteger w, NSUInteger h) {
    if (!w || !h) return;
    id<MTLComputeCommandEncoder> e = encoder();
    [e setComputePipelineState:p];
    [e setBytes:op length:sizeof *op atIndex:1];
    const NSUInteger tw = p.threadExecutionWidth;
    const NSUInteger th = p.maxTotalThreadsPerThreadgroup / tw;
    [e dispatchThreads:MTLSizeMake(w, h, 1) threadsPerThreadgroup:MTLSizeMake(tw, th, 1)];
}

/* GPU wrote native rows y .. y+h-1 (wrapping); the CPU mirror lacks them. */
static void mark_stale(int y, int h) {
    s_cpu_stale = 1;
    if (h >= VRAM_H) { s_stale_y0 = 0; s_stale_y1 = VRAM_H - 1; return; }
    y &= VRAM_H - 1;
    if (y + h > VRAM_H) { s_stale_y0 = 0; s_stale_y1 = VRAM_H - 1; return; }
    if (y < s_stale_y0) s_stale_y0 = y;
    if (y + h - 1 > s_stale_y1) s_stale_y1 = y + h - 1;
}

/* Brings the CPU mirror up to date with the GPU's native VRAM. */
static void ensure_cpu(void) {
    if (!s_ready) return;
    metal_renderer_finish();
    if (!s_cpu_stale) return;
    const uint16_t *raw = s_raw.contents;
    memcpy(s_vram + (size_t)s_stale_y0 * VRAM_W, raw + (size_t)s_stale_y0 * VRAM_W,
           (size_t)(s_stale_y1 - s_stale_y0 + 1) * VRAM_W * sizeof(uint16_t));
    s_cpu_stale = 0;
    s_stale_y0 = VRAM_H;
    s_stale_y1 = -1;
}

/* Queues w x h pixels for the GPU at (x, y), wrapping. */
static void queue_upload(int x, int y, int w, int h, const uint16_t *data) {
    if (!s_ready || w <= 0 || h <= 0) return;
    const size_t bytes = (size_t)w * h * sizeof(uint16_t);
    if (bytes > STAGING_BYTES) psx_fatal_halt("Metal renderer: upload larger than staging");
    if (s_staging_used + bytes > STAGING_BYTES) metal_renderer_finish();
    memcpy((uint8_t *)s_staging.contents + s_staging_used, data, bytes);
    MetalOp op = {0};
    op.x = x & (VRAM_W - 1);
    op.y = y & (VRAM_H - 1);
    op.w = w;
    op.h = h;
    op.scale = (uint32_t)s_scale;
    op.offset = s_staging_used / sizeof(uint16_t);
    s_staging_used += (uint32_t)((bytes + 3u) & ~3u);
    id<MTLComputeCommandEncoder> e = encoder();
    [e setTexture:s_hr atIndex:0];
    [e setBuffer:s_raw offset:0 atIndex:0];
    [e setBuffer:s_staging offset:0 atIndex:2];
    dispatch(s_upload, &op, (NSUInteger)w * s_scale, (NSUInteger)h * s_scale);
}

/* ---- GpuRenderBackend --------------------------------------------------- */

static void mb_set_scale(int scale) {
    if (scale < 1) scale = 1;
    if (scale > MAX_SCALE) scale = MAX_SCALE;
    if (scale == s_scale && s_ready) return;
    s_scale = scale;
    if (s_vram && device_ready()) {
        ensure_cpu();
        if (!allocate()) psx_fatal_halt("Metal renderer: surfaces did not allocate");
        queue_upload(0, 0, VRAM_W, VRAM_H, s_vram);
        memcpy(s_raw.contents, s_vram, (size_t)VRAM_W * VRAM_H * sizeof(uint16_t));
    }
}

static void mb_init(uint16_t *vram) {
    s_vram = vram;
    sw_renderer_init(vram);   /* the CPU mirror's transfers and reads, at 1x */
    /* Draw state starts as sw_renderer_init leaves it. */
    s_mask_set = s_mask_check = 0;
    sw_get_draw_area(&s_clip[0], &s_clip[1], &s_clip[2], &s_clip[3]);
    s_ready = 0;
    if (!device_ready() || !allocate()) psx_fatal_halt("Metal renderer: initialization failed");
    memcpy(s_raw.contents, vram, (size_t)VRAM_W * VRAM_H * sizeof(uint16_t));
    queue_upload(0, 0, VRAM_W, VRAM_H, vram);
}

static int mb_scale(void) { return s_scale; }

static void mb_set_texture_filter(int bilinear) { sw_set_texture_filter(bilinear); }
static int mb_texture_filter(void) { return sw_texture_filter(); }
static void mb_set_semi(int enabled, int mode) { sw_set_semi_transparency(enabled, mode); }
static void mb_set_mask_bits(int set_bit, int check_bit) {
    s_mask_set = set_bit;
    s_mask_check = check_bit;
    sw_set_mask_bits(set_bit, check_bit);
}
static void mb_set_texture_window(uint32_t raw) { sw_set_texture_window(raw); }
static void mb_set_color_modulation(int r, int g, int b, int raw) {
    sw_set_color_modulation(r, g, b, raw);
}

static void mb_fill_rect(int x, int y, int w, int h, uint16_t color) {
    if (!s_ready || w <= 0 || h <= 0) return;
    MetalOp op = {0};
    op.x = x & (VRAM_W - 1);
    op.y = y & (VRAM_H - 1);
    op.w = w;
    op.h = h;
    op.color = color;
    op.scale = (uint32_t)s_scale;
    id<MTLComputeCommandEncoder> e = encoder();
    [e setTexture:s_hr atIndex:0];
    [e setBuffer:s_raw offset:0 atIndex:0];
    dispatch(s_fill, &op, (NSUInteger)w * s_scale, (NSUInteger)h * s_scale);
    mark_stale(op.y, h);
    gpu_vram_dirty_mark_rect(op.x, op.y, w, h);
}

/* Whether wrapped intervals [a, a+n) and [b, b+m) of a size-mod axis meet. */
static int spans_meet(int a, int n, int b, int m, int mod) {
    if (n >= mod || m >= mod) return 1;
    a &= mod - 1;
    b &= mod - 1;
    for (int k = -1; k <= 1; ++k) {
        const int bb = b + k * mod;
        if (a < bb + m && bb < a + n) return 1;
    }
    return 0;
}

static void copy_rows(const MetalOp *base, int row0, int rows) {
    MetalOp op = *base;
    op.row0 = row0;
    op.rows = rows;
    id<MTLComputeCommandEncoder> e = encoder();
    [e setBuffer:s_raw offset:0 atIndex:0];
    [e setBuffer:s_raw_scratch offset:0 atIndex:2];
    [e setTexture:s_hr atIndex:0];
    [e setTexture:s_scratch atIndex:1];
    dispatch(s_gather, &op, (NSUInteger)op.w * s_scale, (NSUInteger)rows * s_scale);
    dispatch(s_scatter, &op, (NSUInteger)op.w * s_scale, (NSUInteger)rows * s_scale);
}

static void mb_copy_rect(int src_x, int src_y, int dst_x, int dst_y, int w, int h) {
    if (!s_ready || w <= 0 || h <= 0) return;
    if (w > VRAM_W) w = VRAM_W;   /* sw_copy_rect's row buffer holds one VRAM row */
    MetalOp op = {0};
    op.sx = src_x & (VRAM_W - 1);
    op.sy = src_y & (VRAM_H - 1);
    op.x = dst_x & (VRAM_W - 1);
    op.y = dst_y & (VRAM_H - 1);
    op.w = w;
    op.h = h;
    op.scale = (uint32_t)s_scale;
    op.flags = (s_mask_check ? 1u : 0u) | (s_mask_set ? 2u : 0u);
    if (spans_meet(op.sx, w, op.x, w, VRAM_W) && spans_meet(op.sy, h, op.y, h, VRAM_H)) {
        for (int row = 0; row < h; ++row) copy_rows(&op, row, 1);
    } else {
        for (int row = 0; row < h; row += VRAM_H) copy_rows(&op, row, h - row < VRAM_H ? h - row : VRAM_H);
    }
    mark_stale(op.y, h);
    for (int row = 0; row < h && row < VRAM_H; ++row)
        gpu_vram_dirty_mark_row((uint32_t)((op.y + row) & (VRAM_H - 1)));
}

static void mb_draw_flat_triangle(int x0, int y0, int x1, int y1, int x2, int y2, uint16_t c) {
    (void)x0; (void)y0; (void)x1; (void)y1; (void)x2; (void)y2; (void)c;
    unimplemented("flat triangle drawing");
}
static void mb_draw_gouraud_triangle(int x0, int y0, uint16_t c0, int x1, int y1, uint16_t c1,
                                     int x2, int y2, uint16_t c2) {
    (void)x0; (void)y0; (void)c0; (void)x1; (void)y1; (void)c1; (void)x2; (void)y2; (void)c2;
    unimplemented("Gouraud triangle drawing");
}
static void mb_draw_textured_triangle(int x0, int y0, int u0, int v0, int x1, int y1, int u1,
                                      int v1, int x2, int y2, int u2, int v2, uint16_t cx,
                                      uint16_t cy, uint16_t tp) {
    (void)x0; (void)y0; (void)u0; (void)v0; (void)x1; (void)y1; (void)u1; (void)v1;
    (void)x2; (void)y2; (void)u2; (void)v2; (void)cx; (void)cy; (void)tp;
    unimplemented("textured triangle drawing");
}
static void mb_draw_shaded_textured_triangle(int x0, int y0, int u0, int v0, uint32_t c0,
                                             int x1, int y1, int u1, int v1, uint32_t c1,
                                             int x2, int y2, int u2, int v2, uint32_t c2,
                                             uint16_t cx, uint16_t cy, uint16_t tp, int raw) {
    (void)x0; (void)y0; (void)u0; (void)v0; (void)c0; (void)x1; (void)y1; (void)u1; (void)v1;
    (void)c1; (void)x2; (void)y2; (void)u2; (void)v2; (void)c2; (void)cx; (void)cy; (void)tp;
    (void)raw;
    unimplemented("shaded textured triangle drawing");
}
static void mb_draw_flat_rect(int x, int y, int w, int h, uint16_t c) {
    (void)x; (void)y; (void)w; (void)h; (void)c;
    unimplemented("flat rectangle drawing");
}
static void mb_draw_textured_rect(int x, int y, int w, int h, int u, int v, uint16_t cx,
                                  uint16_t cy, uint16_t tp) {
    (void)x; (void)y; (void)w; (void)h; (void)u; (void)v; (void)cx; (void)cy; (void)tp;
    unimplemented("textured rectangle drawing");
}
static void mb_draw_textured_rect_scaled(int x, int y, int w, int h, int u0, int v0, int u1,
                                         int v1, uint16_t cx, uint16_t cy, uint16_t tp) {
    (void)x; (void)y; (void)w; (void)h; (void)u0; (void)v0; (void)u1; (void)v1;
    (void)cx; (void)cy; (void)tp;
    unimplemented("scaled textured rectangle drawing");
}
static void mb_draw_line(int x0, int y0, int x1, int y1, uint16_t c) {
    (void)x0; (void)y0; (void)x1; (void)y1; (void)c;
    unimplemented("line drawing");
}
static void mb_draw_shaded_line(int x0, int y0, uint16_t c0, int x1, int y1, uint16_t c1) {
    (void)x0; (void)y0; (void)c0; (void)x1; (void)y1; (void)c1;
    unimplemented("shaded line drawing");
}

static int mb_render_display(uint32_t *out, int pitch, int dx, int dy, int dw, int dh) {
    ensure_cpu();
    return sw_render_display(out, pitch, dx, dy, dw, dh);
}

static int mb_render_display_hires(uint32_t *out, int pitch, int dx, int dy, int dw, int dh) {
    if (!s_ready || dw <= 0 || dh <= 0) return 0;
    const int S = s_scale, w = dw * S, h = dh * S;
    const NSUInteger bytes = (NSUInteger)w * h * 4;
    if (s_export_buf.length < bytes)
        s_export_buf = [s_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    MetalOp op = {0};
    op.x = dx & (VRAM_W - 1);
    op.y = dy & (VRAM_H - 1);
    op.w = dw;
    op.h = dh;
    op.scale = (uint32_t)S;
    id<MTLComputeCommandEncoder> e = encoder();
    [e setTexture:s_hr atIndex:0];
    [e setBuffer:s_export_buf offset:0 atIndex:0];
    dispatch(s_export, &op, (NSUInteger)w, (NSUInteger)h);
    metal_renderer_finish();
    const uint32_t *src = s_export_buf.contents;
    for (int row = 0; row < h; ++row)
        memcpy((uint8_t *)out + (size_t)row * pitch, src + (size_t)row * w, (size_t)w * 4);
    return w * h;
}

static void mb_vram_write(int x, int y, uint16_t pixel) {
    ensure_cpu();
    sw_vram_write(x, y, pixel);
    queue_upload(x, y, 1, 1, &pixel);
}

static uint16_t mb_vram_read(int x, int y) {
    ensure_cpu();
    return sw_vram_read(x, y);
}

static void mb_vram_transfer_in(int x, int y, int w, int h, const uint16_t *data) {
    if (!data || w <= 0 || h <= 0) return;
    /* Rows the GPU wrote but the mirror lacks must land before the
     * transfer's rows overwrite part of them. */
    if (s_cpu_stale) ensure_cpu();
    sw_vram_transfer_in(x, y, w, h, data);
    queue_upload(x, y, w, h, data);
}

static void mb_vram_transfer_out(int x, int y, int w, int h, uint16_t *data) {
    ensure_cpu();
    sw_vram_transfer_out(x, y, w, h, data);
}

static void mb_set_draw_area(int x1, int y1, int x2, int y2) {
    sw_set_draw_area(x1, y1, x2, y2);
    sw_get_draw_area(&s_clip[0], &s_clip[1], &s_clip[2], &s_clip[3]);
}
static void mb_get_draw_area(int *x1, int *y1, int *x2, int *y2) {
    *x1 = s_clip[0]; *y1 = s_clip[1]; *x2 = s_clip[2]; *y2 = s_clip[3];
}
static void mb_set_draw_offset(int x, int y) { sw_set_draw_offset(x, y); }

static void mb_display_mode_changed(void) {}

static const GpuRenderBackend k_metal = {
    .name = "metal",
    .init = mb_init,
    .set_scale = mb_set_scale,
    .scale = mb_scale,
    .set_texture_filter = mb_set_texture_filter,
    .texture_filter = mb_texture_filter,
    .set_semi_transparency = mb_set_semi,
    .set_mask_bits = mb_set_mask_bits,
    .set_texture_window = mb_set_texture_window,
    .set_color_modulation = mb_set_color_modulation,
    .fill_rect = mb_fill_rect,
    .copy_rect = mb_copy_rect,
    .draw_flat_triangle = mb_draw_flat_triangle,
    .draw_gouraud_triangle = mb_draw_gouraud_triangle,
    .draw_textured_triangle = mb_draw_textured_triangle,
    .draw_shaded_textured_triangle = mb_draw_shaded_textured_triangle,
    .draw_flat_rect = mb_draw_flat_rect,
    .draw_textured_rect = mb_draw_textured_rect,
    .draw_textured_rect_scaled = mb_draw_textured_rect_scaled,
    .draw_line = mb_draw_line,
    .draw_shaded_line = mb_draw_shaded_line,
    .render_display = mb_render_display,
    .render_display_hires = mb_render_display_hires,
    .vram_write = mb_vram_write,
    .vram_read = mb_vram_read,
    .vram_transfer_in = mb_vram_transfer_in,
    .vram_transfer_out = mb_vram_transfer_out,
    .set_draw_area = mb_set_draw_area,
    .get_draw_area = mb_get_draw_area,
    .set_draw_offset = mb_set_draw_offset,
    .display_mode_changed = mb_display_mode_changed,
};

const GpuRenderBackend *metal_backend_get(void) {
    return device_ready() ? &k_metal : NULL;
}
