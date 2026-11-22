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
#include <math.h>
#import <CoreVideo/CoreVideo.h>
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
    "}\n"
    /* The same display decode into the window's present texture: no CPU
     * copy. flags 1: also clear the row under the image (linear filtering
     * at the bottom edge then blends with black, as the CPU path pads). */
    "kernel void export_present(texture2d<float, access::read> hr [[texture(0)]],\n"
    "                           texture2d<float, access::write> out [[texture(1)]],\n"
    "                           constant Op &o [[buffer(1)]], uint2 g [[thread_position_in_grid]]) {\n"
    "    uint S = o.scale;\n"
    "    if (g.x >= uint(o.w) * S || g.y >= out.get_height() || g.x >= out.get_width()) return;\n"
    "    if (g.y >= uint(o.h) * S) {\n"
    "        if ((o.flags & 1u) && g.y == uint(o.h) * S) out.write(float4(0.0, 0.0, 0.0, 1.0), g);\n"
    "        return;\n"
    "    }\n"
    "    uint2 p = uint2((uint(o.x) * S + g.x) % (1024u * S), (uint(o.y) * S + g.y) % (512u * S));\n"
    "    float4 c = hr.read(p);\n"
    "    uint r = uint(rint(c.r * 255.0)) >> 3, gg = uint(rint(c.g * 255.0)) >> 3;\n"
    "    uint b = uint(rint(c.b * 255.0)) >> 3;\n"
    "    r = (r << 3) | (r >> 2); gg = (gg << 3) | (gg >> 2); b = (b << 3) | (b >> 2);\n"
    "    out.write(float4(float(r), float(gg), float(b), 255.0) / 255.0, g);\n"
    "}\n"
    /* Native VRAM from the image, as OpenGL's pack pass: each native pixel
     * is the top-left sample of its S x S block. */
    "kernel void pack(texture2d<float, access::read> hr [[texture(0)]],\n"
    "                 device ushort *raw [[buffer(0)]], constant Op &o [[buffer(1)]],\n"
    "                 uint2 g [[thread_position_in_grid]]) {\n"
    "    if (g.x >= uint(o.w) || g.y >= uint(o.h)) return;\n"
    "    uint nx = uint(o.x) + g.x, ny = uint(o.y) + g.y;\n"
    "    float4 c = hr.read(uint2(nx, ny) * o.scale);\n"
    "    uint r = uint(c.r * 255.0 + 0.5) >> 3, gg = uint(c.g * 255.0 + 0.5) >> 3;\n"
    "    uint b = uint(c.b * 255.0 + 0.5) >> 3;\n"
    "    raw[ny * 1024u + nx] = ushort(r | (gg << 5) | (b << 10) | (c.a >= 0.5 ? 0x8000u : 0u));\n"
    "}\n"
    /* Primitives, as gpu_gl_renderer.c's GEO and TEX programs. Blending and
     * the mask check read the destination in the shader (Apple GPUs keep
     * per-pixel submission order), so every mode is one pass. */
    "struct Vtx { float2 pos, uv; float4 col; uint flags, page, clut, twin; int4 lim; };\n"
    "struct U { float shift; uint scale, dither, pad; };\n"
    "struct VOut {\n"
    "    float4 position [[position]];\n"
    "    float2 uv [[center_no_perspective]];\n"
    "    float4 col [[center_no_perspective]];\n"
    "    float2 npos [[center_no_perspective]];\n"
    "    uint flags [[flat]]; uint page [[flat]]; uint clut [[flat]]; uint twin [[flat]];\n"
    "    int4 lim [[flat]];\n"
    "};\n"
    "vertex VOut draw_vs(uint vid [[vertex_id]], constant Vtx *v [[buffer(0)]],\n"
    "                    constant U &u [[buffer(1)]]) {\n"
    "    Vtx x = v[vid]; VOut o;\n"
    "    o.position = float4((x.pos.x + u.shift) / 512.0 - 1.0, 1.0 - (x.pos.y + u.shift) / 256.0, 0.0, 1.0);\n"
    "    o.uv = x.uv; o.col = x.col; o.npos = x.pos;\n"
    "    o.flags = x.flags; o.page = x.page; o.clut = x.clut; o.twin = x.twin; o.lim = x.lim;\n"
    "    return o;\n"
    "}\n"
    "static int vram_at(const device ushort *raw, int x, int y) {\n"
    "    return int(raw[uint(y & 511) * 1024u + uint(x & 1023)]);\n"
    "}\n"
    "static int fetch_texel(const device ushort *raw, VOut in, int u, int v) {\n"
    "    u &= 255; v &= 255;\n"
    "    int4 tw = int4(in.twin & 31u, (in.twin >> 5) & 31u, (in.twin >> 10) & 31u, (in.twin >> 15) & 31u);\n"
    "    if ((tw.x | tw.y) != 0) {\n"
    "        u = (u & ~(tw.x * 8)) | ((tw.z & tw.x) * 8);\n"
    "        v = (v & ~(tw.y * 8)) | ((tw.w & tw.y) * 8);\n"
    "    } else {\n"
    "        u = clamp(u, in.lim.x, in.lim.z);\n"
    "        v = clamp(v, in.lim.y, in.lim.w);\n"
    "    }\n"
    "    int px = int(in.page & 0xFFFFu), py = int(in.page >> 16);\n"
    "    int cx = int(in.clut & 0xFFFFu), cy = int(in.clut >> 16);\n"
    "    uint depth = (in.flags >> 2) & 3u;\n"
    "    if (depth == 0u) return vram_at(raw, cx + ((vram_at(raw, px + (u >> 2), py + v) >> ((u & 3) * 4)) & 0xF), cy);\n"
    "    if (depth == 1u) return vram_at(raw, cx + ((vram_at(raw, px + (u >> 1), py + v) >> ((u & 1) * 8)) & 0xFF), cy);\n"
    "    return vram_at(raw, px + u, py + v);\n"
    "}\n"
    "static float3 col5(int c) {\n"
    "    return float3(float(c & 31), float((c >> 5) & 31), float((c >> 10) & 31)) * (8.0 / 255.0);\n"
    "}\n"
    /* Flags: 1 textured, 2 raw texture, 4..8 depth, 16 dithered, 32..128
     * semi mode + 1 (0 = opaque), 256 set mask, 512 check mask. */
    "fragment float4 draw_fs(VOut in [[stage_in]], float4 dst [[color(0)]],\n"
    "                        const device ushort *raw [[buffer(0)]], constant U &u [[buffer(1)]]) {\n"
    "    uint f = in.flags;\n"
    "    if ((f & 512u) && dst.a >= 0.5) discard_fragment();\n"
    "    float3 rgb = in.col.rgb; uint stp = 0u;\n"
    "    if (f & 1u) {\n"
    "        int t = fetch_texel(raw, in, int(floor(in.uv.x)), int(floor(in.uv.y)));\n"
    "        if (t == 0) discard_fragment();\n"
    "        rgb = col5(t); stp = uint(t >> 15) & 1u;\n"
    "        if (!(f & 2u)) rgb = clamp(rgb * in.col.rgb * (255.0 / 128.0), 0.0, 1.0);\n"
    "    }\n"
    "    if ((f & 16u) && u.dither != 0u) {\n"
    "        const float m[16] = {-4.0,0.0,-3.0,1.0, 2.0,-2.0,3.0,-1.0, -3.0,1.0,-4.0,0.0, 3.0,-1.0,2.0,-2.0};\n"
    "        int2 p = (u.dither == 2u ? int2(in.position.xy) : int2(floor(in.npos))) & 3;\n"
    "        rgb = clamp(floor((rgb * 255.0 + m[p.y * 4 + p.x]) / 8.0), 0.0, 31.0) * (8.0 / 255.0);\n"
    "    }\n"
    "    uint semi = (f >> 5) & 7u;\n"
    "    if (semi != 0u && (!(f & 1u) || stp != 0u)) {\n"
    "        const float half_ = 128.0 / 255.0, quarter = 64.0 / 255.0;\n"
    "        if (semi == 1u) rgb = dst.rgb * half_ + rgb * half_;\n"
    "        else if (semi == 2u) rgb = dst.rgb + rgb;\n"
    "        else if (semi == 3u) rgb = dst.rgb - rgb;\n"
    "        else rgb = dst.rgb + rgb * quarter;\n"
    "        rgb = clamp(rgb, 0.0, 1.0);\n"
    "    }\n"
    "    return float4(rgb, (stp != 0u || (f & 256u)) ? 1.0 : 0.0);\n"
    "}\n";

static id<MTLDevice> s_device;
static id<MTLCommandQueue> s_queue;
static id<MTLComputePipelineState> s_fill, s_upload, s_gather, s_scatter, s_export, s_pack,
    s_export_present;
static id<MTLRenderPipelineState> s_draw;
static id<MTLRenderCommandEncoder> s_render;
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
/* Native rect drawn since the last pack: s_raw lacks it until packed. */
static int s_drawn, s_drawn_x0, s_drawn_y0, s_drawn_x1, s_drawn_y1;
static int s_semi_en, s_semi_mode, s_mod_r = 128, s_mod_g = 128, s_mod_b = 128, s_mod_raw;
static uint32_t s_twin;
static int s_dither_bit, s_dither_mode = 1;   /* as gpu_gl_renderer.c: native-grid dithering */

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
    /* Drawing reads the destination pixel in the fragment shader. */
    if (![s_device supportsFamily:MTLGPUFamilyApple1]) return 0;
    /* Fast math reorders the shaders' float arithmetic: a modulated texel
     * that lands on a half-step rounding tie then quantizes one step away
     * from OpenGL and the software renderer. */
    MTLCompileOptions *options = [MTLCompileOptions new];
    if (@available(macOS 15.0, *)) {
        options.mathMode = MTLMathModeSafe;
        options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
    } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled = NO;   /* the pre-macOS 15 equivalent */
#pragma clang diagnostic pop
    }
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
    s_pack = pipeline(library, @"pack");
    s_export_present = pipeline(library, @"export_present");
    MTLRenderPipelineDescriptor *rd = [MTLRenderPipelineDescriptor new];
    rd.vertexFunction = [library newFunctionWithName:@"draw_vs"];
    rd.fragmentFunction = [library newFunctionWithName:@"draw_fs"];
    rd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    s_draw = [s_device newRenderPipelineStateWithDescriptor:rd error:&error];
    s_queue = [s_device newCommandQueue];
    ok = s_fill && s_upload && s_gather && s_scatter && s_export && s_pack && s_export_present &&
         s_draw && s_queue;
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

static id<MTLCommandBuffer> command(void) {
    if (!s_command) s_command = [s_queue commandBuffer];
    return s_command;
}

static id<MTLComputeCommandEncoder> compute(void) {
    if (s_render) {
        [s_render endEncoding];
        s_render = nil;
    }
    if (!s_encoder)
        s_encoder = [command() computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    return s_encoder;
}

static void mark_stale(int y, int h);
static void dispatch_on(id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> p,
                        const MetalOp *op, NSUInteger w, NSUInteger h);

/* Brings s_raw up to date with what primitives drew. */
static void pack_drawn(void) {
    if (!s_drawn) return;
    s_drawn = 0;
    MetalOp op = {0};
    op.x = s_drawn_x0;
    op.y = s_drawn_y0;
    op.w = s_drawn_x1 - s_drawn_x0 + 1;
    op.h = s_drawn_y1 - s_drawn_y0 + 1;
    op.scale = (uint32_t)s_scale;
    id<MTLComputeCommandEncoder> e = compute();
    [e setTexture:s_hr atIndex:0];
    [e setBuffer:s_raw offset:0 atIndex:0];
    dispatch_on(e, s_pack, &op, (NSUInteger)op.w, (NSUInteger)op.h);
    mark_stale(op.y, op.h);
}

/* The compute encoder, with native VRAM current. */
static id<MTLComputeCommandEncoder> encoder(void) {
    pack_drawn();
    return compute();
}

static id<MTLRenderCommandEncoder> render(void) {
    if (!s_render) {
        if (s_encoder) {
            [s_encoder endEncoding];
            s_encoder = nil;
        }
        MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
        pass.colorAttachments[0].texture = s_hr;
        pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        s_render = [command() renderCommandEncoderWithDescriptor:pass];
        [s_render setRenderPipelineState:s_draw];
        [s_render setFragmentBuffer:s_raw offset:0 atIndex:0];
    }
    return s_render;
}

/* Submits queued work and waits for it. */
void metal_renderer_finish(void) {
    pack_drawn();
    if (s_render) {
        [s_render endEncoding];
        s_render = nil;
    }
    if (s_encoder) {
        [s_encoder endEncoding];
        s_encoder = nil;
    }
    if (!s_command) return;
    [s_command commit];
    [s_command waitUntilCompleted];
    if (s_command.status != MTLCommandBufferStatusCompleted)
        psx_fatal_halt("Metal renderer: a command buffer failed");
    s_command = nil;
    s_staging_used = 0;
}

static void dispatch_on(id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> p,
                        const MetalOp *op, NSUInteger w, NSUInteger h) {
    if (!w || !h) return;
    [e setComputePipelineState:p];
    [e setBytes:op length:sizeof *op atIndex:1];
    const NSUInteger tw = p.threadExecutionWidth;
    const NSUInteger th = p.maxTotalThreadsPerThreadgroup / tw;
    [e dispatchThreads:MTLSizeMake(w, h, 1) threadsPerThreadgroup:MTLSizeMake(tw, th, 1)];
}

/* Dispatches on the compute encoder the caller just took with encoder(). */
static void dispatch(id<MTLComputePipelineState> p, const MetalOp *op, NSUInteger w, NSUInteger h) {
    dispatch_on(s_encoder, p, op, w, h);
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
    if (s_ready) metal_renderer_finish();
    s_vram = vram;
    sw_renderer_init(vram);   /* the CPU mirror's transfers and reads, at 1x */
    /* Draw state starts as sw_renderer_init leaves it. */
    s_mask_set = s_mask_check = 0;
    s_semi_en = s_semi_mode = 0;
    s_mod_r = s_mod_g = s_mod_b = 128;
    s_mod_raw = 0;
    s_twin = 0;
    s_dither_bit = 0;
    s_drawn = 0;
    sw_get_draw_area(&s_clip[0], &s_clip[1], &s_clip[2], &s_clip[3]);
    s_ready = 0;
    if (!device_ready() || !allocate()) psx_fatal_halt("Metal renderer: initialization failed");
    memcpy(s_raw.contents, vram, (size_t)VRAM_W * VRAM_H * sizeof(uint16_t));
    queue_upload(0, 0, VRAM_W, VRAM_H, vram);
}

static int mb_scale(void) { return s_scale; }

static void mb_set_texture_filter(int bilinear) { sw_set_texture_filter(bilinear); }
static int mb_texture_filter(void) { return sw_texture_filter(); }
static void mb_set_semi(int enabled, int mode) {
    s_semi_en = enabled ? 1 : 0;
    s_semi_mode = mode & 3;
    sw_set_semi_transparency(enabled, mode);
}
static void mb_set_mask_bits(int set_bit, int check_bit) {
    s_mask_set = set_bit;
    s_mask_check = check_bit;
    sw_set_mask_bits(set_bit, check_bit);
}
static void mb_set_texture_window(uint32_t raw) {
    s_twin = raw & 0xFFFFFu;
    sw_set_texture_window(raw);
}
static void mb_set_dither(int enabled) { s_dither_bit = enabled ? 1 : 0; }
static void mb_set_color_modulation(int r, int g, int b, int raw) {
    s_mod_r = r;
    s_mod_g = g;
    s_mod_b = b;
    s_mod_raw = raw ? 1 : 0;
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

/* ---- Primitives ----------------------------------------------------------
 * As gpu_gl_renderer.c draws them: triangles in native VRAM coordinates,
 * shifted by half an internal pixel less 1/64 (see its u_shift comment),
 * colours in the k*8/255 basis, textures sampled from native VRAM with the
 * shared gpu_uv.h centre-sampling model. Blending, the mask check and the
 * mask bit happen in the fragment shader against the destination pixel. */

#include "gpu_uv.h"

typedef struct {
    float pos[2], uv[2];
    float col[4];
    uint32_t flags, page, clut, twin;
    int32_t lim[4];
} MetalVtx;   /* matches the shader's Vtx: 64 bytes */

typedef struct {
    float shift;
    uint32_t scale, dither, pad;
} MetalU;

enum {
    VF_TEXTURED = 1u, VF_RAW = 2u, VF_DITHER = 16u, VF_SET_MASK = 256u, VF_CHECK_MASK = 512u
};

static int semi_code(void) { return s_semi_en ? s_semi_mode : -1; }

/* Flags shared by every vertex of a primitive. */
static uint32_t prim_flags(int semi, int dither) {
    uint32_t f = 0;
    if (semi >= 0) f |= (uint32_t)((semi & 3) + 1) << 5;
    if (dither && s_dither_mode && s_dither_bit) f |= VF_DITHER;
    if (s_mask_set) f |= VF_SET_MASK;
    if (s_mask_check) f |= VF_CHECK_MASK;
    return f;
}

static int rects_meet(int ax0, int ay0, int ax1, int ay1, int bx0, int by0, int bx1, int by1) {
    return ax0 <= bx1 && bx0 <= ax1 && ay0 <= by1 && by0 <= ay1;
}

/* Native VRAM a textured primitive reads must include what earlier
 * primitives drew there: pack first when the page or CLUT row meets it. */
static void pack_if_sampling(int base_x, int base_y, int depth, int cx, int cy) {
    if (!s_drawn) return;
    const int width = depth == 0 ? 64 : depth == 1 ? 128 : 256;
    const int x0 = s_drawn_x0, y0 = s_drawn_y0, x1 = s_drawn_x1, y1 = s_drawn_y1;
    int hit = rects_meet(base_x, base_y, base_x + width - 1, base_y + 255, x0, y0, x1, y1);
    if (base_x + width > VRAM_W)   /* the page wraps past column 1023 */
        hit |= rects_meet(0, base_y, base_x + width - VRAM_W - 1, base_y + 255, x0, y0, x1, y1);
    if (depth < 2) {
        const int entries = depth == 0 ? 16 : 256;
        hit |= rects_meet(cx, cy, cx + entries - 1, cy, x0, y0, x1, y1);
        if (cx + entries > VRAM_W)
            hit |= rects_meet(0, cy, cx + entries - VRAM_W - 1, cy, x0, y0, x1, y1);
    }
    if (hit) pack_drawn();
}

/* Draws n vertices (triangles, or one line when line != 0) clipped to the
 * draw area. bx/by: the primitive's native bounding box. */
static void draw(const MetalVtx *v, int n, int line, int bx0, int by0, int bx1, int by1) {
    if (!s_ready || n <= 0) return;
    int x0 = s_clip[0], y0 = s_clip[1], x1 = s_clip[2], y1 = s_clip[3];
    if (x1 > VRAM_W - 1) x1 = VRAM_W - 1;
    if (y1 > VRAM_H - 1) y1 = VRAM_H - 1;
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (bx0 > x0) x0 = bx0;
    if (by0 > y0) y0 = by0;
    if (bx1 < x1) x1 = bx1;
    if (by1 < y1) y1 = by1;
    if (x0 > x1 || y0 > y1) return;
    const int S = s_scale;
    id<MTLRenderCommandEncoder> e = render();
    MTLScissorRect sc = {
        (NSUInteger)(s_clip[0] < 0 ? 0 : s_clip[0]) * S,
        (NSUInteger)(s_clip[1] < 0 ? 0 : s_clip[1]) * S, 0, 0
    };
    const int cw = (s_clip[2] > VRAM_W - 1 ? VRAM_W - 1 : s_clip[2]) - (int)(sc.x / S) + 1;
    const int ch = (s_clip[3] > VRAM_H - 1 ? VRAM_H - 1 : s_clip[3]) - (int)(sc.y / S) + 1;
    if (cw <= 0 || ch <= 0) return;
    sc.width = (NSUInteger)cw * S;
    sc.height = (NSUInteger)ch * S;
    [e setScissorRect:sc];
    MetalU u = {0.5f / (float)S - 1.0f / 64.0f, (uint32_t)S, (uint32_t)s_dither_mode, 0};
    [e setVertexBytes:v length:(NSUInteger)n * sizeof *v atIndex:0];
    [e setVertexBytes:&u length:sizeof u atIndex:1];
    [e setFragmentBytes:&u length:sizeof u atIndex:1];
    [e drawPrimitives:line ? MTLPrimitiveTypeLine : MTLPrimitiveTypeTriangle
          vertexStart:0 vertexCount:(NSUInteger)n];
    if (!s_drawn) {
        s_drawn = 1;
        s_drawn_x0 = x0; s_drawn_y0 = y0; s_drawn_x1 = x1; s_drawn_y1 = y1;
    } else {
        if (x0 < s_drawn_x0) s_drawn_x0 = x0;
        if (y0 < s_drawn_y0) s_drawn_y0 = y0;
        if (x1 > s_drawn_x1) s_drawn_x1 = x1;
        if (y1 > s_drawn_y1) s_drawn_y1 = y1;
    }
    gpu_vram_dirty_mark_rect(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
}

static void bounds(const int *xs, const int *ys, int n, int *b) {
    b[0] = b[2] = xs[0];
    b[1] = b[3] = ys[0];
    for (int i = 1; i < n; ++i) {
        if (xs[i] < b[0]) b[0] = xs[i];
        if (xs[i] > b[2]) b[2] = xs[i];
        if (ys[i] < b[1]) b[1] = ys[i];
        if (ys[i] > b[3]) b[3] = ys[i];
    }
}

static void color5(uint16_t c, float *out) {
    out[0] = (float)((c & 0x1F) << 3) / 255.0f;
    out[1] = (float)(((c >> 5) & 0x1F) << 3) / 255.0f;
    out[2] = (float)(((c >> 10) & 0x1F) << 3) / 255.0f;
    out[3] = 1.0f;
}

/* Flat or Gouraud geometry: triangles (n = 3) or a line (n = 2). */
static void geometry(const int *xs, const int *ys, const uint16_t *cs, int n, int dither) {
    MetalVtx v[6];
    memset(v, 0, sizeof v);
    const uint32_t f = prim_flags(semi_code(), dither);
    int b[4];
    bounds(xs, ys, n, b);
    if (n == 3) {
        for (int i = 0; i < 3; ++i) {
            v[i].pos[0] = (float)xs[i];
            v[i].pos[1] = (float)ys[i];
            color5(cs[i], v[i].col);
            v[i].flags = f;
        }
        draw(v, 3, 0, b[0], b[1], b[2], b[3]);
        return;
    }
    if (s_scale == 1) {   /* native: a line, as OpenGL's GL_LINES */
        for (int i = 0; i < 2; ++i) {
            v[i].pos[0] = (float)xs[i];
            v[i].pos[1] = (float)ys[i];
            color5(cs[i], v[i].col);
            v[i].flags = f;
        }
        draw(v, 2, 1, b[0], b[1], b[2], b[3]);
        return;
    }
    /* A quad one native pixel thick (gpu_gl_renderer.c line_to_quad). */
    const float x0 = (float)xs[0] + 0.5f, y0 = (float)ys[0] + 0.5f;
    const float x1 = (float)xs[1] + 0.5f, y1 = (float)ys[1] + 0.5f;
    const float dx = x1 - x0, dy = y1 - y0, ax = fabsf(dx), ay = fabsf(dy);
    float ex, ey, nx, ny;
    if (ax >= ay) {
        const float sx = dx < 0.0f ? -1.0f : 1.0f;
        ex = 0.5f * sx;
        ey = ax > 0.0f ? 0.5f * sx * dy / dx : 0.0f;
        nx = 0.0f;
        ny = 0.5f;
    } else {
        const float sy = dy < 0.0f ? -1.0f : 1.0f;
        ey = 0.5f * sy;
        ex = 0.5f * sy * dx / dy;
        nx = 0.5f;
        ny = 0.0f;
    }
    const float a[2] = {x0 - ex, y0 - ey}, bb[2] = {x1 + ex, y1 + ey};
    const float qx[6] = {a[0] - nx, bb[0] - nx, a[0] + nx, bb[0] - nx, a[0] + nx, bb[0] + nx};
    const float qy[6] = {a[1] - ny, bb[1] - ny, a[1] + ny, bb[1] - ny, a[1] + ny, bb[1] + ny};
    for (int i = 0; i < 6; ++i) {
        v[i].pos[0] = qx[i];
        v[i].pos[1] = qy[i];
        color5(cs[i & 1], v[i].col);
        v[i].flags = f;
    }
    draw(v, 6, 0, b[0], b[1], b[2], b[3]);
}

/* A textured triangle (gpu_gl_renderer.c gpu_textured_triangle, nearest
 * sampling). col: three rgb modulation colours in 0..1; lim: uv bounds, or
 * NULL to compute them with the centre-sampling compensation. */
static void textured(const int *xs, const int *ys, const int *us_in, const int *vs_in,
                     const float *col, uint16_t tp, uint16_t cx, uint16_t cy, int raw,
                     int semi, int dither, const int *lim) {
    int us[3] = {us_in[0], us_in[1], us_in[2]}, vs[3] = {vs_in[0], vs_in[1], vs_in[2]};
    int lim_buf[4];
    if (!lim) {
        psx_uv_tri_center_sample(xs, ys, us, vs, 0, lim_buf);
        lim = lim_buf;
    }
    const int base_x = (tp & 0xF) * 64, base_y = ((tp >> 4) & 1) * 256;
    int depth = (tp >> 7) & 3;
    if (depth > 2) depth = 2;
    pack_if_sampling(base_x, base_y, depth, cx, cy);
    uint32_t f = prim_flags(semi, dither && !raw) | VF_TEXTURED | ((uint32_t)depth << 2);
    if (raw) f |= VF_RAW;
    MetalVtx v[3];
    memset(v, 0, sizeof v);
    for (int i = 0; i < 3; ++i) {
        v[i].pos[0] = (float)xs[i];
        v[i].pos[1] = (float)ys[i];
        v[i].uv[0] = (float)us[i];
        v[i].uv[1] = (float)vs[i];
        v[i].col[0] = col[i * 3];
        v[i].col[1] = col[i * 3 + 1];
        v[i].col[2] = col[i * 3 + 2];
        v[i].col[3] = 1.0f;
        v[i].flags = f;
        v[i].page = (uint32_t)base_x | ((uint32_t)base_y << 16);
        v[i].clut = (uint32_t)cx | ((uint32_t)cy << 16);
        v[i].twin = s_twin;
        for (int k = 0; k < 4; ++k) v[i].lim[k] = lim[k];
    }
    int b[4];
    bounds(xs, ys, 3, b);
    draw(v, 3, 0, b[0], b[1], b[2], b[3]);
}

static void modulation(float *col) {
    for (int i = 0; i < 3; ++i) {
        col[i * 3] = (float)s_mod_r / 255.0f;
        col[i * 3 + 1] = (float)s_mod_g / 255.0f;
        col[i * 3 + 2] = (float)s_mod_b / 255.0f;
    }
}

static void textured_rect(int x, int y, int w, int h, int u0, int v0, int u1, int v1,
                          uint16_t cx, uint16_t cy, uint16_t tp) {
    if (w <= 0 || h <= 0) return;
    float col[9];
    modulation(col);
    int lim[4];
    psx_uv_rect_limits(u0, v0, u1, v1, lim);
    psx_uv_rect_mirror_offset(&u0, &v0, &u1, &v1);
    const int semi = semi_code();
    const int xs1[3] = {x, x + w, x}, ys1[3] = {y, y, y + h};
    const int us1[3] = {u0, u1, u0}, vs1[3] = {v0, v0, v1};
    textured(xs1, ys1, us1, vs1, col, tp, cx, cy, s_mod_raw, semi, 0, lim);
    const int xs2[3] = {x + w, x, x + w}, ys2[3] = {y, y + h, y + h};
    const int us2[3] = {u1, u0, u1}, vs2[3] = {v0, v1, v1};
    textured(xs2, ys2, us2, vs2, col, tp, cx, cy, s_mod_raw, semi, 0, lim);
}

static void mb_draw_flat_triangle(int x0, int y0, int x1, int y1, int x2, int y2, uint16_t c) {
    const int xs[3] = {x0, x1, x2}, ys[3] = {y0, y1, y2};
    const uint16_t cs[3] = {c, c, c};
    geometry(xs, ys, cs, 3, 0);
}
static void mb_draw_gouraud_triangle(int x0, int y0, uint16_t c0, int x1, int y1, uint16_t c1,
                                     int x2, int y2, uint16_t c2) {
    const int xs[3] = {x0, x1, x2}, ys[3] = {y0, y1, y2};
    const uint16_t cs[3] = {c0, c1, c2};
    geometry(xs, ys, cs, 3, 1);
}
static void mb_draw_textured_triangle(int x0, int y0, int u0, int v0, int x1, int y1, int u1,
                                      int v1, int x2, int y2, int u2, int v2, uint16_t cx,
                                      uint16_t cy, uint16_t tp) {
    const int xs[3] = {x0, x1, x2}, ys[3] = {y0, y1, y2};
    const int us[3] = {u0, u1, u2}, vs[3] = {v0, v1, v2};
    float col[9];
    modulation(col);
    textured(xs, ys, us, vs, col, tp, cx, cy, s_mod_raw, semi_code(), 1, NULL);
}
static void mb_draw_shaded_textured_triangle(int x0, int y0, int u0, int v0, uint32_t c0,
                                             int x1, int y1, int u1, int v1, uint32_t c1,
                                             int x2, int y2, int u2, int v2, uint32_t c2,
                                             uint16_t cx, uint16_t cy, uint16_t tp, int raw) {
    const int xs[3] = {x0, x1, x2}, ys[3] = {y0, y1, y2};
    const int us[3] = {u0, u1, u2}, vs[3] = {v0, v1, v2};
    const uint32_t cc[3] = {c0, c1, c2};
    float col[9];
    for (int i = 0; i < 3; ++i) {
        col[i * 3] = (float)(cc[i] & 0xFF) / 255.0f;
        col[i * 3 + 1] = (float)((cc[i] >> 8) & 0xFF) / 255.0f;
        col[i * 3 + 2] = (float)((cc[i] >> 16) & 0xFF) / 255.0f;
    }
    textured(xs, ys, us, vs, col, tp, cx, cy, raw, semi_code(), 1, NULL);
}
static void mb_draw_flat_rect(int x, int y, int w, int h, uint16_t c) {
    if (w <= 0 || h <= 0) return;
    const uint16_t cs[3] = {c, c, c};
    const int xs1[3] = {x, x + w, x}, ys1[3] = {y, y, y + h};
    geometry(xs1, ys1, cs, 3, 0);
    const int xs2[3] = {x + w, x, x + w}, ys2[3] = {y, y + h, y + h};
    geometry(xs2, ys2, cs, 3, 0);
}
static void mb_draw_textured_rect(int x, int y, int w, int h, int u, int v, uint16_t cx,
                                  uint16_t cy, uint16_t tp) {
    textured_rect(x, y, w, h, u, v, u + w, v + h, cx, cy, tp);
}
static void mb_draw_textured_rect_scaled(int x, int y, int w, int h, int u0, int v0, int u1,
                                         int v1, uint16_t cx, uint16_t cy, uint16_t tp) {
    textured_rect(x, y, w, h, u0, v0, u1, v1, cx, cy, tp);
}
static void mb_draw_line(int x0, int y0, int x1, int y1, uint16_t c) {
    const int xs[2] = {x0, x1}, ys[2] = {y0, y1};
    const uint16_t cs[2] = {c, c};
    geometry(xs, ys, cs, 2, 0);
}
static void mb_draw_shaded_line(int x0, int y0, uint16_t c0, int x1, int y1, uint16_t c1) {
    const int xs[2] = {x0, x1}, ys[2] = {y0, y1};
    const uint16_t cs[2] = {c0, c1};
    geometry(xs, ys, cs, 2, 1);
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

/* ---- Direct present ------------------------------------------------------
 * The window's present texture is an IOSurface the backend also wraps as a
 * Metal texture. Each frame the display decode writes it on the GPU; SDL's
 * Metal renderer samples the same surface, so no frame crosses to the CPU. */

static CVPixelBufferRef s_present_pb;
static id<MTLTexture> s_present_tex;

void *metal_renderer_present_pixel_buffer(int width, int height) {
    if (!device_ready() || width <= 0 || height <= 0) return NULL;
    if (s_present_pb && (int)CVPixelBufferGetWidth(s_present_pb) >= width &&
        (int)CVPixelBufferGetHeight(s_present_pb) >= height)
        return s_present_pb;
    if (s_ready) metal_renderer_finish();
    NSDictionary *attrs = @{
        (NSString *)kCVPixelBufferIOSurfacePropertiesKey: @{},
        (NSString *)kCVPixelBufferMetalCompatibilityKey: @YES,
    };
    CVPixelBufferRef pb = NULL;
    if (CVPixelBufferCreate(kCFAllocatorDefault, (size_t)width, (size_t)height,
                            kCVPixelFormatType_32BGRA, (__bridge CFDictionaryRef)attrs,
                            &pb) != kCVReturnSuccess || !pb)
        return NULL;
    IOSurfaceRef surface = CVPixelBufferGetIOSurface(pb);
    MTLTextureDescriptor *d =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                           width:(NSUInteger)width
                                                          height:(NSUInteger)height
                                                       mipmapped:NO];
    d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    id<MTLTexture> tex = surface ? [s_device newTextureWithDescriptor:d iosurface:surface plane:0]
                                 : nil;
    if (!tex) {
        CVPixelBufferRelease(pb);
        return NULL;
    }
    if (s_present_pb) CVPixelBufferRelease(s_present_pb);
    s_present_pb = pb;
    s_present_tex = tex;
    return s_present_pb;
}

int metal_renderer_present_direct(int dx, int dy, int dw, int dh, int pad_row) {
    if (!s_ready || !s_present_tex || dw <= 0 || dh <= 0) return 0;
    const int S = s_scale;
    if ((NSUInteger)(dw * S) > s_present_tex.width || (NSUInteger)(dh * S) > s_present_tex.height)
        return 0;
    MetalOp op = {0};
    op.x = dx & (VRAM_W - 1);
    op.y = dy & (VRAM_H - 1);
    op.w = dw;
    op.h = dh;
    op.scale = (uint32_t)S;
    op.flags = pad_row ? 1u : 0u;
    id<MTLComputeCommandEncoder> e = encoder();
    [e setTexture:s_hr atIndex:0];
    [e setTexture:s_present_tex atIndex:1];
    dispatch(s_export_present, &op, (NSUInteger)dw * S, (NSUInteger)dh * S + (pad_row ? 1u : 0u));
    /* SDL samples the surface on its own queue: the image must be complete. */
    metal_renderer_finish();
    return 1;
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
    .set_dither = mb_set_dither,
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
