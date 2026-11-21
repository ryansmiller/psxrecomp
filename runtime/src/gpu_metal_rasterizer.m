#import <Metal/Metal.h>

#include "gpu_metal_rasterizer.h"
#include "psx_metal_timing.h"
#include <string.h>

typedef struct {
    uint32_t offset, width;
} PackedSpan;

static id<MTLDevice> device;
static id<MTLCommandQueue> queue;
static id<MTLComputePipelineState> pipeline;
static id<MTLBuffer> pixels;
static id<MTLBuffer> span_buffer;
static uint64_t completed_draws;

static NSString *const shader =
    @"#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct Span { uint offset, width; };\n"
    "kernel void raster_spans(device ushort *pixels [[buffer(0)]],\n"
    "                         constant Span *spans [[buffer(1)]],\n"
    "                         constant uint4 &state [[buffer(2)]],\n"
    "                         uint2 xy [[thread_position_in_grid]]) {\n"
    "    Span span = spans[xy.y];\n"
    "    if (xy.x >= span.width) return;\n"
    "    uint index = span.offset + xy.x;\n"
    "    uint back = pixels[index];\n"
    "    if (state.z && (back & 0x8000u)) return;\n"
    "    uint result = state.x;\n"
    "    if (state.y < 4) {\n"
    "        result = 0;\n"
    "        for (uint shift = 0; shift < 15; shift += 5) {\n"
    "            int b = (back >> shift) & 31u;\n"
    "            int f = (state.x >> shift) & 31u;\n"
    "            int c;\n"
    "            switch (state.y) {\n"
    "                case 0: c = (b + f) >> 1; break;\n"
    "                case 1: c = min(b + f, 31); break;\n"
    "                case 2: c = max(b - f, 0); break;\n"
    "                default: c = min(b + (f >> 2), 31); break;\n"
    "            }\n"
    "            result |= uint(c) << shift;\n"
    "        }\n"
    "    }\n"
    "    pixels[index] = ushort(result | state.w);\n"
    "}\n";

static int initialize(void) {
    static int attempted;
    if (!attempted) {
        attempted = 1;
        device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:shader options:nil error:&error];
        id<MTLFunction> function = [library newFunctionWithName:@"raster_spans"];
        if (function)
            pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (pipeline)
            queue = [device newCommandQueue];
        if (!queue)
            NSLog(@"PSX Metal rasterizer unavailable: %@", error ?: @"no usable Metal device or command queue");
    }
    return queue != nil;
}

static int reserve(id<MTLBuffer> __strong *buffer, NSUInteger size) {
    if ((*buffer).length < size)
        *buffer = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
    return *buffer != nil;
}

int psx_metal_draw_spans(uint16_t *rows, int pitch, const PsxMetalSpan *spans,
                         int height, uint16_t color, int blend,
                         int check_mask, int set_mask) {
    @autoreleasepool {
        uint32_t count = 0, max_width = 0;
        for (int y = 0; y < height; ++y) {
            uint32_t width = spans[y].width;
            count += width;
            if (width > max_width) max_width = width;
        }
        if (!count || !initialize() ||
            !reserve(&pixels, count * sizeof(uint16_t)) ||
            !reserve(&span_buffer, height * sizeof(PackedSpan)))
            return 0;

        PackedSpan *packed = span_buffer.contents;
        uint16_t *data = pixels.contents;
        uint32_t offset = 0;
        for (int y = 0; y < height; ++y) {
            PsxMetalSpan span = spans[y];
            packed[y] = (PackedSpan){offset, span.width};
            memcpy(data + offset, rows + (size_t)y * pitch + span.x,
                   span.width * sizeof(*data));
            offset += span.width;
        }

        uint32_t state[] = {color, (uint32_t)blend, check_mask != 0,
                            set_mask ? 0x8000u : 0u};
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!encoder) return 0;
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:pixels offset:0 atIndex:0];
        [encoder setBuffer:span_buffer offset:0 atIndex:1];
        [encoder setBytes:state length:sizeof(state) atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(max_width, height, 1)
           threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth, 1, 1)];
        [encoder endEncoding];
        [command commit];
        static PsxMetalTimingStream wait_timing;
        uint64_t wait_begin = psx_metal_timing_begin(&wait_timing, "gpu-wait");
        [command waitUntilCompleted];
        if (wait_begin)
            psx_metal_timing_record(&wait_timing, wait_begin, psx_metal_timing_now(),
                                    command.status, count);
        if (command.status != MTLCommandBufferStatusCompleted) return 0;

        for (int y = 0; y < height; ++y)
            memcpy(rows + (size_t)y * pitch + spans[y].x, data + packed[y].offset,
                   packed[y].width * sizeof(*data));
        if (++completed_draws == 1)
            NSLog(@"PSX Metal rasterizer completed its first flat primitive on %@", device.name);
        return 1;
    }
}

uint64_t psx_metal_completed_draws(void) {
    return completed_draws;
}
