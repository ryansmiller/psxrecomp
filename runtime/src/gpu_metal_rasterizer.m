#import <Metal/Metal.h>
#include <mach/vm_page_size.h>

#include "gpu_metal_rasterizer.h"
#include "psx_metal_timing.h"
#include <stdlib.h>
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

static NSString *const batch_shader =
    @"#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct Primitive { ushort color; uchar blend; uchar flags; };\n"
    "struct Entry { uint x, width, primitive; };\n"
    "struct Row { uint pixel, x, width, first, count; };\n"
    "kernel void raster_batch(device ushort *pixels [[buffer(0)]],\n"
    "                         constant Row *rows [[buffer(1)]],\n"
    "                         constant Entry *entries [[buffer(2)]],\n"
    "                         constant Primitive *primitives [[buffer(3)]],\n"
    "                         device atomic_uint *dirty [[buffer(4)]],\n"
    "                         uint2 xy [[thread_position_in_grid]]) {\n"
    "    Row row = rows[xy.y];\n"
    "    if (xy.x >= row.width) return;\n"
    "    uint x = row.x + xy.x;\n"
    "    uint index = row.pixel + xy.x;\n"
    "    uint back = pixels[index];\n"
    "    bool wrote = false;\n"
    "    for (uint i = row.first; i < row.first + row.count; ++i) {\n"
    "        Entry entry = entries[i];\n"
    "        if (x - entry.x >= entry.width) continue;\n"
    "        Primitive p = primitives[entry.primitive];\n"
    "        if ((p.flags & 1u) && (back & 0x8000u)) continue;\n"
    "        uint result = p.color;\n"
    "        if (p.blend < 4) {\n"
    "            result = 0;\n"
    "            for (uint shift = 0; shift < 15; shift += 5) {\n"
    "                int b = (back >> shift) & 31u;\n"
    "                int f = (uint(p.color) >> shift) & 31u;\n"
    "                int c;\n"
    "                switch (p.blend) {\n"
    "                    case 0: c = (b + f) >> 1; break;\n"
    "                    case 1: c = min(b + f, 31); break;\n"
    "                    case 2: c = max(b - f, 0); break;\n"
    "                    default: c = min(b + (f >> 2), 31); break;\n"
    "                }\n"
    "                result |= uint(c) << shift;\n"
    "            }\n"
    "        }\n"
    "        back = result | ((p.flags & 2u) ? 0x8000u : 0u);\n"
    "        wrote = wrote || (p.flags & 4u);\n"
    "    }\n"
    "    pixels[index] = ushort(back);\n"
    "    if (wrote) atomic_store_explicit(&dirty[xy.y], 1u, memory_order_relaxed);\n"
    "}\n";

static id<MTLComputePipelineState> batch_pipeline;
static id<MTLBuffer> batch_pixels, batch_rows, batch_entries, batch_primitives, batch_dirty;

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
        id<MTLLibrary> batch_library =
            [device newLibraryWithSource:batch_shader options:nil error:&error];
        id<MTLFunction> batch_function = [batch_library newFunctionWithName:@"raster_batch"];
        if (batch_function)
            batch_pipeline = [device newComputePipelineStateWithFunction:batch_function
                                                                   error:&error];
        if (pipeline && batch_pipeline)
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

typedef struct {
    uint32_t pixel, x, width, first, count;
} PackedRow;

/* Shared buffers that use renderer memory in place. Apple GPUs share memory
 * with the CPU, so a page-aligned target needs no copies; the CPU only waits
 * for the command buffer before touching it again. */
enum { WRAPPED_TARGETS = 4 };
static struct {
    const void *memory;
    size_t bytes;
    id<MTLBuffer> buffer;
} wrapped[WRAPPED_TARGETS];
static unsigned wrapped_next;
static uint64_t zero_copy_flushes, copied_flushes;

void psx_metal_forget_memory(const void *memory) {
    for (int i = 0; i < WRAPPED_TARGETS; ++i)
        if (wrapped[i].memory == memory) {
            wrapped[i].memory = NULL;
            wrapped[i].bytes = 0;
            wrapped[i].buffer = nil;
        }
}

static id<MTLBuffer> wrap_target(void *memory, size_t bytes) {
    if ((uintptr_t)memory % vm_page_size || bytes % vm_page_size) return nil;
    for (int i = 0; i < WRAPPED_TARGETS; ++i)
        if (wrapped[i].memory == memory && wrapped[i].bytes == bytes) return wrapped[i].buffer;
    id<MTLBuffer> buffer = [device newBufferWithBytesNoCopy:memory
                                                     length:bytes
                                                    options:MTLResourceStorageModeShared
                                                deallocator:nil];
    if (!buffer) return nil;
    const unsigned slot = wrapped_next++ % WRAPPED_TARGETS;
    wrapped[slot].memory = memory;
    wrapped[slot].bytes = bytes;
    wrapped[slot].buffer = buffer;
    return buffer;
}

void psx_metal_flush_counts(uint64_t *zero_copy, uint64_t *copied) {
    *zero_copy = zero_copy_flushes;
    *copied = copied_flushes;
}

typedef struct {
    uint32_t x, width, primitive;
} PackedEntry;

int psx_metal_draw_batch(uint16_t *buf, int pitch, int rows,
                         const PsxMetalPrimitive *primitives,
                         uint32_t primitive_count,
                         const PsxMetalEntry *entries, uint32_t entry_count,
                         unsigned char *dirty_rows) {
    @autoreleasepool {
        if (!entry_count || rows <= 0 || !initialize()) return 0;
        /* Stable counting sort by row keeps each row's primitives in draw order. */
        uint32_t *row_counts = calloc((size_t)rows + 1u, sizeof *row_counts);
        if (!row_counts) return 0;
        for (uint32_t i = 0; i < entry_count; ++i) {
            if (entries[i].row >= (uint32_t)rows ||
                entries[i].primitive >= primitive_count) {
                free(row_counts);
                return 0;
            }
            row_counts[entries[i].row + 1u]++;
        }
        uint32_t row_count = 0;
        for (int y = 0; y < rows; ++y)
            if (row_counts[y + 1]) ++row_count;
        for (int y = 0; y < rows; ++y)
            row_counts[y + 1] += row_counts[y];

        if (!reserve(&batch_entries, entry_count * sizeof(PackedEntry)) ||
            !reserve(&batch_rows, row_count * sizeof(PackedRow)) ||
            !reserve(&batch_primitives, primitive_count * sizeof(PsxMetalPrimitive)) ||
            !reserve(&batch_dirty, row_count * sizeof(uint32_t))) {
            free(row_counts);
            return 0;
        }
        PackedEntry *sorted = batch_entries.contents;
        uint32_t *cursor = malloc((size_t)rows * sizeof *cursor);
        if (!cursor) {
            free(row_counts);
            return 0;
        }
        memcpy(cursor, row_counts, (size_t)rows * sizeof *cursor);
        for (uint32_t i = 0; i < entry_count; ++i) {
            const PsxMetalEntry e = entries[i];
            sorted[cursor[e.row]++] = (PackedEntry){e.x, e.width, e.primitive};
        }
        free(cursor);

        /* Each touched row uploads only the extent its spans cover. */
        id<MTLBuffer> target = wrap_target(buf, (size_t)rows * (size_t)pitch * sizeof(uint16_t));
        PackedRow *packed = batch_rows.contents;
        uint32_t *row_of = malloc((size_t)row_count * sizeof *row_of);
        if (!row_of) {
            free(row_counts);
            return 0;
        }
        uint32_t pixel_count = 0, max_width = 0, r = 0;
        for (int y = 0; y < rows; ++y) {
            const uint32_t first = row_counts[y], count = row_counts[y + 1] - first;
            if (!count) continue;
            uint32_t left = UINT32_MAX, right = 0;
            for (uint32_t i = first; i < first + count; ++i) {
                if (sorted[i].x < left) left = sorted[i].x;
                if (sorted[i].x + sorted[i].width > right) right = sorted[i].x + sorted[i].width;
            }
            const uint32_t pixel = target ? (uint32_t)y * (uint32_t)pitch + left : pixel_count;
            packed[r] = (PackedRow){pixel, left, right - left, first, count};
            row_of[r++] = (uint32_t)y;
            pixel_count += right - left;
            if (right - left > max_width) max_width = right - left;
        }
        free(row_counts);
        if (!target && !reserve(&batch_pixels, pixel_count * sizeof(uint16_t))) {
            free(row_of);
            return 0;
        }
        uint16_t *data = target ? NULL : batch_pixels.contents;
        for (uint32_t i = 0; data && i < row_count; ++i)
            memcpy(data + packed[i].pixel, buf + (size_t)row_of[i] * pitch + packed[i].x,
                   packed[i].width * sizeof(*data));
        memcpy(batch_primitives.contents, primitives,
               primitive_count * sizeof(PsxMetalPrimitive));
        memset(batch_dirty.contents, 0, row_count * sizeof(uint32_t));

        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!encoder) {
            free(row_of);
            return 0;
        }
        [encoder setComputePipelineState:batch_pipeline];
        [encoder setBuffer:(target ? target : batch_pixels) offset:0 atIndex:0];
        [encoder setBuffer:batch_rows offset:0 atIndex:1];
        [encoder setBuffer:batch_entries offset:0 atIndex:2];
        [encoder setBuffer:batch_primitives offset:0 atIndex:3];
        [encoder setBuffer:batch_dirty offset:0 atIndex:4];
        [encoder dispatchThreads:MTLSizeMake(max_width, row_count, 1)
           threadsPerThreadgroup:MTLSizeMake(batch_pipeline.threadExecutionWidth, 1, 1)];
        [encoder endEncoding];
        [command commit];
        static PsxMetalTimingStream wait_timing;
        uint64_t wait_begin = psx_metal_timing_begin(&wait_timing, "gpu-batch-wait");
        [command waitUntilCompleted];
        if (wait_begin)
            psx_metal_timing_record(&wait_timing, wait_begin, psx_metal_timing_now(),
                                    primitive_count, pixel_count);
        if (command.status != MTLCommandBufferStatusCompleted) {
            free(row_of);
            return 0;
        }

        const uint32_t *dirty = batch_dirty.contents;
        for (uint32_t i = 0; i < row_count; ++i) {
            if (data)
                memcpy(buf + (size_t)row_of[i] * pitch + packed[i].x, data + packed[i].pixel,
                       packed[i].width * sizeof(*data));
            if (dirty_rows && dirty[i]) dirty_rows[row_of[i]] = 1;
        }
        free(row_of);
        completed_draws += primitive_count;
        ++*(target ? &zero_copy_flushes : &copied_flushes);
        return 1;
    }
}
