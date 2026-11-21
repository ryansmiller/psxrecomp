#ifndef PSX_METAL_TIMING_H
#define PSX_METAL_TIMING_H

#ifdef __APPLE__
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Separate streams share the same monotonic clock. Each stream has one writer.
 * Records are four native-endian uint64_t values: start, end, tag, value. */
typedef struct {
    FILE *file;
    int initialized;
} PsxMetalTimingStream;

static inline uint64_t psx_metal_timing_now(void) {
    return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
}

static inline uint64_t psx_metal_timing_begin(PsxMetalTimingStream *stream,
                                               const char *name) {
    if (!stream->initialized) {
        stream->initialized = 1;
        const char *dir = getenv("PSX_METAL_TIMING_DIR");
        if (dir && *dir) {
            char path[4096];
            int len = snprintf(path, sizeof(path), "%s/%s.bin", dir, name);
            if (len > 0 && (size_t)len < sizeof(path))
                stream->file = fopen(path, "wb");
            if (stream->file)
                setvbuf(stream->file, NULL, _IOFBF, 65536);
            else
                fprintf(stderr, "Could not open Metal timing stream %s\n", name);
        }
    }
    return stream->file ? psx_metal_timing_now() : 0;
}

static inline void psx_metal_timing_record(PsxMetalTimingStream *stream,
                                           uint64_t begin, uint64_t end,
                                           uint64_t tag, uint64_t value) {
    if (begin) {
        const uint64_t record[] = {begin, end, tag, value};
        if (fwrite(record, sizeof(record), 1, stream->file) != 1) {
            fprintf(stderr, "Metal timing stream write failed\n");
            fclose(stream->file);
            stream->file = NULL;
        }
    }
}
#endif
#endif
