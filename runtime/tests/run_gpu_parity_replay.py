#!/usr/bin/env python3
"""Run test_gpu_parity_replay.c on a seeded random renderer recording.

The recording comes from gen_gr_random.py (no retail payload) with display
reads at 2x. It is replayed twice on a hidden OpenGL context; both runs must
replay every op and capture the same frames, so a frame set from one run is a
stable reference for another backend. Any extra arguments go to
run_gl_render_thread.py (--cc, --sdl-include, --sdl-library). Exits 77 when
the host has no OpenGL context, like the other GL fixtures.
"""

import argparse
import pathlib
import random
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_gr_random  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--seed", type=int, default=7)
    args, runner_args = parser.parse_known_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # A seed whose opening scale is 2 keeps the run short and upscaled.
    seed = args.seed
    while random.Random(seed).choice((1, 2, 4)) != 2:
        seed += 1
    recording = out / f"random-{seed}.grr"
    recording.write_bytes(gen_gr_random.recording(seed, final_displays=8))
    tables = []
    for run in ("a", "b"):
        frames = out / f"frames-{run}"
        frames.mkdir(exist_ok=True)
        r = subprocess.run([sys.executable, str(HERE / "run_gl_render_thread.py"), "--single",
                            "--fixture", str(HERE / "test_gpu_parity_replay.c"),
                            "--output", str(out / "build"), *runner_args,
                            "--probe-args", str(recording), str(frames), "0"])
        if r.returncode:
            return r.returncode
        tables.append((frames / "frames.tsv").read_text())
    if tables[0] != tables[1] or tables[0].count("\n") < 2:
        print("FAIL the two OpenGL replays captured different frames, or none")
        return 1
    print(f"PASS {tables[0].count(chr(10)) - 1} frames identical across two replays")
    return 0


if __name__ == "__main__":
    sys.exit(main())
