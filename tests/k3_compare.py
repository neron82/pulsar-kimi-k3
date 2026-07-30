#!/usr/bin/env python3
"""Compare opt-in PULSAR_K3_COMPARE_DIR snapshots from three K3 runs."""
import math
import struct
import sys
from pathlib import Path


def load(path):
    data = path.read_bytes()
    return list(struct.unpack(f"<{len(data) // 4}f", data))


def metrics(a, b, tol=1e-3):
    d = [x - y for x, y in zip(a, b)]
    aa = [abs(x) for x in d]
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return (max(aa, default=0.0), sum(aa) / max(1, len(aa)),
            math.sqrt(sum(x * x for x in d) / max(1, len(d))),
            sum(x * y for x, y in zip(a, b)) / max(na * nb, 1e-30),
            nb / max(na, 1e-30), next((i for i, x in enumerate(aa) if x > tol), None),
            sum(not math.isfinite(x) for x in a), sum(not math.isfinite(x) for x in b))


def fmt(label, a, b):
    m = metrics(a, b)
    print(f"{label}: max={m[0]:.6e} mean={m[1]:.6e} rms={m[2]:.6e} "
          f"cosine={m[3]:.9f} norm_ratio={m[4]:.9f} first_tol={m[5]} "
          f"nan_inf=({m[6]},{m[7]})")


def main(root):
    root = Path(root)
    names = ("cpu", "cpu-q8", "cuda")
    dirs = {n: root / n for n in names}
    files = sorted({p.name for d in dirs.values() if d.exists() for p in d.glob("*.f32")})
    progression = []
    for name in files:
        values = {n: load(dirs[n] / name) for n in names if (dirs[n] / name).exists()}
        if len(values) != 3:
            continue
        if name.startswith("layer_") and name.endswith("_hidden.f32"):
            layer = int(name.split("_")[1])
            progression.append((layer, metrics(values["cpu"], values["cpu-q8"]),
                                metrics(values["cpu-q8"], values["cuda"]),
                                metrics(values["cpu"], values["cuda"])))
        print(f"{name}:")
        fmt("  F32-vs-Q8", values["cpu"], values["cpu-q8"])
        fmt("  Q8-vs-CUDA", values["cpu-q8"], values["cuda"])
        fmt("  F32-vs-CUDA", values["cpu"], values["cuda"])
        if name == "final_000_logits.f32":
            top = {n: sorted(range(len(v)), key=v.__getitem__, reverse=True)[:10] for n, v in values.items()}
            for pair in (("cpu", "cpu-q8"), ("cpu-q8", "cuda"), ("cpu", "cuda")):
                overlap = len(set(top[pair[0]]) & set(top[pair[1]]))
                print(f"  top10 {pair[0]}-vs-{pair[1]}={overlap}/10")
            for n in names:
                ordered = sorted(values[n], reverse=True)
                print(f"  {n}: token={top[n][0]} margin={ordered[0] - ordered[1]:.6e}")
    if progression:
        print("layer | F32-vs-Q8 cosine | Q8-vs-CUDA cosine | F32-vs-CUDA cosine")
        for layer, f32_q8, q8_cuda, f32_cuda in progression:
            print(f"{layer:5d} | {f32_q8[3]:17.9f} | {q8_cuda[3]:18.9f} | {f32_cuda[3]:18.9f}")
        for label, index in (("F32-vs-Q8", 1), ("Q8-vs-CUDA", 2)):
            first = next((layer for layer, *pairs in progression
                          if pairs[index - 1][3] < 0.999999 or pairs[index - 1][0] > 1e-3), None)
            print(f"first material {label} divergence layer: {first}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} SNAPSHOT_DIR")
    main(sys.argv[1])
