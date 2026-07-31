#!/usr/bin/env python3
"""Compare opt-in PULSAR_K3_COMPARE_DIR snapshots from three K3 runs."""
import math
import hashlib
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


def ordered_bits(v):
    bits = struct.unpack("<I", struct.pack("<f", v))[0]
    return bits ^ ((-(bits >> 31)) & 0x7fffffff)


def exact_fmt(label, a, b):
    d = [x - y for x, y in zip(a, b)]
    ad = [abs(x) for x in d]
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    first = next((i for i, (x, y) in enumerate(zip(a, b))
                  if struct.pack("<f", x) != struct.pack("<f", y)), None)
    ulp = max((abs(ordered_bits(x) - ordered_bits(y)) for x, y in zip(a, b)), default=0)
    thresholds = {t: next((i for i, x in enumerate(ad) if x > t), None)
                  for t in (1e-8, 1e-7, 1e-6, 1e-5)}
    print(f"{label}: shape=[{len(a)}] max={max(ad, default=0):.9e} "
          f"mean={sum(ad) / max(1, len(ad)):.9e} "
          f"rms={math.sqrt(sum(x * x for x in d) / max(1, len(d))):.9e} "
          f"cosine={sum(x * y for x, y in zip(a, b)) / max(na * nb, 1e-30):.12f} "
          f"norm_ratio={nb / max(na, 1e-30):.12f} first={first} "
          f"thresholds={thresholds} max_ulp={ulp} "
          f"nan_inf=({sum(not math.isfinite(x) for x in a)},"
          f"{sum(not math.isfinite(x) for x in b)})")
    if first is not None:
        lo, hi = max(0, first - 2), min(len(a), first + 3)
        for i in range(lo, hi):
            print(f"  index={i} cpu={a[i]:.9e} cuda={b[i]:.9e} "
                  f"ulp={abs(ordered_bits(a[i]) - ordered_bits(b[i]))}")


def expert_report(root):
    """Compare the focused CPU-Q8/CUDA expert-498 artifacts."""
    root = Path(root)
    dirs = {name: root / name for name in ("cpu-q8", "cuda")}
    cpu_prefix = "expert_001_cpu_q8_"
    cuda_prefix = "expert_001_cuda_"
    cpu = {p.name[len(cpu_prefix):]: p for p in dirs["cpu-q8"].glob("expert_001_cpu_q8_*.f32")}
    cuda = {p.name[len(cuda_prefix):]: p for p in dirs["cuda"].glob("expert_001_cuda_*.f32")}
    print("focused expert 498: layer=1 token=0 rank=0 global_id=498 local_slot=0")
    for name in sorted(cpu.keys() & cuda.keys()):
        print(f"{name}:")
        exact_fmt("  CPU-Q8 vs CUDA", load(cpu[name]), load(cuda[name]))

    for cpu_path in dirs["cpu-q8"].glob("*.bin"):
        cuda_name = cpu_path.name.replace("cpu_q8_", "cuda_", 1)
        cuda_path = dirs["cuda"] / cuda_name
        if not cuda_path.exists():
            continue
        left, right = cpu_path.read_bytes(), cuda_path.read_bytes()
        first = next((i for i, (x, y) in enumerate(zip(left, right)) if x != y), None)
        print(f"{cpu_path.name}: bytes={len(left)} "
              f"sha256_cpu={hashlib.sha256(left).hexdigest()} "
              f"sha256_cuda={hashlib.sha256(right).hexdigest()} "
              f"differing={sum(x != y for x, y in zip(left, right))} "
              f"first_byte={first}")
        if first is not None:
            block, in_block = divmod(first, 292)
            field = "scale" if in_block < 4 else "qs" if in_block < 260 else "bsums"
            print(f"  first_block={block} field={field} offset_in_block={in_block} "
                  f"values=({left[first]},{right[first]})")

    for name in ("moe_001_routed_moe.f32", "layer_001_hidden.f32"):
        left, right = dirs["cpu-q8"] / name, dirs["cuda"] / name
        if left.exists() and right.exists():
            print(f"{name}:")
            exact_fmt("  CPU-Q8 vs CUDA", load(left), load(right))


def main(root):
    root = Path(root)
    expert_report(root)
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
