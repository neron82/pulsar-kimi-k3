# K3 CUDA Device Assignment

Run from the repository root after building `pulsar-cli`:

```bash
nvidia-smi -L

PULSAR_GPU=0 PULSAR_PROFILE=1 \
  ./target/release/pulsar-cli \
  -m /home/neron/models/kimi-k3/Q2_K/Kimi-K3-Q2_K-00001-of-00024.gguf \
  --tokens 1 -n 1 --ctx 16

PULSAR_GPU=1 PULSAR_PROFILE=1 \
  ./target/release/pulsar-cli \
  -m /home/neron/models/kimi-k3/Q2_K/Kimi-K3-Q2_K-00001-of-00024.gguf \
  --tokens 1 -n 1 --ctx 16
```

`PULSAR_GPU` is a process-local CUDA index. `CUDA_VISIBLE_DEVICES` is applied
by CUDA first, so remapping changes the valid process-local indices and the
startup UUID/name identifies the physical GPU unambiguously. For example:

```bash
CUDA_VISIBLE_DEVICES=1 PULSAR_GPU=0 PULSAR_PROFILE=1 \
  ./target/release/pulsar-cli -m /path/to/k3.gguf --tokens 1 -n 1 --ctx 16
```

This resolves process-local device 0 to the physical GPU shown as host device
1 by `nvidia-smi`. An index outside the remapped visible range fails during
startup with an explicit `PULSAR_GPU` error.
