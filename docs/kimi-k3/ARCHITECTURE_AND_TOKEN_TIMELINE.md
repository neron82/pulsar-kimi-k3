# Pulsar Kimi-K3: Datenpfad, Speicherhierarchie und Token-Zeitstrahl

Status: Arbeitsstand nach einem aktuellen Q8-Hotspot-Smoke-Run. Messwerte sind Rohlog-basiert; nicht gemessene Größen sind ausdrücklich als solche markiert.

## Kurzfassung

Pulsar ist aktuell ein sequenzieller Single-Token-Decodepfad für Kimi-K3:

- 24 GGUF-Shards werden als eine virtuelle GGUF-Datei behandelt.
- Kimi-K3 wird mit 93 Layern und 896 Routing-Experten geladen; Top-k ist 16.
- Die Expertentensoren Gate/Up/Down bleiben als Q2_K/Q2_K/Q3_K-Slabs im GGUF und werden über ein Tier-/Staging-System aufgelöst.
- Zwei große zuvor F32-gemappte Absorbed-Weights (`ffn_latent_up`, `ffn_down_shexp`) werden inzwischen beim Laden in Q8_0 konvertiert.
- Direct-Q2_K/Q3_K-MatMul vermeidet die frühere Q8-Konvertierung der quantisierten K3-Projektionsgewichte.
- KDA und Kimi-K3-Decode sind derzeit sequenziell; Prefill wird für K3 tokenweise ausgeführt.
- Ein aktueller Smoke-Run erzeugte einen echten Token (`id 58810`), aber nur mit 0,03 tok/s. Das Ziel von 0,33 tok/s ist noch nicht erreicht.

## Hardware und Gerätezuordnung

Live geprüft mit `nvidia-smi -L`:

| CUDA-Gerät | Hardware |
|---:|---|
| 0 | NVIDIA GeForce RTX 3060 Ti, 8 GiB |
| 1 | NVIDIA GeForce RTX 3090, 24 GiB |

Die aktuellen K3-Testkommandos setzen `PULSAR_GPU=1` und loggen `using CUDA device 1`; dieser Pfad läuft damit auf der RTX 3090. Andere laufende Services, insbesondere `llama-server` und ComfyUI, können gleichzeitig Speicher auf beiden Geräten belegen. GPU-Auslastung muss deshalb immer während des K3-Prozesses und nicht nach dessen Ende gemessen werden.

Wichtig: Der generische `ExpertTier`-Builder kann bei vorhandener Warm-Census-Datei zusätzlich einen Tier auf CUDA-Gerät 0 anlegen. Der aktuelle K3-spezifische Resolver nutzt diese generische Tier-Liste jedoch noch nicht; K3 verwendet wirksam den Primary-DeviceSlabCache auf Gerät 1 und das Primary-Staging. Ein Log wie `expert tier on CUDA device 0: 0 triples` ist daher keine K3-Expert-Residency und darf nicht als Nutzung der RTX 3060 Ti gewertet werden. Remote-GPU-Expert-Ausführung bzw. ein sicherer Peer-Copy-Pfad ist noch offen.

Honcho läuft geschützt auf `127.0.0.1:8001` und muss während der Tests verfügbar bleiben.

## Datenpfad: SSD → RAM → VRAM → Kernel

### 1. SSD / GGUF

Quelle:

```text
/home/neron/models/kimi-k3/Q2_K/
  Kimi-K3-Q2_K-00001-of-00024.gguf ... 00024-of-00024.gguf
```

Die 24 Shards werden über eine virtuelle Datei adressiert. Tensor-Offsets werden auf virtuelle absolute Offsets abgebildet; ein Tensor soll keinen Shard-Grenzübergang benötigen.

### 2. SSD → Host-RAM

Für Expertenslabs verwendet die Streaming-Schicht:

- Linux `io_uring`
- `O_DIRECT`
- 4096-Byte-Ausrichtung
- bounded in-flight reads
- vorallokierte, ausgerichtete Buffer
- optional CUDA-pinned staging memory
- Completion-Callbacks, während spätere Reads noch in flight sind

Die generische `stream::fetch::Fetcher` akzeptiert eine konfigurierbare Queue Depth `qd` und reicht bis zu `qd` Reads gleichzeitig ein. Die `io_uring`-Ringgröße ist `qd * 2`.

Die Pipeline-Schicht begrenzt zusätzlich die Gesamtzahl der Buffer über `max_slots`; das Speicherlimit ist ungefähr:

```text
max_slots × max_bracket_size
```

Die konkrete K3-Layer-Auflösung läuft aktuell über `StreamingStore::ensure_with()`, `DeviceSlabCache` und das K3-Staging-Bufferobjekt. Der erste Resolver-Pfad kann das Staging dynamisch auf die tatsächlich angeforderten Slabs vergrößern.

### 3. Host-RAM → VRAM

Für einen GPU-fähigen K3-Expertentriplettpfad gilt:

```text
GGUF expert slab
  → StreamingStore / Host-Cache
  → DeviceSlabCache hit
      oder
    staging.write(...)
  → ExpertPtrs mit Device-Adresse
  → CUDA MoE-Kernel
```

Bei einem VRAM-Cache-Miss werden die Slabs entweder in den persistenten Device-Cache eingefügt oder in das aktuelle Staging geschrieben. `ExpertPtrs` zeigt danach auf Cache- oder Staging-Adressen.

Aktuelle Smoke-Konfiguration:

```text
PULSAR_DEV_CACHE_GB=4
PULSAR_CACHE_GB=8
```

Startup meldete:

```text
expert cache 4.3GB, staging 0.2GB
```

### 4. Kernelpfad

Für quantisierte Dense-Projektionen:

```text
f32 activation
  → quantize_q8_k
  → direct matmul_kq(Q2_K/Q3_K weight, Q8_K activation)
```

Für K3-Experten:

```text
Q8_K latent activation
  → moe_pair_swiglu(Q2_K/Q3_K expert Gate+Up)
  → Q8_K expert-mid
  → moe_down(Q3_K Down)
```

Die K3-SiTU-GLU-Aktivierung ist über `act_op=4` vorgesehen. Der Host-Pfad bleibt die Referenz für nicht unterstützte Layouts/Quanttypen.

Die beiden großen Absorbed-Weights werden inzwischen als Q8_0 geladen:

```text
ffn_latent_up    F32 [3584 × 7168] → Q8_0
ffn_down_shexp   F32 [6144 × 7168] → Q8_0
```

Damit werden gemappte F32-PCIe-Reads durch kompaktere Q8_0-MatMuls ersetzt.

## Dateiformat und physische Ablage

### GGUF

- GGUF mit gesplitteten Shards.
- Shard 0 trägt die Metadaten und Tokenizerinformationen.
- Tensoren werden über virtuelle absolute Offsets referenziert.
- Layernorms, Projektionsgewichte und Absorbed-Tensoren werden beim Laden in Runtime-Repräsentationen überführt.

### Layer

K3-Layer enthalten architekturspezifisch u. a.:

- KDA-Recurrent-State und KDA-Projektionen
- MLA-/Absorbed-Attention-Gewichte
- Latent-MoE-Down/Norm/Up
- Router und Router-Bias
- Q2_K/Q2_K/Q3_K Expertenslabs
- Shared-Expert-Gate/Up/Down

### Experten

Die Experten liegen physisch nicht als einzelne Dateien vor. Sie sind als contiguous Expertenslabs in GGUF-Tensoren abgelegt:

```text
expert_tensor_base + expert_id × expert_bytes
```

Für jedes ausgewählte Expertentriplett werden Gate-, Up- und Down-Slab separat über ihre Metadaten (`abs_offset`, `expert_bytes`, `row_bytes`, `quant`) adressiert.

## Prefetch-Strategie und Queue-Tiefen

### Generische SSD-Pipeline

`Fetcher`:

- konfigurierbare `qd`
- bis zu `qd` Reads in flight
- `io_uring` queue size `qd × 2`
- O_DIRECT und 4096-Byte-Brackets
- bounded Buffer-Allokation
- Completion-Callback bei weiterlaufenden Reads

`Pipeline`:

- `qd` = maximale I/O-Konkurrenz
- `max_slots` = Speicher-/Backpressure-Grenze
- Submit blockiert bzw. drainiert Completions, wenn keine Buffer-Slots frei sind
- Drop verwirft/cancelt noch nicht verbrauchte Completions

### K3-Decode

Der aktuelle K3-Decodepfad löst die pro Layer gerouteten Expertenslabs auf. Ein vollständiger globaler Prefetch aller 93 Layer ist nicht implementiert; die Laufzeit arbeitet layerweise und cache-/staging-bounded.

Der Startup-Logwert `staging 0.0GB` war zwischenzeitlich irreführend: der K3-Resolver kann das Staging beim ersten konkreten `wants`-Set dynamisch vergrößern. Für Architekturgespräche muss daher die tatsächliche `stage_len`-Allocation instrumentiert werden, nicht nur die initiale Budgetzeile.

## Synchronisationspunkte

### CPU ↔ I/O

- `io_uring::submit_and_wait(1)` wartet auf mindestens eine Completion.
- Callback-Verarbeitung läuft während weitere Reads in flight bleiben.
- `StreamingStore::ensure_with()` stellt die benötigten Slabs vor dem Kernel-Dispatch bereit.

### CPU ↔ CUDA

Aktuelle synchrone/serialisierende Punkte im K3-Pfad:

- Router-Auswahl wird nach dem CUDA-Routerkernel auf Host gelesen.
- Expert-Slab-Resolution muss vor dem `ExpertPtrs`-Dispatch abgeschlossen sein.
- K3-Latent-MoE-Norm liest aktuell Werte auf den Host, berechnet RMSNorm auf dem Host und schreibt zurück.
- KDA-Conv/Silu- und Gate/Beta-Abschnitte enthalten weitere Host-Roundtrips.
- `PULSAR_DEBUG_CUDA_SYNC=1` fügt absichtliche `cudaDeviceSynchronize()`-Punkte nach Diagnoseabschnitten ein; für Performance-Messungen darf diese Variable nicht gesetzt sein.

### CUDA-Kernel

Die normalen Kernelstarts sind asynchron; Fehler werden teilweise erst bei späteren D2H-Copies oder expliziten Syncs sichtbar. Deshalb sind alte `status 700`-Logs ohne Tree-/Binary-Korrelation nicht ausreichend.

## Was bleibt dauerhaft in RAM/VRAM?

### VRAM

Dauerhaft bzw. über mehrere Token im Runtime-State:

- feste K3-Aktivierungs- und Scratch-Buffer
- KDA-Recurrent-States
- MLA-KV-/Latent-State
- Router-/Logit-/Selection-Buffer
- `DeviceSlabCache`-Slots für Expertenslabs
- aktueller Expert-Staging-Buffer
- Q8_K-Aktivierungsscratch
- Q8_0-Absorbed-Weights, sofern nicht host-pinned ausgelagert

### Host-RAM

- virtuelle GGUF-/Shard-Dateien und Betriebssystem-Pagecache
- `StreamingStore`-Hostcache, begrenzt durch `PULSAR_CACHE_GB`
- io_uring-/O_DIRECT-Buffer und optionale pinned Buffer
- Host-Referenzpfad für unsupported quant/layouts
- Tokenizer und GGUF-Metadaten

`PULSAR_K3_HOST=1` verlagert K3-residente Gewichte in host-pinned/mapped Speicher. Das spart VRAM, kann aber bei großen F32-MatMuls PCIe-Zugriffe erzwingen; deshalb wurden die beiden großen Absorbed-Weights auf Q8_0 umgestellt.

## MoE-Routing-Ablauf

Pro K3-MoE-Layer:

1. Latent-Down-Projektion der normierten Hidden-Aktivierung.
2. Router-Projektion auf `n_expert=896`.
3. CUDA-Routerauswahl mit Top-k=16.
4. Auswahlindizes und Gewichte werden auf den Host gelesen.
5. Für jedes ausgewählte Expertentriplett werden Gate/Up/Down-Offets gebildet.
6. `DeviceSlabCache` wird zuerst geprüft.
7. Fehlende Slabs gehen über Hostcache/StreamingStore in Device-Cache oder Staging.
8. `ExpertPtrs` werden für die CUDA-MoE-Kernel aufgebaut.
9. Gate/Up-SiTU-GLU und Down werden ausgeführt.
10. Latent-Norm/Up und Shared Experts werden berechnet.
11. Routed- und Shared-Expert-Output werden addiert.

GPU-MoE ist aktuell explizit für den unterstützten Q2_K/Q2_K/Q3_K-Contract aktiviert; andere Layouts fallen auf den Host-Referenzpfad zurück.

## Prefill vs. Decode

### Prefill

K3 unterstützt aktuell keinen effizienten batchierten Prefillpfad. Der Runtime-Code verarbeitet K3 tokenweise, weil KDA-Recurrent-State und AttnRes-Snapshots sequentiell sind.

Das bedeutet: ein Prompt mit sieben Tokens kann ungefähr sieben vollständige 93-Layer-Forwards verursachen.

### Decode

Decode verarbeitet ebenfalls einen Token pro vollständigem Layerdurchlauf. Der neue Direct-KQ-1-Token-Kernelpfad reduziert Leerlauf-Warps im Q2_K/Q3_K-MatMul. Die aktuelle Engine-Q8-Reuse reduziert zusätzliche Aktivierungsquantisierungen.

## Profilierter Ein-Token-Zeitstrahl

Quelle: `/tmp/pulsar-k3-q8-ab.log`, aktueller Tree, aktueller Build.

Kommando-Konfiguration:

```text
PULSAR_GPU=1
PULSAR_K3_HOST=1
PULSAR_K3_GPU_MOE=1
PULSAR_DEV_CACHE_GB=4
PULSAR_CACHE_GB=8
PULSAR_BATCH=1
PULSAR_K3_TIMING=1
prompt='x'
n_predict=1
ctx=16
```

Gemessen:

| Abschnitt | Zeit | Status |
|---|---:|---|
| Modellload, 24 Shards | 56,3 s | gemessen |
| Prefill, 1 Token | 40,56 s | gemessen |
| Decode, 1 Token | 34,88 s | gemessen |
| Gesamt Layerzeiten Prefill | 40,56 s | gemessen |
| Gesamt Layerzeiten Decode | 34,88 s | gemessen |
| Generierte Token | `id 58810` | gemessen |
| Durchsatz Decode | 0,03 tok/s | gemessen |

Layerprofil:

- KDA-/frühe Layer: ungefähr 0,43–0,56 s.
- spätere Layer nach dem Q8-Hotspot-Fix: ungefähr 0,32–0,42 s.
- wiederkehrende ca. 0,41-s-Layer entsprechen KDA-/Host-Synchronisationsabschnitten.

### Zeitstrahl im gewünschten Format

Die folgende Aufschlüsselung ist der belastbare Stand; nicht instrumentierte Kategorien bleiben offen:

```text
Model load / Shard-Setup:       56.300 ms   gemessen, nicht Tokenzeit
Prefill 1 Token:                40.560 ms   gemessen
Decode 1 Token:                 34.880 ms   gemessen
  Layer 0..92 aggregate:        34.880 ms   gemessen
  Router:                       offen       nicht separat instrumentiert
  SSD read während Decode:      offen       nicht separat instrumentiert
  CPU unpack/dequant:           offen       nicht separat instrumentiert
  H2D transfer:                 offen       nicht separat instrumentiert
  GPU compute:                  offen       CUDA/Nsight-Aufteilung fehlt
  CPU↔CUDA synchronization:     enthalten    nicht separat instrumentiert
  Sonstiges/Allocator/Host:     enthalten    nicht separat instrumentiert
```

Diese Tabelle ist absichtlich nicht künstlich auf 1000 ms normiert. Ein einzelner K3-Decode-Token dauert im aktuellen Lauf ca. 34,88 s; die Aufteilung in SSD/PCIe/Kernel/Sync muss noch mit Stage-Telemetrie und `nvidia-smi dmon`/Nsight während eines laufenden Tokens aufgenommen werden.

## GPU-, PCIe- und SSD-Durchsatz

### Gemessen

- In früheren K3-Runs wurde während der Forward-Phase GPU-Auslastung von ungefähr 100 % beobachtet.
- Aktuelle Gerätezuordnung ist verifiziert: `PULSAR_GPU=1` → RTX 3090.
- Aktueller Smoke-Run verwendete 4,3 GB Expert-Cache und 0,2 GB initiales Staging.

### Noch zu messen

Nicht belastbar gemessen sind:

- PCIe Read/Write-Durchsatz pro Token
- SSD Read-Durchsatz pro Token
- exakte Host-Dequant-/Unpack-Zeit
- H2D-Zeit pro Expertenslab
- GPU-Kernelzeit ohne Host-/Sync-Anteile
- Cache-Hit-/Miss-Aufschlüsselung pro Layer und Token

Für eine belastbare Messung muss ein K3-Prozess auf GPU 1 laufen, während parallel aufgezeichnet wird:

```bash
nvidia-smi dmon -s pucm -d 1
```

Zusätzlich benötigt der Runtime-Pfad Zähler für:

- `StreamingStore` hits/misses und Bytes
- `DeviceSlabCache` hits/misses und Bytes
- `stage_len` und tatsächliche staged bytes
- Host dequant/requant time
- H2D copy time
- kernel launch/sync time

Ohne diese Zähler wären konkrete PCIe-/SSD-Werte erfunden.

## Pulsar-eigene Komponenten vs. bestehende Libraries

### Pulsar-eigene Implementierung

- Kimi-K3-Modellloader und Layer-/Shape-Erkennung
- GGUF-Split-/Virtual-File-Adressierung
- K3 KDA/MLA-Forwardlogik
- K3 Direct-Q2_K/Q3_K-MatMul-Vertrag
- K3 Router und Top-k-Selection
- K3 GPU-MoE-Dispatch mit `ExpertPtrs`
- SiTU-GLU-/KDA-/MLA-CUDA-Kernel
- `DeviceSlabCache`, Expert-Tier-Priorisierung und Warm-Census-Anbindung
- bounded SSD→Host→staging-Pipeline
- K3-spezifische Tests und CPU-Referenzpfade

### Bestehende Libraries/Runtime-Bausteine

- Rust standard library und Cargo
- CUDA Runtime/Driver APIs und NVCC für CUDA-Kompilation
- `io-uring` Rust crate für Linux-I/O
- Linux `O_DIRECT`, pinned host allocation und CUDA memory APIs
- GGUF-/Tokenizer-Code als lokale Crates im Pulsar-Workspace
- lokale Quantisierungs-/Requantisierungsroutinen

Pulsar verwendet keine fertige allgemeine K3-Inference-Engine; die architekturspezifische Forward- und Tierlogik ist eigene Implementierung.

## Aktuelle Gates

Grün:

- K3 lädt 24 Shards / 93 Layer.
- Engine-Tests: 41/41.
- Workspace-Tests: grün.
- CUDA-/Kernel-Selftests: 17/17.
- Q2_K/Q3_K Direct-MatMul-Selftests: grün.
- Q8_0-MatMul-Selftests: grün.
- Honcho Healthcheck: `{"status":"ok"}`.
- Aktueller Smoke-Run erzeugt einen Token.

Offen:

- semantisch sichtbare Antwort auf einen natürlichen Prompt
- 0,33 tok/s cold
- 2–3× hot-cache-Speedup
- separate SSD/PCIe/H2D/Kernel/Sync-Telemetrie
- vollständiger semantic-prompt Run nach dem Q8-Hotspot-Fix

## Raw Evidence

- `/tmp/pulsar-k3-q8-ab.log` — aktueller Q8-Hotspot-Smoke-Run
- `/home/neron/.hermes/sessions/k3_performance_analysis.md` — read-only Performanceanalyse
- `/home/neron/projects/pulsar-kimi-k3/crates/engine/src/real/kimi_k3.rs`
- `/home/neron/projects/pulsar-kimi-k3/crates/engine/src/lib.rs`
- `/home/neron/projects/pulsar-kimi-k3/crates/kernels/cuda/pulsar_kernels.cu`
- `/home/neron/projects/pulsar-kimi-k3/crates/stream/src/lib.rs`
