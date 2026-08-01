# Setting Up the RHOAI Workbench

This guide walks through creating a Red Hat OpenShift AI (RHOAI)
workbench for running the experiments in this repository.

## Prerequisites

Before you start, make sure the following are in place on your OpenShift
cluster:

- **Red Hat OpenShift AI operator** installed (tested with 3.4.2)
- **NVIDIA GPU Operator** running, with at least one GPU node available
- A user account with permissions to create Data Science Projects

### Verifying GPU availability

These commands check that GPU nodes exist and drivers are loaded. They
require cluster-level access — if you don't have it, ask your cluster
admin to confirm GPU availability.

Check that GPU-capable nodes exist:

```bash
oc get nodes -l nvidia.com/gpu.present=true
```

Check GPU capacity per node:

```bash
oc get nodes -l nvidia.com/gpu.present=true \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}'
```

Verify the NVIDIA driver is running:

```bash
oc get pods -n nvidia-gpu-operator | grep driver
```

Run `nvidia-smi` through the driver pod to confirm the GPU is visible:

```bash
# Replace the pod name with the one from the previous command
oc exec -n nvidia-gpu-operator <driver-pod-name> -- nvidia-smi
```

## Create the Workbench

In the RHOAI dashboard:

1. **Create a Data Science Project**
   - Name: `kvcache-research` (or any name you prefer)

2. **Create a Workbench** inside the project with these settings:

   | Setting | Value |
   |---------|-------|
   | **Name** | `kvcache-notebooks` |
   | **Image** | CUDA v13.0, Python v3.12, PyTorch v2.10.0, Training Hub v0.6.0 |
   | **Hardware profile** | GPU profile |
   | **CPU** | requests: 4, limits: 8 |
   | **Memory** | requests: 16 GB, limits: 16 GB |
   | **GPU** | requests: 1, limits: 1 |

   The image provides PyTorch with CUDA support and a pre-installed
   vLLM build. The GPU profile requests one NVIDIA GPU (A10G or similar
   with 20+ GB VRAM).

3. **Wait for the workbench pod to start.** This may take a few minutes
   on first launch as the image is pulled.

### Verifying the workbench

Once the workbench is running, you can check its status from the
command line:

```bash
oc get all -n kvcache-research
```

Check which pods are using GPUs across the cluster:

```bash
oc get pods -A -o jsonpath='{range .items[*]}{range .spec.containers[*]}{.resources.limits.nvidia\.com/gpu}{end}{" "}{.metadata.namespace}{"/"}{.metadata.name}{"\n"}{end}' | grep -v "^ "
```

Monitor GPU utilization across all driver pods:

```bash
for pod in $(oc get pods -n nvidia-gpu-operator -o name | grep driver-daemonset); do
  echo "=== $pod ==="
  oc exec -n nvidia-gpu-operator $pod -- nvidia-smi
done
```

## Clone and Install

Open a terminal in the workbench (JupyterLab) and clone the repository:

```bash
git clone https://github.com/fax4ever/kvpress-vllm-tests.git
cd kvpress-vllm-tests
```

Then open and run **`notebooks/00_setup_check.ipynb`**. This notebook:

1. Verifies GPU access (`nvidia-smi` and PyTorch CUDA check)
2. Installs all Python dependencies (kvpress, vLLM, transformers,
   datasets, etc.)
3. Verifies imports and versions
4. Runs a CUDA smoke test
5. Checks dataset access

**Important:** set your Hugging Face token in the notebook before
running it. The cell near the top has
`os.environ["HF_TOKEN"] = "YOUR_TOKEN_HERE"` — replace this with your
actual token. You need it to download the Qwen3-8B model.

If the pod gets recreated (e.g., after a cluster restart), re-run
`00_setup_check.ipynb` — pip will skip already-installed packages.

## Run the Experiments

Once setup is complete, the experiments can be run in order:

### NIAH Comparison (`notebooks/`)

| Order | Notebook | What it does |
|-------|----------|--------------|
| 1 | `01_kvpress_fork_setup` | *(Optional)* Installs kvpress from a fork for testing unreleased changes |
| 2 | `02_kvpress_niah` | KeyDiffPress NIAH at compression ratios 0%, 25%, 50%, 75% |
| 3 | `03_vllm_fork_setup` | *(Optional)* Prepares a vLLM fork for testing modifications |
| 4 | `04_vllm_niah` | vLLM baseline NIAH (no compression) |
| 5 | `05_compare_results` | Loads results, produces comparison heatmaps and charts |

Notebooks 01 and 02 each take several minutes to run — they load the
Qwen3-8B model and run inference at multiple context lengths and needle
depths.

