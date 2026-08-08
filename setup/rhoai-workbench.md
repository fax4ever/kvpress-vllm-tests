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

## Run the Experiments

Sometimes the Jupyter UI becomes unresponsive — the WebSocket disconnects after long-running tasks,
and you lose visibility on what's happening. This does not mean the inference has stopped.

```bash
oc logs podname --tail=1
```

that will produce:

```bash
fax@fercoli-mac kvpress-vllm-tests % oc logs pods/kvcache-notebooks-0 --tail=1
Defaulted container "kvcache-notebooks" out of: kvcache-notebooks, kube-rbac-proxy
Running Inference:  50%|████▉     | 3227/6500 [2:18:16<1:59:13,  2.19s/it]%
```

To monitor progress continuously:

```bash
while true; do oc logs pod/kvcache-notebooks-0 --tail=1; sleep 10; done
```
