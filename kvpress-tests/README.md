Run the kvpress test suite on an OpenShift cluster with NVIDIA GPUs.

The container image contains the environment (PyTorch + CUDA + dependencies).
At runtime, the Job clones your kvpress fork branch and runs pytest — no image
rebuild needed for code changes.

## Usage

1. `make build` — build the environment image (one-time or when deps change)
2. `make push` — push to the container registry
3. `make deploy HF_TOKEN=<token> FORK_BRANCH=<branch>` — run tests
4. Check results: `oc logs -n kvpress-tests-research job/kvpress-tests`

Environment variables for `make deploy`:

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | (required) | HuggingFace token for gated models |
| `FORK_BRANCH` | `main` | Branch to clone and test |
| `IMAGE_REPO` | `quay.io/fercoli/kvpress-test` | Container image repository |
| `IMAGE_TAG` | `cu130-py312` | Container image tag |
