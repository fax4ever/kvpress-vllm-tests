IMAGE_REPO   ?= quay.io/fercoli/vllm-test
IMAGE_TAG    ?= v0.18-fax
NAMESPACE    := vllm-tests-research
RELEASE_NAME ?= vllm-tests
HF_TOKEN     ?=
FULL_FP8     ?= false

IMAGE = $(IMAGE_REPO):$(IMAGE_TAG)

.PHONY: build push deploy undeploy redeploy

build:
	podman build --platform linux/amd64 -f Containerfile -t $(IMAGE) .

push:
	podman push $(IMAGE)

deploy:
	helm install $(RELEASE_NAME) chart/ \
		--create-namespace -n $(NAMESPACE) \
		--set image.repository=$(IMAGE_REPO) \
		--set image.tag=$(IMAGE_TAG) \
		--set huggingface.token=$(HF_TOKEN) \
		--set gpu.fullFp8Support=$(FULL_FP8) \
		--set pytest.parallel=$(FULL_FP8)

undeploy:
	helm uninstall $(RELEASE_NAME) -n $(NAMESPACE)

redeploy: undeploy deploy
