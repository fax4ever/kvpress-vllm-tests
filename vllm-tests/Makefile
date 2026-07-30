IMAGE_REPO   ?= quay.io/fercoli/vllm-test
IMAGE_TAG    ?= v0.18-fax
NAMESPACE    ?= vllm-tests
RELEASE_NAME ?= vllm-tests
HF_TOKEN     ?=

IMAGE = $(IMAGE_REPO):$(IMAGE_TAG)

.PHONY: build push deploy undeploy

build:
	podman build -f Containerfile -t $(IMAGE) .

push:
	podman push $(IMAGE)

deploy:
	helm install $(RELEASE_NAME) chart/ \
		--create-namespace -n $(NAMESPACE) \
		--set image.repository=$(IMAGE_REPO) \
		--set image.tag=$(IMAGE_TAG) \
		--set huggingface.token=$(HF_TOKEN)

undeploy:
	helm uninstall $(RELEASE_NAME) -n $(NAMESPACE)
