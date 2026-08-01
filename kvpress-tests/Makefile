IMAGE_REPO   ?= quay.io/fercoli/kvpress-test
IMAGE_TAG    ?= cu130-py312
NAMESPACE    := kvpress-tests-research
RELEASE_NAME ?= kvpress-tests
HF_TOKEN     ?=
FORK_BRANCH  ?= main

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
		--set fork.branch=$(FORK_BRANCH)

undeploy:
	helm uninstall $(RELEASE_NAME) -n $(NAMESPACE)

redeploy: undeploy deploy
