Project to run vllm tests applying changes from [my fork](https://github.com/fax4ever/vllm.git) branch `fax-0.18`.
Those tests will be runned ad OpenShift Job on a cluster with NVIDIA GPU operator and a free gpu!

Commands:

1. make build

2. make push

3. make deploy

* open the Makefile to see the options you can configure using environment variables.