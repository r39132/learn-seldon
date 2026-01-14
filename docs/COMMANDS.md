# Command Reference

Quick reference for all `make` commands in this project.

## Table of Contents

- [Development Commands](#development-commands)
- [Kubernetes Commands](#kubernetes-commands)
- [Local UI Commands](#local-ui-commands)

## Development Commands

```bash
make data                      # Generate training data
make train                     # Train model
make notebook                  # Start Jupyter notebook
make clean-build-artifacts     # Clean Python caches
```

## Kubernetes Commands

```bash
make k8s-deploy-model-server   # Deploy model to K8s
make k8s-ms-status             # Check status
make k8s-ms-logs               # Stream logs
make k8s-ms-port-fwd           # Port forward service to localhost:8080
make k8s-ms-test               # Run tests against model server
make k8s-clean                 # Delete all K8s resources
```

## Local UI Commands

```bash
make run-ui                    # Start UI server
make stop-ui                   # Stop UI server
```
