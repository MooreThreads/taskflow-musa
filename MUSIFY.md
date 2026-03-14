# MUSIFY — taskflow

| Field | Value |
|-------|-------|
| Project | taskflow |
| Source repo | https://github.com/taskflow/taskflow |
| Migration branch | musa |
| Artifact type | Lib |
| Build system | CMake (enable_language(CUDA), TF_BUILD_CUDA option) |
| Test infrastructure | doctest (unittests/cuda/), CUDA examples (examples/cuda/) |
| MUSA SDK version | 4.3.5 |
| torch_musa version | v2.7.1 |
| MT GPU type | S5000 |
| Pipeline start | 2026-03-14 15:03:22 UTC |
| Pipeline end | |
| Elapsed | |
| Last updated | 2026-03-14 15:10 |
| Director | xiaofeng |
| Status | in-progress |

## Migration Plan

Taskflow is a header-only C++ task-parallel programming library with optional CUDA support (gated by `TF_BUILD_CUDA`). The CUDA layer lives entirely in `taskflow/cuda/*.hpp` headers and `.cu` source files in `examples/cuda/`, `unittests/cuda/`, `benchmarks/hetero_traversal/`, and `sandbox/cublas_*`. No Python bindings exist.

Migration approach:
1. Port all `.cu` source files to `.mu` equivalents using SimplePorting
2. Modify CMakeLists.txt to support MUSA via `TF_BUILD_MUSA` option
3. Replace CUDA API calls in headers with MUSA equivalents
4. Build and verify with mcc compiler

## CUDA Dependencies

None — all dependencies are MUSA-compatible or CUDA-free. The project uses only standard CUDA Runtime API (no cuBLAS, cuDNN, etc. in active code).

## Pipeline Progress

### taskflow (Main)

| Stage | Status | Started | Completed | Notes |
|-------|--------|---------|-----------|-------|
| 0. sdk-version | done | 2026-03-14 15:03 | 2026-03-14 15:05 | v2.7.1 / SDK 4.3.5 |
| 1a. reader (code analysis) | in-progress | 2026-03-14 15:05 | | |
| 1b. lookuper (document-site) | pending | | | |
| 1c. lookuper (docker setup) | pending | | | |
| 2. api-mapping | pending | | | |
| 3. musa-adapt | pending | | | |

#### Stage Details

**Stage 0 — SDK Version Resolution (Core):**
- torch_musa version: v2.7.1
- MUSA SDK version: 4.3.5
- Docker image: 4.3.5_kuae2.1_20260119_torch2.7.1_ubuntu

**Stage 1 — Parallel Preparation:**

*1a. Reader (code analysis):*
- Project type: Lib (header-only C++ library with CUDA extensions)
- CUDA files: 53 .cu files total (12 examples, 18 unittests, 4 benchmarks, 7 sandbox, 12 3rd-party eigen)
- CUDA headers: 21 .hpp files in taskflow/cuda/ (the actual library CUDA layer)
- No .cuh files (headers use .hpp extension)
- Minimum CUDA version: 12 (explicit check in CMakeLists.txt)
- Dependencies requiring musification: none
- Build system: CMake with TF_BUILD_CUDA option

## Issues

| # | Team | Stage | Severity | Summary | Status | Resolution |
|---|------|-------|----------|---------|--------|------------|

## Knowledge Base

## Final Report
