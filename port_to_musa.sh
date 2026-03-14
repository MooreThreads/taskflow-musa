#!/bin/bash
# Port taskflow CUDA code to MUSA
# This script creates MUSA-equivalent files from CUDA originals
# Original CUDA files are never modified

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Porting taskflow CUDA to MUSA ==="
echo "Project dir: $PROJECT_DIR"

# ============================================================================
# Step 1: Create taskflow/musa/ directory from taskflow/cuda/
# ============================================================================
echo "--- Step 1: Porting taskflow/cuda/ headers to taskflow/musa/ ---"

CUDA_HPP_DIR="$PROJECT_DIR/taskflow/cuda"
MUSA_HPP_DIR="$PROJECT_DIR/taskflow/musa"

# Remove old musa dir if exists
rm -rf "$MUSA_HPP_DIR"
# Copy the entire cuda directory to musa
cp -r "$CUDA_HPP_DIR" "$MUSA_HPP_DIR"

# Rename files: .cu -> .mu, .cuh -> .muh (none in this project, but handle it)
find "$MUSA_HPP_DIR" -name "*.cu" -exec bash -c 'mv "$1" "${1%.cu}.mu"' _ {} \;
find "$MUSA_HPP_DIR" -name "*.cuh" -exec bash -c 'mv "$1" "${1%.cuh}.muh"' _ {} \;

# Apply CUDA SDK -> MUSA SDK replacements in all header files
# These are the actual SDK types and function calls, NOT taskflow's own identifiers
find "$MUSA_HPP_DIR" -type f \( -name "*.hpp" -o -name "*.h" -o -name "*.cpp" -o -name "*.mu" -o -name "*.muh" \) | while read file; do
    # Include headers
    sed -i 's|#include <cuda\.h>|#include <musa.h>|g' "$file"
    sed -i 's|#include <cuda_runtime\.h>|#include <musa_runtime.h>|g' "$file"
    sed -i 's|#include <cuda_runtime_api\.h>|#include <musa_runtime_api.h>|g' "$file"

    # Internal includes: cuda/ -> musa/ for taskflow headers
    sed -i 's|"cuda_error\.hpp"|"musa_error.hpp"|g' "$file"
    sed -i 's|"cuda_device\.hpp"|"musa_device.hpp"|g' "$file"
    sed -i 's|"cuda_memory\.hpp"|"musa_memory.hpp"|g' "$file"
    sed -i 's|"cuda_stream\.hpp"|"musa_stream.hpp"|g' "$file"
    sed -i 's|"cuda_graph\.hpp"|"musa_graph.hpp"|g' "$file"
    sed -i 's|"cuda_graph_exec\.hpp"|"musa_graph_exec.hpp"|g' "$file"
    sed -i 's|"cuda_execution_policy\.hpp"|"musa_execution_policy.hpp"|g' "$file"
    sed -i 's|"cuda_meta\.hpp"|"musa_meta.hpp"|g' "$file"
    sed -i 's|"cuda_capturer\.hpp"|"musa_capturer.hpp"|g' "$file"
    sed -i 's|"cuda_optimizer\.hpp"|"musa_optimizer.hpp"|g' "$file"

    # Relative includes for algorithm/
    sed -i 's|"algorithm/single_task\.hpp"|"algorithm/single_task.hpp"|g' "$file"

    # cudaflow.hpp -> musaflow.hpp
    sed -i 's|"cudaflow\.hpp"|"musaflow.hpp"|g' "$file"

    # CUDA SDK error types and functions
    sed -i 's/cudaSuccess/musaSuccess/g' "$file"
    sed -i 's/cudaGetErrorString/musaGetErrorString/g' "$file"
    sed -i 's/cudaGetErrorName/musaGetErrorName/g' "$file"
    sed -i 's/cudaError_t/musaError_t/g' "$file"

    # CUDA SDK memory functions
    sed -i 's/cudaMalloc\b/musaMalloc/g' "$file"
    sed -i 's/cudaMallocManaged/musaMallocManaged/g' "$file"
    sed -i 's/cudaFree\b/musaFree/g' "$file"
    sed -i 's/cudaMemcpy\b/musaMemcpy/g' "$file"
    sed -i 's/cudaMemcpyAsync/musaMemcpyAsync/g' "$file"
    sed -i 's/cudaMemcpyDefault/musaMemcpyDefault/g' "$file"
    sed -i 's/cudaMemsetAsync/musaMemsetAsync/g' "$file"
    sed -i 's/cudaMemGetInfo/musaMemGetInfo/g' "$file"
    sed -i 's/cudaMemcpy3DParms/musaMemcpy3DParms/g' "$file"
    sed -i 's/cudaMemsetParams/musaMemsetParams/g' "$file"

    # CUDA SDK device functions
    sed -i 's/cudaGetDeviceCount/musaGetDeviceCount/g' "$file"
    sed -i 's/cudaGetDevice\b/musaGetDevice/g' "$file"
    sed -i 's/cudaSetDevice/musaSetDevice/g' "$file"
    sed -i 's/cudaGetDeviceProperties/musaGetDeviceProperties/g' "$file"
    sed -i 's/cudaDeviceGetAttribute/musaDeviceGetAttribute/g' "$file"
    sed -i 's/cudaDeviceProp\b/musaDeviceProp/g' "$file"
    sed -i 's/cudaDriverGetVersion/musaDriverGetVersion/g' "$file"
    sed -i 's/cudaRuntimeGetVersion/musaRuntimeGetVersion/g' "$file"

    # CUDA SDK device attributes
    sed -i 's/cudaDevAttrMaxThreadsPerBlock/musaDevAttrMaxThreadsPerBlock/g' "$file"
    sed -i 's/cudaDevAttrMaxBlockDimX/musaDevAttrMaxBlockDimX/g' "$file"
    sed -i 's/cudaDevAttrMaxBlockDimY/musaDevAttrMaxBlockDimY/g' "$file"
    sed -i 's/cudaDevAttrMaxBlockDimZ/musaDevAttrMaxBlockDimZ/g' "$file"
    sed -i 's/cudaDevAttrMaxGridDimX/musaDevAttrMaxGridDimX/g' "$file"
    sed -i 's/cudaDevAttrMaxGridDimY/musaDevAttrMaxGridDimY/g' "$file"
    sed -i 's/cudaDevAttrMaxGridDimZ/musaDevAttrMaxGridDimZ/g' "$file"
    sed -i 's/cudaDevAttrMaxSharedMemoryPerBlock/musaDevAttrMaxSharedMemoryPerBlock/g' "$file"
    sed -i 's/cudaDevAttrWarpSize/musaDevAttrWarpSize/g' "$file"
    sed -i 's/cudaDevAttrComputeCapabilityMajor/musaDevAttrComputeCapabilityMajor/g' "$file"
    sed -i 's/cudaDevAttrComputeCapabilityMinor/musaDevAttrComputeCapabilityMinor/g' "$file"
    sed -i 's/cudaDevAttrUnifiedAddressing/musaDevAttrUnifiedAddressing/g' "$file"

    # CUDA SDK stream types and functions
    sed -i 's/cudaStream_t\b/musaStream_t/g' "$file"
    sed -i 's/cudaStreamCreate\b/musaStreamCreate/g' "$file"
    sed -i 's/cudaStreamDestroy/musaStreamDestroy/g' "$file"
    sed -i 's/cudaStreamSynchronize/musaStreamSynchronize/g' "$file"
    sed -i 's/cudaStreamBeginCapture/musaStreamBeginCapture/g' "$file"
    sed -i 's/cudaStreamEndCapture/musaStreamEndCapture/g' "$file"
    sed -i 's/cudaStreamWaitEvent/musaStreamWaitEvent/g' "$file"
    sed -i 's/cudaStreamCaptureMode/musaStreamCaptureMode/g' "$file"
    sed -i 's/cudaStreamCaptureModeGlobal/musaStreamCaptureModeGlobal/g' "$file"
    sed -i 's/cudaStreamCaptureModeThreadLocal/musaStreamCaptureModeThreadLocal/g' "$file"
    sed -i 's/cudaStreamCaptureModeRelaxed/musaStreamCaptureModeRelaxed/g' "$file"

    # CUDA SDK event types and functions
    sed -i 's/cudaEvent_t\b/musaEvent_t/g' "$file"
    sed -i 's/cudaEventCreate\b/musaEventCreate/g' "$file"
    sed -i 's/cudaEventCreateWithFlags/musaEventCreateWithFlags/g' "$file"
    sed -i 's/cudaEventDestroy/musaEventDestroy/g' "$file"
    sed -i 's/cudaEventRecord/musaEventRecord/g' "$file"
    sed -i 's/cudaEventQuery/musaEventQuery/g' "$file"

    # CUDA SDK graph types and functions
    sed -i 's/cudaGraph_t\b/musaGraph_t/g' "$file"
    sed -i 's/cudaGraphNode_t/musaGraphNode_t/g' "$file"
    sed -i 's/cudaGraphExec_t/musaGraphExec_t/g' "$file"
    sed -i 's/cudaGraphCreate/musaGraphCreate/g' "$file"
    sed -i 's/cudaGraphDestroy/musaGraphDestroy/g' "$file"
    sed -i 's/cudaGraphGetNodes/musaGraphGetNodes/g' "$file"
    sed -i 's/cudaGraphGetRootNodes/musaGraphGetRootNodes/g' "$file"
    sed -i 's/cudaGraphGetEdges/musaGraphGetEdges/g' "$file"
    sed -i 's/cudaGraphNodeGetType/musaGraphNodeGetType/g' "$file"
    sed -i 's/cudaGraphNodeGetDependencies/musaGraphNodeGetDependencies/g' "$file"
    sed -i 's/cudaGraphNodeGetDependentNodes/musaGraphNodeGetDependentNodes/g' "$file"
    sed -i 's/cudaGraphAddDependencies/musaGraphAddDependencies/g' "$file"
    sed -i 's/cudaGraphAddEmptyNode/musaGraphAddEmptyNode/g' "$file"
    sed -i 's/cudaGraphAddHostNode/musaGraphAddHostNode/g' "$file"
    sed -i 's/cudaGraphAddKernelNode/musaGraphAddKernelNode/g' "$file"
    sed -i 's/cudaGraphAddMemsetNode/musaGraphAddMemsetNode/g' "$file"
    sed -i 's/cudaGraphAddMemcpyNode/musaGraphAddMemcpyNode/g' "$file"
    sed -i 's/cudaGraphInstantiate/musaGraphInstantiate/g' "$file"
    sed -i 's/cudaGraphLaunch/musaGraphLaunch/g' "$file"
    sed -i 's/cudaGraphExecDestroy/musaGraphExecDestroy/g' "$file"
    sed -i 's/cudaGraphExecUpdate/musaGraphExecUpdate/g' "$file"
    sed -i 's/cudaGraphDebugDotPrint/musaGraphDebugDotPrint/g' "$file"
    sed -i 's/cudaGraphChildGraphNodeGetGraph/musaGraphChildGraphNodeGetGraph/g' "$file"
    sed -i 's/cudaGraphExecKernelNodeSetParams/musaGraphExecKernelNodeSetParams/g' "$file"
    sed -i 's/cudaGraphExecMemcpyNodeSetParams/musaGraphExecMemcpyNodeSetParams/g' "$file"
    sed -i 's/cudaGraphExecMemsetNodeSetParams/musaGraphExecMemsetNodeSetParams/g' "$file"
    sed -i 's/cudaGraphNodeSetEnabled/musaGraphNodeSetEnabled/g' "$file"

    # CUDA SDK graph node type enum
    sed -i 's/cudaGraphNodeType\b/musaGraphNodeType/g' "$file"
    sed -i 's/cudaGraphNodeTypeKernel/musaGraphNodeTypeKernel/g' "$file"
    sed -i 's/cudaGraphNodeTypeMemcpy/musaGraphNodeTypeMemcpy/g' "$file"
    sed -i 's/cudaGraphNodeTypeMemset/musaGraphNodeTypeMemset/g' "$file"
    sed -i 's/cudaGraphNodeTypeHost/musaGraphNodeTypeHost/g' "$file"
    sed -i 's/cudaGraphNodeTypeGraph/musaGraphNodeTypeGraph/g' "$file"
    sed -i 's/cudaGraphNodeTypeEmpty/musaGraphNodeTypeEmpty/g' "$file"
    sed -i 's/cudaGraphNodeTypeWaitEvent/musaGraphNodeTypeWaitEvent/g' "$file"
    sed -i 's/cudaGraphNodeTypeEventRecord/musaGraphNodeTypeEventRecord/g' "$file"
    sed -i 's/cudaGraphNodeTypeExtSemaphoreSignal/musaGraphNodeTypeExtSemaphoreSignal/g' "$file"
    sed -i 's/cudaGraphNodeTypeExtSemaphoreWait/musaGraphNodeTypeExtSemaphoreWait/g' "$file"
    sed -i 's/cudaGraphNodeTypeMemAlloc/musaGraphNodeTypeMemAlloc/g' "$file"
    sed -i 's/cudaGraphNodeTypeMemFree/musaGraphNodeTypeMemFree/g' "$file"
    sed -i 's/cudaGraphNodeTypeConditional/musaGraphNodeTypeConditional/g' "$file"
    sed -i 's/cudaGraphExecUpdateResult/musaGraphExecUpdateResult/g' "$file"
    sed -i 's/cudaGraphExecUpdateSuccess/musaGraphExecUpdateSuccess/g' "$file"

    # CUDA SDK kernel/host node params
    sed -i 's/cudaKernelNodeParams/musaKernelNodeParams/g' "$file"
    sed -i 's/cudaHostNodeParams/musaHostNodeParams/g' "$file"

    # CUDA SDK pitched ptr / extent / pos
    sed -i 's/make_cudaPitchedPtr/make_musaPitchedPtr/g' "$file"
    sed -i 's/make_cudaPos/make_musaPos/g' "$file"
    sed -i 's/make_cudaExtent/make_musaExtent/g' "$file"

    # Compiler macros
    sed -i 's/__CUDACC__/__MUSACC__/g' "$file"
    sed -i 's/__CUDACC_VER_MAJOR__/__MUSACC_VER_MAJOR__/g' "$file"
    sed -i 's/__NVCC__/__MUSACC__/g' "$file"

    # Occupancy
    sed -i 's/cudaOccupancyMaxPotentialBlockSize/musaOccupancyMaxPotentialBlockSize/g' "$file"
    sed -i 's/cudaOccupancyMaxActiveBlocksPerMultiprocessor/musaOccupancyMaxActiveBlocksPerMultiprocessor/g' "$file"

    # Misc SDK functions
    sed -i 's/cudaDeviceSynchronize/musaDeviceSynchronize/g' "$file"
    sed -i 's/cudaLaunchKernel/musaLaunchKernel/g' "$file"
    sed -i 's/cudaFuncSetAttribute/musaFuncSetAttribute/g' "$file"
    sed -i 's/cudaFuncAttribute/musaFuncAttribute/g' "$file"
    sed -i 's/cudaFuncAttributeMaxDynamicSharedMemorySize/musaFuncAttributeMaxDynamicSharedMemorySize/g' "$file"

done

# Rename header files: cuda_*.hpp -> musa_*.hpp, cudaflow.hpp -> musaflow.hpp
cd "$MUSA_HPP_DIR"
for f in cuda_*.hpp; do
    [ -f "$f" ] && mv "$f" "musa_${f#cuda_}"
done
[ -f "cudaflow.hpp" ] && mv "cudaflow.hpp" "musaflow.hpp"

echo "--- Header porting complete ---"

# ============================================================================
# Step 2: Port .cu example and test files
# ============================================================================
echo "--- Step 2: Porting .cu source files ---"

# Port examples/cuda/ -> examples/musa/
CUDA_EXAMPLES="$PROJECT_DIR/examples/cuda"
MUSA_EXAMPLES="$PROJECT_DIR/examples/musa"
rm -rf "$MUSA_EXAMPLES"
cp -r "$CUDA_EXAMPLES" "$MUSA_EXAMPLES"

# Port unittests/cuda/ -> unittests/musa/
CUDA_TESTS="$PROJECT_DIR/unittests/cuda"
MUSA_TESTS="$PROJECT_DIR/unittests/musa"
rm -rf "$MUSA_TESTS"
cp -r "$CUDA_TESTS" "$MUSA_TESTS"

# Rename .cu -> .mu in both directories
find "$MUSA_EXAMPLES" "$MUSA_TESTS" -name "*.cu" -exec bash -c 'mv "$1" "${1%.cu}.mu"' _ {} \;

# Apply same CUDA SDK -> MUSA SDK replacements to .mu files
for dir in "$MUSA_EXAMPLES" "$MUSA_TESTS"; do
    find "$dir" -type f \( -name "*.mu" -o -name "*.hpp" -o -name "*.h" -o -name "*.cpp" \) | while read file; do
        # Include header path changes: taskflow/cuda/ -> taskflow/musa/
        sed -i 's|taskflow/cuda/cudaflow\.hpp|taskflow/musa/musaflow.hpp|g' "$file"
        sed -i 's|taskflow/cuda/cuda_|taskflow/musa/musa_|g' "$file"
        sed -i 's|taskflow/cuda/algorithm/|taskflow/musa/algorithm/|g' "$file"

        # CUDA SDK includes
        sed -i 's|#include <cuda\.h>|#include <musa.h>|g' "$file"
        sed -i 's|#include <cuda_runtime\.h>|#include <musa_runtime.h>|g' "$file"
        sed -i 's|#include <cuda_runtime_api\.h>|#include <musa_runtime_api.h>|g' "$file"

        # Apply same SDK API replacements
        sed -i 's/cudaSuccess/musaSuccess/g' "$file"
        sed -i 's/cudaGetErrorString/musaGetErrorString/g' "$file"
        sed -i 's/cudaGetErrorName/musaGetErrorName/g' "$file"
        sed -i 's/cudaMalloc\b/musaMalloc/g' "$file"
        sed -i 's/cudaMallocManaged/musaMallocManaged/g' "$file"
        sed -i 's/cudaFree\b/musaFree/g' "$file"
        sed -i 's/cudaMemcpy\b/musaMemcpy/g' "$file"
        sed -i 's/cudaMemcpyAsync/musaMemcpyAsync/g' "$file"
        sed -i 's/cudaMemcpyDefault/musaMemcpyDefault/g' "$file"
        sed -i 's/cudaMemcpyHostToDevice/musaMemcpyHostToDevice/g' "$file"
        sed -i 's/cudaMemcpyDeviceToHost/musaMemcpyDeviceToHost/g' "$file"
        sed -i 's/cudaMemsetAsync/musaMemsetAsync/g' "$file"
        sed -i 's/cudaMemGetInfo/musaMemGetInfo/g' "$file"
        sed -i 's/cudaMemcpy3DParms/musaMemcpy3DParms/g' "$file"
        sed -i 's/cudaMemsetParams/musaMemsetParams/g' "$file"
        sed -i 's/cudaGetDeviceCount/musaGetDeviceCount/g' "$file"
        sed -i 's/cudaGetDevice\b/musaGetDevice/g' "$file"
        sed -i 's/cudaSetDevice/musaSetDevice/g' "$file"
        sed -i 's/cudaGetDeviceProperties/musaGetDeviceProperties/g' "$file"
        sed -i 's/cudaDeviceGetAttribute/musaDeviceGetAttribute/g' "$file"
        sed -i 's/cudaDeviceProp\b/musaDeviceProp/g' "$file"
        sed -i 's/cudaDeviceSynchronize/musaDeviceSynchronize/g' "$file"
        sed -i 's/cudaStream_t\b/musaStream_t/g' "$file"
        sed -i 's/cudaStreamCreate\b/musaStreamCreate/g' "$file"
        sed -i 's/cudaStreamDestroy/musaStreamDestroy/g' "$file"
        sed -i 's/cudaStreamSynchronize/musaStreamSynchronize/g' "$file"
        sed -i 's/cudaEvent_t\b/musaEvent_t/g' "$file"
        sed -i 's/cudaEventCreate\b/musaEventCreate/g' "$file"
        sed -i 's/cudaEventDestroy/musaEventDestroy/g' "$file"
        sed -i 's/cudaEventRecord/musaEventRecord/g' "$file"
        sed -i 's/cudaGraph_t\b/musaGraph_t/g' "$file"
        sed -i 's/cudaGraphNode_t/musaGraphNode_t/g' "$file"
        sed -i 's/cudaGraphExec_t/musaGraphExec_t/g' "$file"
        sed -i 's/cudaGraphNodeType\b/musaGraphNodeType/g' "$file"
        sed -i 's/cudaKernelNodeParams/musaKernelNodeParams/g' "$file"
        sed -i 's/cudaHostNodeParams/musaHostNodeParams/g' "$file"
        sed -i 's/make_cudaPitchedPtr/make_musaPitchedPtr/g' "$file"
        sed -i 's/make_cudaPos/make_musaPos/g' "$file"
        sed -i 's/make_cudaExtent/make_musaExtent/g' "$file"
        sed -i 's/__CUDACC__/__MUSACC__/g' "$file"
        sed -i 's/__CUDACC_VER_MAJOR__/__MUSACC_VER_MAJOR__/g' "$file"
        sed -i 's/cudaOccupancyMaxPotentialBlockSize/musaOccupancyMaxPotentialBlockSize/g' "$file"

        # Graph API
        sed -i 's/cudaGraphCreate/musaGraphCreate/g' "$file"
        sed -i 's/cudaGraphDestroy/musaGraphDestroy/g' "$file"
        sed -i 's/cudaGraphGetNodes/musaGraphGetNodes/g' "$file"
        sed -i 's/cudaGraphGetRootNodes/musaGraphGetRootNodes/g' "$file"
        sed -i 's/cudaGraphGetEdges/musaGraphGetEdges/g' "$file"
        sed -i 's/cudaGraphAddDependencies/musaGraphAddDependencies/g' "$file"
        sed -i 's/cudaGraphAddEmptyNode/musaGraphAddEmptyNode/g' "$file"
        sed -i 's/cudaGraphAddHostNode/musaGraphAddHostNode/g' "$file"
        sed -i 's/cudaGraphAddKernelNode/musaGraphAddKernelNode/g' "$file"
        sed -i 's/cudaGraphAddMemsetNode/musaGraphAddMemsetNode/g' "$file"
        sed -i 's/cudaGraphAddMemcpyNode/musaGraphAddMemcpyNode/g' "$file"
        sed -i 's/cudaGraphInstantiate/musaGraphInstantiate/g' "$file"
        sed -i 's/cudaGraphLaunch/musaGraphLaunch/g' "$file"
        sed -i 's/cudaGraphExecDestroy/musaGraphExecDestroy/g' "$file"
        sed -i 's/cudaGraphExecUpdate/musaGraphExecUpdate/g' "$file"
        sed -i 's/cudaGraphDebugDotPrint/musaGraphDebugDotPrint/g' "$file"
        sed -i 's/cudaGraphNodeGetType/musaGraphNodeGetType/g' "$file"
        sed -i 's/cudaGraphNodeGetDependencies/musaGraphNodeGetDependencies/g' "$file"
        sed -i 's/cudaGraphNodeGetDependentNodes/musaGraphNodeGetDependentNodes/g' "$file"
        sed -i 's/cudaGraphExecKernelNodeSetParams/musaGraphExecKernelNodeSetParams/g' "$file"
        sed -i 's/cudaGraphExecMemcpyNodeSetParams/musaGraphExecMemcpyNodeSetParams/g' "$file"
        sed -i 's/cudaGraphExecMemsetNodeSetParams/musaGraphExecMemsetNodeSetParams/g' "$file"
        sed -i 's/cudaGraphNodeSetEnabled/musaGraphNodeSetEnabled/g' "$file"

        # Stream capture
        sed -i 's/cudaStreamBeginCapture/musaStreamBeginCapture/g' "$file"
        sed -i 's/cudaStreamEndCapture/musaStreamEndCapture/g' "$file"
        sed -i 's/cudaStreamWaitEvent/musaStreamWaitEvent/g' "$file"
        sed -i 's/cudaStreamCaptureMode/musaStreamCaptureMode/g' "$file"
        sed -i 's/cudaStreamCaptureModeGlobal/musaStreamCaptureModeGlobal/g' "$file"

        # Device attributes
        sed -i 's/cudaDevAttrMaxThreadsPerBlock/musaDevAttrMaxThreadsPerBlock/g' "$file"
        sed -i 's/cudaDevAttrMaxBlockDimX/musaDevAttrMaxBlockDimX/g' "$file"
        sed -i 's/cudaDevAttrMaxBlockDimY/musaDevAttrMaxBlockDimY/g' "$file"
        sed -i 's/cudaDevAttrMaxBlockDimZ/musaDevAttrMaxBlockDimZ/g' "$file"
        sed -i 's/cudaDevAttrMaxGridDimX/musaDevAttrMaxGridDimX/g' "$file"
        sed -i 's/cudaDevAttrMaxGridDimY/musaDevAttrMaxGridDimY/g' "$file"
        sed -i 's/cudaDevAttrMaxGridDimZ/musaDevAttrMaxGridDimZ/g' "$file"
        sed -i 's/cudaDevAttrMaxSharedMemoryPerBlock/musaDevAttrMaxSharedMemoryPerBlock/g' "$file"
        sed -i 's/cudaDevAttrWarpSize/musaDevAttrWarpSize/g' "$file"
        sed -i 's/cudaDevAttrComputeCapabilityMajor/musaDevAttrComputeCapabilityMajor/g' "$file"
        sed -i 's/cudaDevAttrComputeCapabilityMinor/musaDevAttrComputeCapabilityMinor/g' "$file"
        sed -i 's/cudaDevAttrUnifiedAddressing/musaDevAttrUnifiedAddressing/g' "$file"
        sed -i 's/cudaDriverGetVersion/musaDriverGetVersion/g' "$file"
        sed -i 's/cudaRuntimeGetVersion/musaRuntimeGetVersion/g' "$file"

        # Error type
        sed -i 's/cudaError_t/musaError_t/g' "$file"

        # Misc
        sed -i 's/cudaGraphNodeTypeKernel/musaGraphNodeTypeKernel/g' "$file"
        sed -i 's/cudaGraphNodeTypeMemcpy/musaGraphNodeTypeMemcpy/g' "$file"
        sed -i 's/cudaGraphNodeTypeMemset/musaGraphNodeTypeMemset/g' "$file"
        sed -i 's/cudaGraphNodeTypeHost/musaGraphNodeTypeHost/g' "$file"
        sed -i 's/cudaGraphNodeTypeGraph/musaGraphNodeTypeGraph/g' "$file"
        sed -i 's/cudaGraphNodeTypeEmpty/musaGraphNodeTypeEmpty/g' "$file"
        sed -i 's/cudaGraphNodeTypeWaitEvent/musaGraphNodeTypeWaitEvent/g' "$file"
        sed -i 's/cudaGraphNodeTypeEventRecord/musaGraphNodeTypeEventRecord/g' "$file"
        sed -i 's/cudaGraphNodeTypeExtSemaphoreSignal/musaGraphNodeTypeExtSemaphoreSignal/g' "$file"
        sed -i 's/cudaGraphNodeTypeExtSemaphoreWait/musaGraphNodeTypeExtSemaphoreWait/g' "$file"
        sed -i 's/cudaGraphNodeTypeMemAlloc/musaGraphNodeTypeMemAlloc/g' "$file"
        sed -i 's/cudaGraphNodeTypeMemFree/musaGraphNodeTypeMemFree/g' "$file"
        sed -i 's/cudaGraphNodeTypeConditional/musaGraphNodeTypeConditional/g' "$file"
        sed -i 's/cudaGraphExecUpdateResult/musaGraphExecUpdateResult/g' "$file"
        sed -i 's/cudaGraphExecUpdateSuccess/musaGraphExecUpdateSuccess/g' "$file"
        sed -i 's/cudaEventCreateWithFlags/musaEventCreateWithFlags/g' "$file"
        sed -i 's/cudaFuncSetAttribute/musaFuncSetAttribute/g' "$file"
        sed -i 's/cudaFuncAttribute/musaFuncAttribute/g' "$file"
        sed -i 's/cudaFuncAttributeMaxDynamicSharedMemorySize/musaFuncAttributeMaxDynamicSharedMemorySize/g' "$file"
        sed -i 's/cudaLaunchKernel/musaLaunchKernel/g' "$file"
        sed -i 's/cudaGraphChildGraphNodeGetGraph/musaGraphChildGraphNodeGetGraph/g' "$file"
        sed -i 's/cudaOccupancyMaxActiveBlocksPerMultiprocessor/musaOccupancyMaxActiveBlocksPerMultiprocessor/g' "$file"
        sed -i 's/cudaEventQuery/musaEventQuery/g' "$file"

    done
done

echo "--- Source file porting complete ---"
echo "=== MUSA porting complete ==="
echo ""
echo "Created:"
echo "  taskflow/musa/  (ported headers)"
echo "  examples/musa/  (ported examples)"
echo "  unittests/musa/ (ported unit tests)"
