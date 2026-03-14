#pragma once

#include "../taskflow.hpp"
#include "musa_graph.hpp"
#include "musa_graph_exec.hpp"
#include "algorithm/single_task.hpp"

/**
@file taskflow/cuda/cudaflow.hpp
@brief cudaFlow include file
*/

namespace tf {

/**
@brief default smart pointer type to manage a `musaGraph_t` object with unique ownership
*/
using cudaGraph = cudaGraphBase<cudaGraphCreator, cudaGraphDeleter>;

/**
@brief default smart pointer type to manage a `musaGraphExec_t` object with unique ownership
*/
using cudaGraphExec = cudaGraphExecBase<cudaGraphExecCreator, cudaGraphExecDeleter>;

}  // end of namespace tf -----------------------------------------------------


