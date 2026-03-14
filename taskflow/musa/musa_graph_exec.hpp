#pragma once

#include "musa_graph.hpp"


namespace tf {

// ----------------------------------------------------------------------------
// cudaGraphExec
// ----------------------------------------------------------------------------

/**
@class cudaGraphExecCreator
@brief class to create functors for constructing executable CUDA graphs

This class provides an overloaded function call operator to create a
new executable CUDA graph using `musaGraphCreate`. 
*/
class cudaGraphExecCreator {
  
  public:

  /**
  @brief returns a null executable CUDA graph
  */
  musaGraphExec_t operator () () const { 
    return nullptr;
  }
  
  /**
  @brief returns the given executable graph
  */
  musaGraphExec_t operator () (musaGraphExec_t exec) const {
    return exec;
  }

  /**
  @brief returns a newly instantiated executable graph from the given CUDA graph
  */
  musaGraphExec_t operator () (musaGraph_t graph) const {
    musaGraphExec_t exec;
    TF_CHECK_CUDA(
      musaGraphInstantiate(&exec, graph, nullptr, nullptr, 0),
      "failed to create an executable graph"
    );
    return exec;
  }

  /**
  @brief returns a newly instantiated executable graph from the given CUDA graph
  */
  template <typename C, typename D>
  musaGraphExec_t operator () (const cudaGraphBase<C, D>& graph) const {
    return this->operator()(graph.get());
  }
};
  
/**
@class cudaGraphExecDeleter
@brief class to create a functor for deleting an executable CUDA graph

This class provides an overloaded function call operator to safely
destroy a CUDA graph using `musaGraphDestroy`.
*/
class cudaGraphExecDeleter {

  public:

  /**
   @brief deletes an executable CUDA graph
   
   Calls `musaGraphDestroy` to release the CUDA graph resource if it is valid.
   
   @param executable the executable CUDA graph to be destroyed
  */
  void operator () (musaGraphExec_t executable) const {
    musaGraphExecDestroy(executable);
  }
};

/**
@class cudaGraphExecBase

@brief class to create an executable CUDA graph with unique ownership

@tparam Creator functor to create the stream (used in constructor)
@tparam Deleter functor to delete the stream (used in destructor)

This class wraps a `musaGraphExec_t` handle with `std::unique_ptr` to ensure proper 
resource management and automatic cleanup.
*/
template <typename Creator, typename Deleter>
class cudaGraphExecBase : public std::unique_ptr<std::remove_pointer_t<musaGraphExec_t>, Deleter> {
  
  static_assert(std::is_pointer_v<musaGraphExec_t>, "musaGraphExec_t is not a pointer type");

  public:
  
  /**
  @brief base std::unique_ptr type
  */
  using base_type = std::unique_ptr<std::remove_pointer_t<musaGraphExec_t>, Deleter>;

  /**
  @brief constructs a `cudaGraphExec` object by passing the given arguments to the executable CUDA graph creator

  Constructs a `cudaGraphExec` object by passing the given arguments to the executable CUDA graph creator

  @param args arguments to pass to the executable CUDA graph creator
  */
  template <typename... ArgsT>
  explicit cudaGraphExecBase(ArgsT&& ... args) : base_type(
    Creator{}(std::forward<ArgsT>(args)...), Deleter()
  ) {}  

  /**
  @brief constructs a `cudaGraphExec` from the given rhs using move semantics
  */
  cudaGraphExecBase(cudaGraphExecBase&&) = default;

  /**
  @brief assign the rhs to `*this` using move semantics
  */
  cudaGraphExecBase& operator = (cudaGraphExecBase&&) = default;

  // ----------------------------------------------------------------------------------------------
  // Update Methods
  // ----------------------------------------------------------------------------------------------

  /**
  @brief updates parameters of a host task

  This method updates the parameter of the given host task (similar to tf::cudaFlow::host).
  */
  template <typename C>
  void host(cudaTask task, C&& callable, void* user_data);
  
  /**
  @brief updates parameters of a kernel task

  The method is similar to tf::cudaFlow::kernel but operates on a task
  of type tf::cudaTaskType::KERNEL.
  The kernel function name must NOT change.
  */
  template <typename F, typename... ArgsT>
  void kernel(
    cudaTask task, dim3 g, dim3 b, size_t shm, F f, ArgsT... args
  );
  
  /**
  @brief updates parameters of a memset task

  The method is similar to tf::cudaFlow::memset but operates on a task
  of type tf::cudaTaskType::MEMSET.
  The source/destination memory may have different address values but
  must be allocated from the same contexts as the original
  source/destination memory.
  */
  void memset(cudaTask task, void* dst, int ch, size_t count);

  /**
  @brief updates parameters of a memcpy task

  The method is similar to tf::cudaFlow::memcpy but operates on a task
  of type tf::cudaTaskType::MEMCPY.
  The source/destination memory may have different address values but
  must be allocated from the same contexts as the original
  source/destination memory.
  */
  void memcpy(cudaTask task, void* tgt, const void* src, size_t bytes);
  
  /**
  @brief updates parameters of a memset task to a zero task

  The method is similar to tf::cudaFlow::zero but operates on
  a task of type tf::cudaTaskType::MEMSET.

  The source/destination memory may have different address values but
  must be allocated from the same contexts as the original
  source/destination memory.
  */
  template <typename T, std::enable_if_t<
    is_pod_v<T> && (sizeof(T)==1 || sizeof(T)==2 || sizeof(T)==4), void>* = nullptr
  >
  void zero(cudaTask task, T* dst, size_t count);

  /**
  @brief updates parameters of a memset task to a fill task

  The method is similar to tf::cudaFlow::fill but operates on a task
  of type tf::cudaTaskType::MEMSET.

  The source/destination memory may have different address values but
  must be allocated from the same contexts as the original
  source/destination memory.
  */
  template <typename T, std::enable_if_t<
    is_pod_v<T> && (sizeof(T)==1 || sizeof(T)==2 || sizeof(T)==4), void>* = nullptr
  >
  void fill(cudaTask task, T* dst, T value, size_t count);
  
  /**
  @brief updates parameters of a memcpy task to a copy task

  The method is similar to tf::cudaFlow::copy but operates on a task
  of type tf::cudaTaskType::MEMCPY.
  The source/destination memory may have different address values but
  must be allocated from the same contexts as the original
  source/destination memory.
  */
  template <typename T,
    std::enable_if_t<!std::is_same_v<T, void>, void>* = nullptr
  >
  void copy(cudaTask task, T* tgt, const T* src, size_t num);
  
  //---------------------------------------------------------------------------
  // Algorithm Primitives
  //---------------------------------------------------------------------------

  /**
  @brief updates a single-threaded kernel task

  This method is similar to cudaFlow::single_task but operates
  on an existing task.
  */
  template <typename C>
  void single_task(cudaTask task, C c);
  
  /**
  @brief updates parameters of a `for_each` kernel task created from the CUDA graph of `*this`
  */
  template <typename I, typename C, typename E = cudaDefaultExecutionPolicy>
  void for_each(cudaTask task, I first, I last, C callable);
  
  /**
  @brief updates parameters of a `for_each_index` kernel task created from the CUDA graph of `*this`
  */
  template <typename I, typename C, typename E = cudaDefaultExecutionPolicy>
  void for_each_index(cudaTask task, I first, I last, I step, C callable);

  /**
  @brief updates parameters of a `transform` kernel task created from the CUDA graph of `*this`
  */
  template <typename I, typename O, typename C, typename E = cudaDefaultExecutionPolicy>
  void transform(cudaTask task, I first, I last, O output, C c);

  /**
  @brief updates parameters of a `transform` kernel task created from the CUDA graph of `*this`
  */
  template <typename I1, typename I2, typename O, typename C, typename E = cudaDefaultExecutionPolicy>
  void transform(cudaTask task, I1 first1, I1 last1, I2 first2, O output, C c);

  
  private:

  cudaGraphExecBase(const cudaGraphExecBase&) = delete;

  cudaGraphExecBase& operator = (const cudaGraphExecBase&) = delete;
};

// ------------------------------------------------------------------------------------------------
// update methods
// ------------------------------------------------------------------------------------------------

// Function: host
template <typename Creator, typename Deleter>
template <typename C>
void cudaGraphExecBase<Creator, Deleter>::host(cudaTask task, C&& func, void* user_data) {
  musaHostNodeParams p {func, user_data};
  TF_CHECK_CUDA(
    cudaGraphExecHostNodeSetParams(this->get(), task._native_node, &p),
    "failed to update kernel parameters on ", task
  );
}

// Function: update kernel parameters
template <typename Creator, typename Deleter>
template <typename F, typename... ArgsT>
void cudaGraphExecBase<Creator, Deleter>::kernel(
  cudaTask task, dim3 g, dim3 b, size_t s, F f, ArgsT... args
) {
  musaKernelNodeParams p;

  void* arguments[sizeof...(ArgsT)] = { (void*)(&args)... };
  p.func = (void*)f;
  p.gridDim = g;
  p.blockDim = b;
  p.sharedMemBytes = s;
  p.kernelParams = arguments;
  p.extra = nullptr;

  TF_CHECK_CUDA(
    musaGraphExecKernelNodeSetParams(this->get(), task._native_node, &p),
    "failed to update kernel parameters on ", task
  );
}

// Function: update copy parameters
template <typename Creator, typename Deleter>
template <typename T, std::enable_if_t<!std::is_same_v<T, void>, void>*>
void cudaGraphExecBase<Creator, Deleter>::copy(cudaTask task, T* tgt, const T* src, size_t num) {
  auto p = cuda_get_copy_parms(tgt, src, num);
  TF_CHECK_CUDA(
    musaGraphExecMemcpyNodeSetParams(this->get(), task._native_node, &p),
    "failed to update memcpy parameters on ", task
  );
}

// Function: update memcpy parameters
template <typename Creator, typename Deleter>
void cudaGraphExecBase<Creator, Deleter>::memcpy(
  cudaTask task, void* tgt, const void* src, size_t bytes
) {
  auto p = cuda_get_memcpy_parms(tgt, src, bytes);

  TF_CHECK_CUDA(
    musaGraphExecMemcpyNodeSetParams(this->get(), task._native_node, &p),
    "failed to update memcpy parameters on ", task
  );
}

// Procedure: memset
template <typename Creator, typename Deleter>
void cudaGraphExecBase<Creator, Deleter>::memset(cudaTask task, void* dst, int ch, size_t count) {
  auto p = cuda_get_memset_parms(dst, ch, count);
  TF_CHECK_CUDA(
    musaGraphExecMemsetNodeSetParams(this->get(), task._native_node, &p),
    "failed to update memset parameters on ", task
  );
}

// Procedure: fill
template <typename Creator, typename Deleter>
template <typename T, std::enable_if_t<
  is_pod_v<T> && (sizeof(T)==1 || sizeof(T)==2 || sizeof(T)==4), void>*
>
void cudaGraphExecBase<Creator, Deleter>::fill(cudaTask task, T* dst, T value, size_t count) {
  auto p = cuda_get_fill_parms(dst, value, count);
  TF_CHECK_CUDA(
    musaGraphExecMemsetNodeSetParams(this->get(), task._native_node, &p),
    "failed to update memset parameters on ", task
  );
}

// Procedure: zero
template <typename Creator, typename Deleter>
template <typename T, std::enable_if_t<
  is_pod_v<T> && (sizeof(T)==1 || sizeof(T)==2 || sizeof(T)==4), void>*
>
void cudaGraphExecBase<Creator, Deleter>::zero(cudaTask task, T* dst, size_t count) {
  auto p = cuda_get_zero_parms(dst, count);
  TF_CHECK_CUDA(
    musaGraphExecMemsetNodeSetParams(this->get(), task._native_node, &p),
    "failed to update memset parameters on ", task
  );
}

//-------------------------------------------------------------------------------------------------
// forward declaration
//-------------------------------------------------------------------------------------------------

/**
@private
*/
template <typename SC, typename SD>
cudaStreamBase<SC, SD>& cudaStreamBase<SC, SD>::run(musaGraphExec_t exec) {
  TF_CHECK_CUDA(
    musaGraphLaunch(exec, this->get()), "failed to launch a CUDA executable graph"
  );  
  return *this;
}

/**
@private
*/
template <typename SC, typename SD>
template <typename EC, typename ED>
cudaStreamBase<SC, SD>& cudaStreamBase<SC, SD>::run(const cudaGraphExecBase<EC, ED>& exec) {
  return run(exec.get());
}



}  // end of namespace tf -------------------------------------------------------------------------
