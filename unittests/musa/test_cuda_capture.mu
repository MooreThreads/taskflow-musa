#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/musa/musaflow.hpp>

void __global__ testKernel() {}

TEST_CASE("cudaFlowCapturer.noEventError") {
  tf::cudaFlow f;
  f.capture([](tf::cudaFlowCapturer& cpt) {
    cpt.on([] (musaStream_t stream) {
      testKernel<<<256,256,0,stream>>>();
    });
    REQUIRE((cudaGetLastError() == musaSuccess));
  });
  REQUIRE((cudaGetLastError() == musaSuccess));
}
