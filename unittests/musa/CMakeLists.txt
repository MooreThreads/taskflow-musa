# MUSA unit tests (ported from CUDA)

list(APPEND TF_MUSA_UNITTESTS
  test_cuda_objects
  test_cuda_basics
  test_cuda_updates
  test_cuda_matrix
  test_cuda_kmeans
  test_cuda_for_each
  test_cuda_transform
)

foreach(musatest IN LISTS TF_MUSA_UNITTESTS)
  set(MU_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${musatest}.mu")
  set(OBJ_FILE "${CMAKE_CURRENT_BINARY_DIR}/${musatest}.o")

  add_custom_command(
    OUTPUT ${OBJ_FILE}
    COMMAND ${MUSA_MCC_EXECUTABLE} ${MUSA_MCC_FLAGS}
      -I${PROJECT_SOURCE_DIR}
      -I${MUSA_INCLUDE_DIRS}
      -I${TF_3RD_PARTY_DIR}/doctest
      -c ${MU_SOURCE} -o ${OBJ_FILE}
    DEPENDS ${MU_SOURCE}
    COMMENT "Compiling MUSA test: ${musatest}.mu"
  )

  add_executable(${musatest}_musa ${OBJ_FILE})
  set_target_properties(${musatest}_musa PROPERTIES LINKER_LANGUAGE CXX)
  target_link_libraries(${musatest}_musa
    ${PROJECT_NAME} ${ATOMIC_LIBRARY} ${MUSA_LIBRARIES} tf::default_settings
  )
  target_include_directories(${musatest}_musa PRIVATE ${TF_3RD_PARTY_DIR}/doctest)

  doctest_discover_tests(${musatest}_musa)
endforeach()
