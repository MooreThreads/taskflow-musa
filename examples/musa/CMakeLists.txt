# -----------------------------------------------------------------------------
# MUSA examples (ported from CUDA)
# -----------------------------------------------------------------------------

list(APPEND TF_MUSA_EXAMPLES
  cuda_device_property
  cuda_saxpy
  cuda_matmul
  cuda_knn
  cuda_saxpy_update
)

foreach(musa_example IN LISTS TF_MUSA_EXAMPLES)
  set(MU_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${musa_example}.mu")
  set(OBJ_FILE "${CMAKE_CURRENT_BINARY_DIR}/${musa_example}.o")

  add_custom_command(
    OUTPUT ${OBJ_FILE}
    COMMAND ${MUSA_MCC_EXECUTABLE} ${MUSA_MCC_FLAGS}
      -I${PROJECT_SOURCE_DIR}
      -I${MUSA_INCLUDE_DIRS}
      -c ${MU_SOURCE} -o ${OBJ_FILE}
    DEPENDS ${MU_SOURCE}
    COMMENT "Compiling MUSA example: ${musa_example}.mu"
  )

  add_executable(${musa_example}_musa ${OBJ_FILE})
  set_target_properties(${musa_example}_musa PROPERTIES LINKER_LANGUAGE CXX)
  target_link_libraries(${musa_example}_musa
    ${PROJECT_NAME} ${ATOMIC_LIBRARY} Threads::Threads ${MUSA_LIBRARIES} tf::default_settings
  )
endforeach()
