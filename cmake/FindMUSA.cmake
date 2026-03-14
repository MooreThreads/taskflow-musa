# FindMUSA.cmake -- Locate MUSA SDK and configure mcc compiler
#
# Sets:
#   MUSA_FOUND, MUSA_TOOLKIT_ROOT, MUSA_INCLUDE_DIRS,
#   MUSA_LIBRARIES, MUSA_MCC_EXECUTABLE, MUSA_ARCH

find_path(MUSA_TOOLKIT_ROOT
    NAMES include/musa.h
    PATHS ENV MUSA_HOME /usr/local/musa /usr/local/musa-*
    DOC "MUSA SDK root directory"
)

if(MUSA_TOOLKIT_ROOT)
    set(MUSA_FOUND TRUE)
    set(MUSA_INCLUDE_DIRS "${MUSA_TOOLKIT_ROOT}/include")

    find_program(MUSA_MCC_EXECUTABLE NAMES mcc
        PATHS "${MUSA_TOOLKIT_ROOT}/bin" DOC "MUSA mcc compiler")

    find_library(MUSA_RUNTIME_LIBRARY NAMES musart
        PATHS "${MUSA_TOOLKIT_ROOT}/lib" DOC "MUSA runtime library")
    find_library(MUSA_DRIVER_LIBRARY NAMES musa
        PATHS "${MUSA_TOOLKIT_ROOT}/lib" DOC "MUSA driver library")

    set(MUSA_LIBRARIES ${MUSA_RUNTIME_LIBRARY} ${MUSA_DRIVER_LIBRARY})

    if(NOT MUSA_ARCH)
        set(MUSA_ARCH "22" CACHE STRING "MUSA GPU arch (default: mp_22 for S4000)")
    endif()

    message(STATUS "MUSA SDK: ${MUSA_TOOLKIT_ROOT}, mcc: ${MUSA_MCC_EXECUTABLE}, arch: mp_${MUSA_ARCH}")
else()
    set(MUSA_FOUND FALSE)
    message(WARNING "MUSA SDK not found. Set MUSA_HOME.")
endif()

mark_as_advanced(MUSA_TOOLKIT_ROOT MUSA_MCC_EXECUTABLE MUSA_RUNTIME_LIBRARY MUSA_DRIVER_LIBRARY)
