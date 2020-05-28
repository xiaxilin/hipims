cmake_minimum_required(VERSION 2.8)

set(EXE hipims-flood-mnodes)
set(SRCS cuda_mnodes_flood_solver.cu)

#set (CUDA_NVCC_FLAGS "-arch=sm_20 -Xptxas=v")
#set (CUDA_SEPARABLE_COMPILATION ON)

file(GLOB_RECURSE gc_headers "${PROJECT_SOURCE_DIR}/lib/include/*.h")

set (GC_INCLUDE_DIRS "")
foreach (_header_file ${gc_headers})
	get_filename_component(_dir ${_header_file} DIRECTORY)
	list (APPEND GC_INCLUDE_DIRS ${_dir})
endforeach()
list(REMOVE_DUPLICATES GC_INCLUDE_DIRS)

include_directories(${GC_INCLUDE_DIRS})

cuda_add_executable(${EXE} ${SRCS})
add_dependencies(${EXE} lib cuda_lib)

#set (CUDA_NVCC_FLAGS "-arch=sm_35")

target_link_libraries(${EXE} lib cuda_lib)
target_link_libraries(${EXE} ${MPI_C_LIBRARIES})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/release/bin)

