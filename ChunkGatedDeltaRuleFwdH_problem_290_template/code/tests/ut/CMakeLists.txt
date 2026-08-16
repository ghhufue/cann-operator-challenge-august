cmake_minimum_required(VERSION 3.16.0)
project(chunk_gated_delta_rule_fwd_h_ut CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ASCEND_HOME_PATH environment variable is not set!")
endif()

message(STATUS "ASCEND_HOME_PATH: $ENV{ASCEND_HOME_PATH}")

if(EXISTS "$ENV{ASCEND_HOME_PATH}/${CMAKE_HOST_SYSTEM_PROCESSOR}-linux")
    set(CANN_ARCH_DIR "$ENV{ASCEND_HOME_PATH}/${CMAKE_HOST_SYSTEM_PROCESSOR}-linux")
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x64)$" AND
       EXISTS "$ENV{ASCEND_HOME_PATH}/x86_64-linux")
    set(CANN_ARCH_DIR "$ENV{ASCEND_HOME_PATH}/x86_64-linux")
elseif(EXISTS "$ENV{ASCEND_HOME_PATH}/aarch64-linux")
    set(CANN_ARCH_DIR "$ENV{ASCEND_HOME_PATH}/aarch64-linux")
else()
    message(FATAL_ERROR "Cannot locate architecture-specific CANN directory")
endif()

message(STATUS "CANN architecture directory: ${CANN_ARCH_DIR}")

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuildGoogleTest.cmake)

# DTYPE 宏定义（根据算子原型和测试用例生成）
add_definitions(
    -DDTYPE_K=float
    -DDTYPE_W=float
    -DDTYPE_U=float
    -DDTYPE_G=float
    -DDTYPE_INITIAL_STATE=float
    -DDTYPE_CU_SEQLENS=int64_t
    -DDTYPE_CHUNK_INDICES=int64_t
    -DDTYPE_H=float
    -DDTYPE_V=float
    -DDTYPE_FINAL_STATE=float
)

set(UT_COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)
set(UT_INFERSHAPE_COMMON_SRCS
    ${UT_COMMON_DIR}/infershape_case_executor.cpp
    ${UT_COMMON_DIR}/infershape_context_faker.cpp
)

set(UT_TILING_COMMON_SRCS
    ${UT_COMMON_DIR}/tiling_case_executor.cpp
    ${UT_COMMON_DIR}/tiling_context_faker.cpp
)

set(UT_COMMON_INCLUDE_DIRS
    ${UT_COMMON_DIR}
    $ENV{ASCEND_HOME_PATH}/include
    ${CANN_ARCH_DIR}/include
    ${CANN_ARCH_DIR}/include/base
    ${CANN_ARCH_DIR}/include/base/context_builder
    ${CANN_ARCH_DIR}/asc/include
    ${CANN_ARCH_DIR}/asc/include/tiling
    ${CANN_ARCH_DIR}/pkg_inc
    ${CANN_ARCH_DIR}/pkg_inc/op_common
    ${CANN_ARCH_DIR}/pkg_inc/base
    ${CANN_ARCH_DIR}/pkg_inc/exe_graph
    ${CANN_ARCH_DIR}/pkg_inc/graph
)

set(UT_COMMON_LIB_DIRS
    $ENV{ASCEND_HOME_PATH}/lib64
    ${CANN_ARCH_DIR}/lib64
)

add_compile_options(
    -Wall
    -Wno-deprecated-declarations
    -Wno-unused-variable
    -fno-access-control
)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

add_subdirectory(op_host)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../../op_api)
    add_subdirectory(op_api)
endif()
add_subdirectory(op_kernel)
