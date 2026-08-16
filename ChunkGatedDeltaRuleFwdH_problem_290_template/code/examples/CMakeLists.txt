cmake_minimum_required(VERSION 3.14)
project(ACLNN_EXAMPLE)

add_compile_options(-std=c++17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "./bin")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

add_executable(test_aclnn_chunk_gated_delta_rule_fwd_h
test_aclnn_chunk_gated_delta_rule_fwd_h.cpp)

if(NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
    set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
else()
    set(ASCEND_PATH "/usr/local/Ascend/cann")
endif()

get_filename_component(OP_PROJECT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(OP_BUILD_DIR "${OP_PROJECT_DIR}/build")
if(EXISTS "${ASCEND_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux")
    set(CANN_ARCH_DIR "${ASCEND_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x64)$" AND EXISTS "${ASCEND_PATH}/x86_64-linux")
    set(CANN_ARCH_DIR "${ASCEND_PATH}/x86_64-linux")
elseif(EXISTS "${ASCEND_PATH}/aarch64-linux")
    set(CANN_ARCH_DIR "${ASCEND_PATH}/aarch64-linux")
else()
    set(CANN_ARCH_DIR "${ASCEND_PATH}")
endif()

# Re-evaluate these paths on every configure so a previous installed package
# does not remain cached after a fresh local operator build.
unset(CUSTOM_OP_INCLUDE_DIR CACHE)
unset(CUSTOM_OP_LIBRARY CACHE)

find_path(CUSTOM_OP_INCLUDE_DIR
    NAMES aclnn_chunk_gated_delta_rule_fwd_h.h
    PATHS
        ${OP_BUILD_DIR}/autogen
        ${OP_BUILD_DIR}/packages/vendors/chunk_gated_delta_rule_fwd_h_custom/op_api/include
        ${ASCEND_PATH}/opp/vendors/chunk_gated_delta_rule_fwd_h_custom/op_api/include
        /usr/local/Ascend/opp/vendors/chunk_gated_delta_rule_fwd_h_custom/op_api/include
        $ENV{HOME}/Ascend/opp/vendors/chunk_gated_delta_rule_fwd_h_custom/op_api/include
)

if(NOT CUSTOM_OP_INCLUDE_DIR)
    message(FATAL_ERROR "未找到自定义算子头文件 aclnn_chunk_gated_delta_rule_fwd_h.h，请先安装算子包")
endif()
message(STATUS "自定义算子头文件目录: ${CUSTOM_OP_INCLUDE_DIR}")

find_library(CUSTOM_OP_LIBRARY cust_opapi
    PATHS
        ${OP_BUILD_DIR}/op_host
        ${OP_BUILD_DIR}/packages/vendors/chunk_gated_delta_rule_fwd_h_custom/op_api/lib
        ${ASCEND_PATH}/opp/vendors/chunk_gated_delta_rule_fwd_h_custom/op_api/lib
        /usr/local/Ascend/opp/vendors/chunk_gated_delta_rule_fwd_h_custom/op_api/lib
        $ENV{HOME}/Ascend/opp/vendors/chunk_gated_delta_rule_fwd_h_custom/op_api/lib
)

if(NOT CUSTOM_OP_LIBRARY)
    message(FATAL_ERROR "未找到自定义算子库 libcust_opapi.so，请先安装算子包")
endif()

find_library(ASCENDCL_LIBRARY
    NAMES ascendcl acl_rt
    PATHS ${ASCEND_PATH}/lib64 ${CANN_ARCH_DIR}/lib64
    NO_DEFAULT_PATH
)
find_library(NNOPBASE_LIBRARY
    NAMES nnopbase
    PATHS ${ASCEND_PATH}/lib64 ${CANN_ARCH_DIR}/lib64
    NO_DEFAULT_PATH
)
if(NOT ASCENDCL_LIBRARY OR NOT NNOPBASE_LIBRARY)
    message(FATAL_ERROR "未找到 ACL/NNOP 运行库，ASCEND_PATH=${ASCEND_PATH}")
endif()

include_directories(
    ${ASCEND_PATH}/include
    ${CUSTOM_OP_INCLUDE_DIR}
)

target_link_libraries(test_aclnn_chunk_gated_delta_rule_fwd_h PRIVATE
  ${CUSTOM_OP_LIBRARY}
  ${ASCENDCL_LIBRARY}
  ${NNOPBASE_LIBRARY}
)
get_filename_component(CUSTOM_OP_LIB_DIR ${CUSTOM_OP_LIBRARY} DIRECTORY)
target_link_options(test_aclnn_chunk_gated_delta_rule_fwd_h PRIVATE
  "-Wl,-rpath,${CUSTOM_OP_LIB_DIR}"
  "-Wl,-rpath,${ASCEND_PATH}/lib64"
  "-Wl,-rpath,${CANN_ARCH_DIR}/lib64"
  "-Wl,-rpath,${ASCEND_PATH}/devlib"
  "-Wl,-rpath,${CANN_ARCH_DIR}/devlib"
)

install(TARGETS test_aclnn_chunk_gated_delta_rule_fwd_h DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
