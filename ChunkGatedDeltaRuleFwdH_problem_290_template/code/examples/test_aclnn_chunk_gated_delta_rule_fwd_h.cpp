#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnn_chunk_gated_delta_rule_fwd_h.h"

namespace {

namespace fs = std::filesystem;

struct DeviceTensor {
    aclTensor* tensor = nullptr;
    void* deviceData = nullptr;
    size_t bytes = 0;
};

void CheckAcl(aclError result, const char* operation)
{
    if (result != ACL_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + " failed, acl error=" + std::to_string(result));
    }
}

std::vector<uint8_t> ReadBinary(const fs::path& path)
{
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("Cannot open " + path.string());
    }
    const std::streamsize size = stream.tellg();
    if (size < 0) {
        throw std::runtime_error("Cannot determine file size for " + path.string());
    }
    std::vector<uint8_t> data(static_cast<size_t>(size));
    stream.seekg(0);
    if (size != 0) {
        stream.read(reinterpret_cast<char*>(data.data()), size);
    }
    if (!stream) {
        throw std::runtime_error("Cannot read " + path.string());
    }
    return data;
}

std::string ReadText(const fs::path& path)
{
    const std::vector<uint8_t> bytes = ReadBinary(path);
    return std::string(bytes.begin(), bytes.end());
}

void WriteBinary(const fs::path& path, const std::vector<uint8_t>& data)
{
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("Cannot create " + path.string());
    }
    if (!data.empty()) {
        stream.write(
            reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size()));
    }
    if (!stream) {
        throw std::runtime_error("Cannot write " + path.string());
    }
}

int64_t JsonInteger(const std::string& json, const std::string& key)
{
    const std::string marker = "\"" + key + "\"";
    const size_t keyPos = json.find(marker);
    if (keyPos == std::string::npos) {
        throw std::runtime_error("Missing integer key in manifest: " + key);
    }
    const size_t colon = json.find(':', keyPos + marker.size());
    if (colon == std::string::npos) {
        throw std::runtime_error("Malformed manifest key: " + key);
    }
    const size_t begin = json.find_first_of("-0123456789", colon + 1);
    if (begin == std::string::npos) {
        throw std::runtime_error("Missing integer value for manifest key: " + key);
    }
    size_t end = begin;
    if (json[end] == '-') {
        ++end;
    }
    while (end < json.size() && json[end] >= '0' && json[end] <= '9') {
        ++end;
    }
    return std::stoll(json.substr(begin, end - begin));
}

size_t DataTypeBytes(aclDataType dataType)
{
    switch (dataType) {
        case ACL_BF16:
        case ACL_FLOAT16:
        case ACL_INT16:
        case ACL_UINT16:
            return 2;
        case ACL_FLOAT:
        case ACL_INT32:
        case ACL_UINT32:
            return 4;
        case ACL_INT64:
        case ACL_UINT64:
        case ACL_DOUBLE:
            return 8;
        case ACL_INT8:
        case ACL_UINT8:
        case ACL_BOOL:
            return 1;
        default:
            throw std::runtime_error("Unsupported ACL data type");
    }
}

size_t TensorBytes(const std::vector<int64_t>& shape, aclDataType dataType)
{
    size_t elements = 1;
    for (const int64_t dim : shape) {
        if (dim <= 0 || elements > std::numeric_limits<size_t>::max() / static_cast<size_t>(dim)) {
            throw std::runtime_error("Invalid or overflowing tensor shape");
        }
        elements *= static_cast<size_t>(dim);
    }
    return elements * DataTypeBytes(dataType);
}

std::vector<int64_t> ContiguousStrides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t index = static_cast<int64_t>(shape.size()) - 2; index >= 0; --index) {
        strides[static_cast<size_t>(index)] =
            shape[static_cast<size_t>(index + 1)] * strides[static_cast<size_t>(index + 1)];
    }
    return strides;
}

DeviceTensor CreateTensor(
    const std::vector<int64_t>& shape,
    aclDataType dataType,
    const std::vector<uint8_t>* hostData)
{
    DeviceTensor result;
    result.bytes = TensorBytes(shape, dataType);
    if (hostData != nullptr && hostData->size() != result.bytes) {
        throw std::runtime_error(
            "Input file size mismatch: expected " + std::to_string(result.bytes) +
            ", got " + std::to_string(hostData->size()));
    }

    CheckAcl(
        aclrtMalloc(&result.deviceData, result.bytes, ACL_MEM_MALLOC_HUGE_FIRST),
        "aclrtMalloc");
    if (hostData != nullptr) {
        CheckAcl(
            aclrtMemcpy(
                result.deviceData,
                result.bytes,
                hostData->data(),
                hostData->size(),
                ACL_MEMCPY_HOST_TO_DEVICE),
            "aclrtMemcpy host-to-device");
    }

    const std::vector<int64_t> strides = ContiguousStrides(shape);
    result.tensor = aclCreateTensor(
        shape.data(),
        static_cast<uint64_t>(shape.size()),
        dataType,
        strides.data(),
        0,
        ACL_FORMAT_ND,
        shape.data(),
        static_cast<uint64_t>(shape.size()),
        result.deviceData);
    if (result.tensor == nullptr) {
        aclrtFree(result.deviceData);
        result.deviceData = nullptr;
        throw std::runtime_error("aclCreateTensor failed");
    }
    return result;
}

DeviceTensor CreateInput(
    const fs::path& path,
    const std::vector<int64_t>& shape,
    aclDataType dataType)
{
    const std::vector<uint8_t> data = ReadBinary(path);
    return CreateTensor(shape, dataType, &data);
}

void DestroyTensor(DeviceTensor& tensor)
{
    if (tensor.tensor != nullptr) {
        aclDestroyTensor(tensor.tensor);
        tensor.tensor = nullptr;
    }
    if (tensor.deviceData != nullptr) {
        aclrtFree(tensor.deviceData);
        tensor.deviceData = nullptr;
    }
}

void CopyOutput(const DeviceTensor& tensor, const fs::path& path)
{
    std::vector<uint8_t> hostData(tensor.bytes);
    CheckAcl(
        aclrtMemcpy(
            hostData.data(),
            hostData.size(),
            tensor.deviceData,
            tensor.bytes,
            ACL_MEMCPY_DEVICE_TO_HOST),
        "aclrtMemcpy device-to-host");
    WriteBinary(path, hostData);
}

int RunCase(const fs::path& caseDir, int32_t deviceId)
{
    const std::string manifest = ReadText(caseDir / "manifest.json");
    const int64_t totalTokens = JsonInteger(manifest, "total_tokens");
    const int64_t chunkCount = JsonInteger(manifest, "total_chunks");
    const int64_t valueHeads = JsonInteger(manifest, "hv");
    const int64_t keyHeads = JsonInteger(manifest, "hk");
    const int64_t dimension = JsonInteger(manifest, "dim");

    const fs::path cuSeqlensPath = caseDir / "input_cu_seqlens.bin";
    const size_t cuSeqlensBytes = fs::file_size(cuSeqlensPath);
    if (cuSeqlensBytes < 2 * sizeof(int64_t) || cuSeqlensBytes % sizeof(int64_t) != 0) {
        throw std::runtime_error("Invalid input_cu_seqlens.bin size");
    }
    const int64_t sequenceCount =
        static_cast<int64_t>(cuSeqlensBytes / sizeof(int64_t)) - 1;
    const bool hasInitialState = fs::exists(caseDir / "input_initial_state.bin");

    std::cout << "Running case " << caseDir.filename().string()
              << ": device=" << deviceId
              << ", T=" << totalTokens
              << ", N=" << sequenceCount
              << ", NT=" << chunkCount
              << ", HV/HK=" << valueHeads << "/" << keyHeads
              << ", D=" << dimension
              << ", initial_state=" << (hasInitialState ? "yes" : "no")
              << std::endl;

    aclrtStream stream = nullptr;
    CheckAcl(aclInit(nullptr), "aclInit");
    uint32_t deviceCount = 0;
    CheckAcl(aclrtGetDeviceCount(&deviceCount), "aclrtGetDeviceCount");
    if (deviceId < 0 || static_cast<uint32_t>(deviceId) >= deviceCount) {
        throw std::runtime_error(
            "Logical device " + std::to_string(deviceId) + " is unavailable; device count=" +
            std::to_string(deviceCount));
    }
    CheckAcl(aclrtSetDevice(deviceId), "aclrtSetDevice");
    CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream");

    DeviceTensor k = CreateInput(
        caseDir / "input_k.bin", {1, totalTokens, keyHeads, dimension}, ACL_BF16);
    DeviceTensor w = CreateInput(
        caseDir / "input_w.bin", {1, valueHeads, totalTokens, dimension}, ACL_BF16);
    DeviceTensor u = CreateInput(
        caseDir / "input_u.bin", {1, valueHeads, totalTokens, dimension}, ACL_BF16);
    DeviceTensor g = CreateInput(
        caseDir / "input_g.bin", {1, valueHeads, totalTokens}, ACL_FLOAT);
    DeviceTensor cuSeqlens = CreateInput(
        cuSeqlensPath, {sequenceCount + 1}, ACL_INT64);
    DeviceTensor chunkIndices = CreateInput(
        caseDir / "input_chunk_indices.bin", {chunkCount, 2}, ACL_INT64);

    DeviceTensor initialState;
    if (hasInitialState) {
        initialState = CreateInput(
            caseDir / "input_initial_state.bin",
            {sequenceCount, valueHeads, dimension, dimension},
            ACL_BF16);
    }

    DeviceTensor h = CreateTensor(
        {1, valueHeads, chunkCount, dimension, dimension}, ACL_BF16, nullptr);
    DeviceTensor v = CreateTensor(
        {1, valueHeads, totalTokens, dimension}, ACL_BF16, nullptr);
    DeviceTensor finalState = CreateTensor(
        {sequenceCount, valueHeads, dimension, dimension}, ACL_BF16, nullptr);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    CheckAcl(
        aclnnChunkGatedDeltaRuleFwdHGetWorkspaceSize(
            k.tensor,
            w.tensor,
            u.tensor,
            g.tensor,
            hasInitialState ? initialState.tensor : nullptr,
            cuSeqlens.tensor,
            chunkIndices.tensor,
            64,
            h.tensor,
            v.tensor,
            finalState.tensor,
            &workspaceSize,
            &executor),
        "aclnnChunkGatedDeltaRuleFwdHGetWorkspaceSize");

    void* workspace = nullptr;
    if (workspaceSize != 0) {
        CheckAcl(
            aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST),
            "aclrtMalloc workspace");
    }
    CheckAcl(
        aclnnChunkGatedDeltaRuleFwdH(
            workspace, workspaceSize, executor, stream),
        "aclnnChunkGatedDeltaRuleFwdH");
    CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");

    CopyOutput(h, caseDir / "actual_h.bin");
    CopyOutput(v, caseDir / "actual_v.bin");
    CopyOutput(finalState, caseDir / "actual_final_state.bin");

    std::cout << "Wrote actual_h.bin, actual_v.bin, and actual_final_state.bin"
              << std::endl;

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    DestroyTensor(k);
    DestroyTensor(w);
    DestroyTensor(u);
    DestroyTensor(g);
    DestroyTensor(initialState);
    DestroyTensor(cuSeqlens);
    DestroyTensor(chunkIndices);
    DestroyTensor(h);
    DestroyTensor(v);
    DestroyTensor(finalState);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: " << argv[0] << " CASE_DIR [LOGICAL_DEVICE_ID]" << std::endl;
        return 2;
    }
    try {
        const int32_t deviceId = argc == 3 ? std::stoi(argv[2]) : 0;
        return RunCase(fs::absolute(argv[1]), deviceId);
    } catch (const std::exception& error) {
        std::cerr << "NPU validation runner failed: " << error.what() << std::endl;
        return 1;
    }
}
