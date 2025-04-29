#pragma once
#include <luisa/core/stl/unordered_map.h>
#include <luisa/runtime/command_list.h>
#include <luisa/tensor/expression.h>
#include <luisa/tensor/pass/shader_manager.h>
#include <luisa/runtime/rhi/argument.h>
namespace luisa::compute {
struct FallbackTensorKernel;
struct FallbackTensorCallback {
    luisa::unordered_map<TensorData *, Argument::Buffer> *args;
    FallbackTensorKernel *kernel;
    ShaderManager* shader_manager;
    Argument::Buffer allocate_temp_buffer(size_t size_bytes);
    Argument::Buffer get_tensor_buffer(TensorData *data);
};
struct ITensorExprExecutor {
    ~ITensorExprExecutor() = default;
    ITensorExprExecutor() = default;
    ITensorExprExecutor(ITensorExprExecutor const &) = delete;
    ITensorExprExecutor(ITensorExprExecutor &&) = delete;
    virtual void execute(FallbackTensorCallback *callback, CommandList &cmdlist) const = 0;
};
}// namespace luisa::compute