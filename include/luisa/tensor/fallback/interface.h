#pragma once
#include <luisa/tensor/tensor_interface.h>
#include <luisa/tensor/pass/expr_topo.h>
#include <luisa/tensor/pass/shader_manager.h>
#include <luisa/vstl/stack_allocator.h>
#include "i_tensor_expr_executor.h"
namespace luisa::compute {
struct TempBufferVisitor : public vstd::StackAllocatorVisitor {
    DeviceInterface *device;
    luisa::vector<uint64_t> freelist;
    uint64 allocate(uint64 size) override {
        auto bf = device->create_buffer(Type::of<float4>(), (size + sizeof(float4) - 1) / sizeof(float4), nullptr);
        freelist.emplace_back(bf.handle);
        return bf.handle;
    }
    void deallocate(uint64 handle) override {
    }
    void dispose(CommandList &cmdlist) {
        cmdlist.add_callback([freelist = std::move(freelist), device = this->device]() {
            for (auto i : freelist) {
                device->destroy_buffer(i);
            }
        });
    }
};
struct FallbackTensorKernel {
    DeviceInterface *device;
    luisa::unique_ptr<TensorBuilder> tensor_builder;
    ExprTopo expr_topo;
    luisa::vector<luisa::unique_ptr<ITensorExprExecutor>> executors;
    luisa::vector<size_t> tensor_sub_nodes;
    BufferCreationInfo tensor_buffer;
    luisa::vector<BufferCreationInfo> outputs;
    TempBufferVisitor temp_buffer_visitor;
    vstd::StackAllocator temp_alloc;

    explicit FallbackTensorKernel(DeviceInterface *device, ShaderManager *shader_manager, luisa::unique_ptr<TensorBuilder> &&t_args);
    ~FallbackTensorKernel();

    void check(luisa::span<Argument::Buffer const> tensors) const;
};

class FallbackTensorInterface : public TensorInterface {
    ShaderManager shader_manager;
public:
    explicit FallbackTensorInterface(Device &device) noexcept : TensorInterface(device), shader_manager(device.impl()) {}
    void *compile_kernel(luisa::unique_ptr<TensorBuilder> &&tensor_builder) noexcept override;
    void destroy_kernel(void *kernel_ptr) noexcept override;
    void execute(
        CommandList &cmdlist,
        void *kernel_ptr,
        luisa::span<Argument::Buffer const> arguments,
        luisa::vector<BufferCreationInfo> &outputs) noexcept override;
};

}// namespace luisa::compute