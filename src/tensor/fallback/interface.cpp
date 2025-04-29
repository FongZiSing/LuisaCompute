#include <luisa/tensor/fallback/interface.h>
#include <luisa/tensor/tensor_builder.h>
#include <luisa/core/first_fit.h>

#include <luisa/core/stl/unordered_map.h>
#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/tensor/fallback/matmul_impl.h>
#include <luisa/tensor/fallback/set_value_impl.h>

#include <luisa/core/logging.h>

namespace luisa::compute {

FallbackTensorKernel::FallbackTensorKernel(DeviceInterface *device, ShaderManager *shader_manager, luisa::unique_ptr<TensorBuilder> &&t_args)
    : device(device), tensor_builder(std::move(t_args)), temp_alloc(65536, &temp_buffer_visitor, 2) {
    expr_topo.init(tensor_builder->root_expr().expressions, tensor_builder->allocated_tensor().size());
    auto sorted_tensors = expr_topo.topo_sort();
    executors.reserve(sorted_tensors.size());
    FirstFit first_fit(1ull << 48ull, 16ull);
    tensor_sub_nodes.resize(tensor_builder->allocated_tensor().size());
    luisa::vector<std::pair<FirstFit::Node *, uint64_t>> allocated_nodes(tensor_sub_nodes.size());
    luisa::vector<FirstFit::Node *> remove_nodes;
    for (auto &i : tensor_builder->arguments()) {
        allocated_nodes[i->idx()].second = std::numeric_limits<uint64_t>::max();
    }
    for (auto &i : tensor_builder->outputs()) {
        allocated_nodes[i->idx()].second = std::numeric_limits<uint64_t>::max();
    }
    // Accumulate ref-count
    uint64_t buffer_size_bytes = 0;
    for (auto &i : sorted_tensors) {
        auto func = [&](TensorData *tensor_data, Usage usage) {
            auto &node = allocated_nodes[tensor_data->idx()];
            // This is an input tensor, no need allocate
            if (node.second == std::numeric_limits<uint64_t>::max()) {
                return;
            }
            node.second++;
        };
        i->get_tensors(func);
    }
    // Make allocate
    for (auto &i : sorted_tensors) {
        auto func = [&](TensorData *tensor_data, Usage usage) {
            auto &node = allocated_nodes[tensor_data->idx()];
            // This is an input tensor, no need allocate
            if (node.second == std::numeric_limits<uint64_t>::max()) {
                return;
            }
            if (!node.first) {
                node.first = first_fit.allocate_best_fit(tensor_data->size_bytes());
                if (!node.first) [[unlikely]] {
                    LUISA_ERROR("Tensor graph too large, allocate failed.");
                }
                buffer_size_bytes = std::max(buffer_size_bytes, node.first->offset() + node.first->size());
                tensor_sub_nodes[tensor_data->idx()] = node.first->offset();
            }
            node.second--;
            if (node.second == 0) {
                remove_nodes.emplace_back(node.first);
            }
        };
        i->get_tensors(func);
        for (auto &i : remove_nodes) {
            first_fit.free(i);
        }
        remove_nodes.clear();
        if (buffer_size_bytes > 0)
            tensor_buffer = device->create_buffer(Type::of<float4>(), std::max<uint64_t>(1, buffer_size_bytes / sizeof(float4)), nullptr);
        else
            tensor_buffer.invalidate();
        // Make executors
        switch (i->tag()) {
            case TensorExpr::Tag::EGEMMExpr: {
                executors.emplace_back(luisa::make_unique<MatMulImpl>(device, shader_manager, static_cast<GEMMExpr *>(i)));
            } break;
            case TensorExpr::Tag::ESetValueExpr: {
                executors.emplace_back(luisa::make_unique<SetValueImpl>(device, shader_manager, static_cast<SetValueExpr *>(i)));
            } break;
            default: {
                LUISA_ERROR("Expr not supported.");
            } break;
        }
    }
    auto output_size = tensor_builder->outputs().size();
    outputs.reserve(output_size);
    for (uint64_t idx = 0; idx < output_size; ++idx) {
        auto tensor = tensor_builder->outputs()[idx];
        auto &buffer = outputs.emplace_back(device->create_buffer(Type::of<float4>(), std::max<uint64_t>(tensor->size_bytes() / sizeof(float4), 1ull), nullptr));
    }
}
FallbackTensorKernel::~FallbackTensorKernel() {
    if (tensor_buffer.handle != invalid_resource_handle)
        device->destroy_buffer(tensor_buffer.handle);
    for (auto &i : outputs) {
        device->destroy_buffer(i.handle);
    }
}
void FallbackTensorKernel::check(luisa::span<Argument::Buffer const> tensors) const {
    auto args = tensor_builder->arguments();
    if (tensors.size() != args.size()) {
        LUISA_ERROR("Argument size dismatch.");
    }
    for (uint64_t i = 0; i < args.size(); ++i) {
        if (args[i]->size_bytes() != tensors[i].size) {
            LUISA_ERROR("Argument {} size {} dismatch with dest {}.", i, tensors[i].size, args[i]->size_bytes());
        }
        uint64_t align = tensor_element_align(args[i]->element_type());
        if ((tensors[i].offset & (align - 1)) != 0) {
            LUISA_ERROR("Argument {} with offset {} should be align to {}", i, tensors[i].offset, align);
        }
    }
}
void *FallbackTensorInterface::compile_kernel(luisa::unique_ptr<TensorBuilder> &&tensor_builder) noexcept {
    return luisa::new_with_allocator<FallbackTensorKernel>(device(), &shader_manager, std::move(tensor_builder));
}
void FallbackTensorInterface::destroy_kernel(void *kernel_ptr) noexcept {
    luisa::delete_with_allocator(static_cast<FallbackTensorKernel *>(kernel_ptr));
}
Argument::Buffer FallbackTensorCallback::allocate_temp_buffer(size_t size_bytes) {
    size_bytes = (size_bytes + 15ull) & (~(15ull));
    auto chunk = kernel->temp_alloc.allocate(size_bytes);
    return Argument::Buffer{
        chunk.handle,
        chunk.offset,
        size_bytes};
}

Argument::Buffer FallbackTensorCallback::get_tensor_buffer(TensorData *data) {
    auto iter = args->find(data);
    // is arg
    if (iter != args->end()) {
        return iter->second;
    }
    return Argument::Buffer{
        .handle = kernel->tensor_buffer.handle,
        .offset = kernel->tensor_sub_nodes[data->idx()],
        .size = data->size_bytes()};
}
void FallbackTensorInterface::execute(
    CommandList &cmdlist,
    void *kernel_ptr,
    luisa::span<Argument::Buffer const> arguments,
    luisa::vector<BufferCreationInfo> &outputs) noexcept {
    auto kernel = static_cast<FallbackTensorKernel *>(kernel_ptr);
    kernel->device = device();
    kernel->check(arguments);
    luisa::unordered_map<TensorData *, Argument::Buffer> arg_map;
    arg_map.reserve(arguments.size() + outputs.size() + kernel->tensor_builder->captured_arguments().size());
    LUISA_ASSERT(arguments.size() == kernel->tensor_builder->arguments().size(), "Argument size mismatch.");
    outputs.clear();
    luisa::vector_resize(outputs, kernel->outputs.size());
    std::memcpy(outputs.data(), kernel->outputs.data(), outputs.size_bytes());
    for (uint64_t idx = 0; idx < arguments.size(); ++idx) {
        arg_map.try_emplace(kernel->tensor_builder->arguments()[idx], arguments[idx]);
    }
    for (uint64_t idx = 0; idx < outputs.size(); ++idx) {
        auto tensor = kernel->tensor_builder->outputs()[idx];
        auto buffer = kernel->outputs[idx];
        arg_map.try_emplace(
            tensor,
            Argument::Buffer{
                .handle = buffer.handle,
                .offset = 0,
                .size = buffer.total_size_bytes});
    }
    for (auto &i : kernel->tensor_builder->captured_arguments()) {
        arg_map.try_emplace(i.first, i.second);
    }

    FallbackTensorCallback callback;
    callback.args = &arg_map;
    callback.kernel = kernel;
    callback.shader_manager = &shader_manager;
    for (auto &i : kernel->executors) {
        i->execute(&callback, cmdlist);
    }
    kernel->temp_buffer_visitor.dispose(cmdlist);
}

LC_TENSOR_API luisa::unique_ptr<TensorInterface> TensorInterface::create_fallback_backend(Device &device) {
    return luisa::make_unique<FallbackTensorInterface>(device);
}
}// namespace luisa::compute