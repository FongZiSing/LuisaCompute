#pragma once

#include <luisa/runtime/rhi/command.h>
#include <luisa/core/stl/functional.h>
#include <d3d12.h>
#include <dxgi1_2.h>
#include <luisa/backends/ext/registry.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/logging.h>

namespace lc::dx {
class LCCmdBuffer;
class LCPreProcessVisitor;
class LCCmdVisitor;
}// namespace lc::dx

namespace luisa::compute {

class DXCustomCmd : public CustomDispatchCommand {
    using ResourceHandle = luisa::variant<
        Argument::Buffer,
        Argument::Texture,
        Argument::BindlessArray>;
public:
    struct ResourceUsage {
        ResourceHandle resource;
        D3D12_RESOURCE_STATES required_state;
        template<typename Arg>
            requires(luisa::is_constructible_v<ResourceHandle, Arg &&>)
        ResourceUsage(
            Arg &&resource,
            D3D12_RESOURCE_STATES required_state)
            : resource{std::forward<Arg>(resource)},
              required_state{required_state} {}
    };
    struct EnhancedResourceUsage {
        ResourceHandle resource;
        D3D12_BARRIER_SYNC sync;
        D3D12_BARRIER_ACCESS access;
        D3D12_BARRIER_LAYOUT texture_layout;// only used for texture_layout
        template<typename Arg>
            requires(luisa::is_constructible_v<ResourceHandle, Arg &&>)
        EnhancedResourceUsage(
            Arg &&resource,
            D3D12_BARRIER_SYNC sync,
            D3D12_BARRIER_ACCESS access,
            D3D12_BARRIER_LAYOUT texture_layout = D3D12_BARRIER_LAYOUT_UNDEFINED)
            : resource{std::forward<Arg>(resource)},
              sync{sync},
              access{access},
              texture_layout{texture_layout} {
            LUISA_ASSERT((this->resource.index() == 1) == (texture_layout != D3D12_BARRIER_LAYOUT_UNDEFINED), "Buffer must not have valid layout and texture must have a defined layout");
        }
    };

private:
    friend class lc::dx::LCCmdBuffer;
    friend class lc::dx::LCPreProcessVisitor;
    friend class lc::dx::LCCmdVisitor;
    virtual void execute(
        IDXGIAdapter1 *adapter,
        IDXGIFactory2 *dxgi_factory,
        ID3D12Device *device,
        ID3D12GraphicsCommandList4 *command_list) const noexcept = 0;
    [[nodiscard]] virtual luisa::span<const ResourceUsage> get_resource_usages() const noexcept {
        return {};
    }
    [[nodiscard]] virtual luisa::span<const EnhancedResourceUsage> get_enhanced_resource_usages() const noexcept {
        return {};
    }

    [[nodiscard]] static auto resource_state_to_usage(D3D12_RESOURCE_STATES state) noexcept {
        constexpr auto read_state =
            D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER |
            D3D12_RESOURCE_STATE_INDEX_BUFFER |
            D3D12_RESOURCE_STATE_DEPTH_READ |
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE |
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
            D3D12_RESOURCE_STATE_STREAM_OUT |
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT |
            D3D12_RESOURCE_STATE_COPY_SOURCE |
            D3D12_RESOURCE_STATE_RESOLVE_SOURCE |
            D3D12_RESOURCE_STATE_SHADING_RATE_SOURCE |
            D3D12_RESOURCE_STATE_VIDEO_DECODE_READ |
            D3D12_RESOURCE_STATE_VIDEO_PROCESS_READ |
            D3D12_RESOURCE_STATE_VIDEO_ENCODE_READ |
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS |
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE;
        constexpr auto write_state =
            D3D12_RESOURCE_STATE_RENDER_TARGET |
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS |
            D3D12_RESOURCE_STATE_DEPTH_WRITE |
            D3D12_RESOURCE_STATE_STREAM_OUT |
            D3D12_RESOURCE_STATE_COPY_DEST |
            D3D12_RESOURCE_STATE_RESOLVE_DEST |
            D3D12_RESOURCE_STATE_VIDEO_DECODE_WRITE |
            D3D12_RESOURCE_STATE_VIDEO_PROCESS_WRITE |
            D3D12_RESOURCE_STATE_VIDEO_ENCODE_WRITE |
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS |
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE;
        if (state == 0) return Usage::READ_WRITE;
        Usage usage = Usage::NONE;
        if ((state & read_state) != 0) {
            usage = static_cast<Usage>(luisa::to_underlying(usage) | luisa::to_underlying(Usage::READ));
        }
        if ((state & write_state) != 0) {
            usage = static_cast<Usage>(luisa::to_underlying(usage) | luisa::to_underlying(Usage::WRITE));
        }
        return usage;
    }
    [[nodiscard]] static auto resource_state_to_usage(
        D3D12_BARRIER_ACCESS state) noexcept {
        constexpr auto read_state =
            D3D12_BARRIER_ACCESS_VERTEX_BUFFER |
            D3D12_BARRIER_ACCESS_CONSTANT_BUFFER |
            D3D12_BARRIER_ACCESS_INDEX_BUFFER |
            D3D12_BARRIER_ACCESS_DEPTH_STENCIL_READ |
            D3D12_BARRIER_ACCESS_SHADER_RESOURCE |
            D3D12_BARRIER_ACCESS_INDIRECT_ARGUMENT |
            D3D12_BARRIER_ACCESS_COPY_SOURCE |
            D3D12_BARRIER_ACCESS_RESOLVE_SOURCE |
            D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_READ |
            D3D12_BARRIER_ACCESS_SHADING_RATE_SOURCE |
            D3D12_BARRIER_ACCESS_VIDEO_DECODE_READ |
            D3D12_BARRIER_ACCESS_VIDEO_PROCESS_READ |
            D3D12_BARRIER_ACCESS_VIDEO_ENCODE_READ |
            D3D12_BARRIER_ACCESS_UNORDERED_ACCESS;

        constexpr auto write_state =
            D3D12_BARRIER_ACCESS_RENDER_TARGET |
            D3D12_BARRIER_ACCESS_UNORDERED_ACCESS |
            D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE |
            D3D12_BARRIER_ACCESS_STREAM_OUTPUT |
            D3D12_BARRIER_ACCESS_COPY_DEST |
            D3D12_BARRIER_ACCESS_RESOLVE_DEST |
            D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_WRITE |
            D3D12_BARRIER_ACCESS_VIDEO_DECODE_WRITE |
            D3D12_BARRIER_ACCESS_VIDEO_PROCESS_WRITE |
            D3D12_BARRIER_ACCESS_VIDEO_ENCODE_WRITE;
        if (state == 0) {
            return Usage::READ_WRITE;
        }
        Usage usage = Usage::NONE;
        if ((state & read_state) != 0) {
            usage = static_cast<Usage>(luisa::to_underlying(usage) | luisa::to_underlying(Usage::READ));
        }
        if ((state & write_state) != 0) {
            usage = static_cast<Usage>(luisa::to_underlying(usage) | luisa::to_underlying(Usage::WRITE));
        }
        return usage;
    }

public:
    void traverse_arguments(ArgumentVisitor &visitor) const noexcept override {
        auto usages = get_resource_usages();
        for (auto &&[handle, state] : usages) {
            luisa::visit([&](auto &&v) {
                visitor.visit(v, resource_state_to_usage(state));
            },
                         handle);
        }
        auto enhanced_usages = get_enhanced_resource_usages();
        for (auto &&[handle, sync, access, layout] : enhanced_usages) {
            luisa::visit([&](auto &&v) {
                visitor.visit(v, resource_state_to_usage(access));
            },
                         handle);
        }
    }
    DXCustomCmd() noexcept = default;
    virtual ~DXCustomCmd() noexcept override = default;
    [[nodiscard]] uint64_t uuid() const noexcept override {
        return luisa::to_underlying(CustomCommandUUID::CUSTOM_DISPATCH);
    }
};

}// namespace luisa::compute
