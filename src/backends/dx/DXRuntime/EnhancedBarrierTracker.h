#pragma once
#include <d3dx12.h>
#include <Resource/BufferView.h>
#include <Resource/Resource.h>
#include <Resource/SwapChain.h>
namespace lc::dx {
namespace detail {
static constexpr D3D12_BARRIER_ACCESS write_access = D3D12_BARRIER_ACCESS_RENDER_TARGET | D3D12_BARRIER_ACCESS_UNORDERED_ACCESS | D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE | D3D12_BARRIER_ACCESS_STREAM_OUTPUT | D3D12_BARRIER_ACCESS_COPY_DEST | D3D12_BARRIER_ACCESS_RESOLVE_DEST | D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_WRITE | D3D12_BARRIER_ACCESS_VIDEO_DECODE_WRITE | D3D12_BARRIER_ACCESS_VIDEO_PROCESS_WRITE | D3D12_BARRIER_ACCESS_VIDEO_ENCODE_WRITE;
}// namespace detail
class CommandBufferBuilder;
class TopAccel;
class EnhancedBarrierTracker : public vstd::IOperatorNewBase {
public:
    struct TexView {
        TextureBase const *tex;
        uint level;
    };
    using ResourceView = vstd::variant<
        BufferView,
        TexView,
        SwapChain const *>;
    enum class Usage : uint {
        ComputeRead,
        ComputeAccelRead,
        ComputeUAV,
        CopySource,
        CopyDest,
        BuildAccel,
        CopyAccelSrc,
        CopyAccelDst,
        DepthRead,
        DepthWrite,
        IndirectArgs,
        VertexRead,
        IndexRead,
        RenderTarget,
        AccelInstanceBuffer,
        RasterRead,
        RasterAccelRead,
        RasterUAV,
    };
    struct Range {
        uint64 min;
        uint64 max;
        Range() {
            min = std::numeric_limits<uint64>::min();
            max = std::numeric_limits<uint64>::max();
        }
        Range(uint64 value) {
            min = value;
            max = value + 1;
        }
        uint64 size() const { return max - min; }
        Range(uint64 min, uint64 size)
            : min(min), max(size + min) {}
        bool Collide(Range const &r) const {
            return min < r.max && r.min < max;
        }
        void Combine(Range const &r) {
            min = std::min(min, r.min);
            max = std::max(max, r.max);
        }
        bool operator==(Range const &r) const {
            return min == r.min && max == r.max;
        }
        bool operator!=(Range const &r) const { return !operator==(r); }
    };
    D3D12_COMMAND_LIST_TYPE listType;
protected:
    struct BufferRange {
        // Range range;
        D3D12_BARRIER_SYNC before_sync;
        D3D12_BARRIER_SYNC after_sync;
        D3D12_BARRIER_ACCESS before_access;
        D3D12_BARRIER_ACCESS after_access;
        // D3D12_BARRIER_ACCESS init_access;
        bool first_time{true};// used for backup
    };
    struct BufferAfterRange {
        // Range range;
        D3D12_BARRIER_SYNC sync;
        D3D12_BARRIER_ACCESS access;
    };
    struct TextureRange {
        bool level_inited{false};
        bool level_require_update;
        bool first_time{true};// used for backup
        D3D12_BARRIER_SYNC before_sync;
        D3D12_BARRIER_SYNC after_sync;
        D3D12_BARRIER_ACCESS before_access;
        D3D12_BARRIER_ACCESS after_access;
        D3D12_BARRIER_LAYOUT before_layout;
        D3D12_BARRIER_LAYOUT after_layout;
    };
    struct ResourceStates {
        vstd::variant<
            BufferRange,
            vstd::vector<TextureRange>>
            layer_states;

        enum class Type : uint8_t {
            Buffer,
            Texture
        };
        size_t size;
        bool require_update{false};

        ResourceStates(Type type, size_t size, Resource const *res);
    };
    vstd::HashMap<Resource const *, ResourceStates> frameStates;
    vstd::vector<std::pair<Resource const *, ResourceStates *>> current_update_states;
    vstd::HashMap<Resource const *, size_t /* size */> writeStateMap;
public:
    vstd::HashMap<Resource const *, size_t> &WriteStateMap() {
        return writeStateMap;
    }
    EnhancedBarrierTracker();
    virtual ~EnhancedBarrierTracker();
    void Record(
        ResourceView const &res,
        D3D12_BARRIER_SYNC sync,
        D3D12_BARRIER_ACCESS access,
        D3D12_BARRIER_LAYOUT layout);
    // fallback
    void Record(
        ResourceView const &res,
        D3D12_RESOURCE_STATES state);
    void Record(
        Resource const *res,
        Range range,
        D3D12_RESOURCE_STATES state);

    void Record(
        ResourceView const &res,
        Usage resUsage);
    void Record(
        Resource const *res,
        Range range,
        Usage resUsage);
    void Record(
        Resource const *res,
        Range range,
        D3D12_BARRIER_SYNC sync,
        D3D12_BARRIER_ACCESS access,
        D3D12_BARRIER_LAYOUT layout);

    virtual void UpdateState(CommandBufferBuilder const &cmdBuffer) = 0;
    virtual void RestoreState(CommandBufferBuilder const &cmdBuffer) = 0;
};
}// namespace lc::dx