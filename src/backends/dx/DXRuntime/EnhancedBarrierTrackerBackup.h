#pragma once
#include "EnhancedBarrierTracker.h"
namespace lc::dx {
class EnhancedBarrierTrackerBackup : public EnhancedBarrierTracker {
protected:
    vstd::vector<D3D12_RESOURCE_BARRIER> barriers;
    void UpdateResourceState(Resource const *resPtr, ResourceStates &state);
    D3D12_RESOURCE_STATES ToStates(
        D3D12_BARRIER_SYNC sync,
        D3D12_BARRIER_ACCESS access,
        D3D12_BARRIER_LAYOUT layout);

public:
    void UpdateState(CommandBufferBuilder const &cmdBuffer) override;
    void RestoreState(CommandBufferBuilder const &cmdBuffer) override;
    EnhancedBarrierTrackerBackup();
    ~EnhancedBarrierTrackerBackup();
};
}// namespace lc::dx