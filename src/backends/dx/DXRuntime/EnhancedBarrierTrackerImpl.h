#pragma once
#include "EnhancedBarrierTracker.h"

namespace lc::dx {
class EnhancedBarrierTrackerImpl : public EnhancedBarrierTracker {
protected:

    ///////////////// State Tracker
    vstd::vector<D3D12_BUFFER_BARRIER> bufferBarriers;
    vstd::vector<D3D12_TEXTURE_BARRIER> texBarriers;
    ///////////////// Commands

    void UpdateResourceState(Resource const *resPtr, ResourceStates &state);
    void BarrierFilter(D3D12_BUFFER_BARRIER &barrier);
    void BarrierFilter(D3D12_TEXTURE_BARRIER &barrier);
public:
    void UpdateState(CommandBufferBuilder const &cmdBuffer) override;
    void RestoreState(CommandBufferBuilder const &cmdBuffer) override;
    EnhancedBarrierTrackerImpl();
    ~EnhancedBarrierTrackerImpl();
};
}// namespace lc::dx