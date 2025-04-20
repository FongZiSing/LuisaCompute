#include <luisa/core/logging.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/debug_break.h>

namespace luisa::compute::xir {

DebugBreakInst::DebugBreakInst(BasicBlock *parent_block, Callback callback,
                               luisa::span<Value *const> operands) noexcept
    : DerivedInstruction{parent_block, nullptr}, _callback{callback} { set_operands(operands); }

DebugBreakInst *DebugBreakInst::clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept {
    luisa::fixed_vector<Value *, 16u> resolved_operands;
    resolved_operands.reserve(operand_count());
    for (auto op_use : operand_uses()) {
        resolved_operands.emplace_back(resolver.resolve(op_use->value()));
    }
    auto cloned = b.debug_break(_callback);
    cloned->set_operands(resolved_operands);
    return cloned;
}

}// namespace luisa::compute::xir
