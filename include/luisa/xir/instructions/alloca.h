#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

class LC_XIR_API AllocaInst final : public InstructionOpMixin<AllocaOp, DerivedInstruction<AllocaInst, DerivedInstructionTag::ALLOCA>> {
public:
    AllocaInst(BasicBlock *parent_block, const Type *type, AllocaOp op) noexcept;
    [[nodiscard]] auto is_local() const noexcept { return op() == AllocaOp::LOCAL; }
    [[nodiscard]] auto is_shared() const noexcept { return op() == AllocaOp::SHARED; }
    [[nodiscard]] bool is_lvalue() const noexcept override { return true; }
    [[nodiscard]] AllocaInst *clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept override;
};

}// namespace luisa::compute::xir
