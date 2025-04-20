#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

class LC_XIR_API DebugBreakInst : public DerivedInstruction<DebugBreakInst, DerivedInstructionTag::DEBUG_BREAK> {

public:
    using Evaluate = const void * /* pointer to evaluated data */
        (*)(void * /* backend context */, size_t /* index */) noexcept;
    using Callback = void (*)(void * /* backend context */, Evaluate);

private:
    Callback _callback;
    luisa::vector<luisa::string> _identifiers;

public:
    explicit DebugBreakInst(BasicBlock *parent_block, Callback callback = nullptr,
                            luisa::span<Value *const> operands = {}) noexcept;
    void set_callback(Callback callback) noexcept { _callback = callback; }
    [[nodiscard]] auto callback() const noexcept { return _callback; }
    [[nodiscard]] DebugBreakInst *clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept override;
};

}// namespace luisa::compute::xir
