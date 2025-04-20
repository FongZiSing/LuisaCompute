#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::xir {

// This pass is used to analyze the lexical scope of each instruction and
// collect the instructions that might break the lexical scope. This pass
// is useful for source code generation. For example, if we have a loop:
//
// loop {
//   def x = 0;
// }
// use x;
//
// For SSA-base IR like XIR, the use of x is valid because the definition
// of x dominates the use of x. However, for source code generation, like
// C++, the use of x is invalid because the definition of x is out of scope.
//
// This pass will collect the instructions that might break the lexical. So
// in the above example, the source code generator can hoist the definition
// of x out of the loop to make the use of x valid.

class Instruction;
class Function;

struct LexScopeInfo {
    luisa::unordered_set<const Instruction *> lexical_scope_breakers;
    luisa::vector<const Instruction *> lexical_scope_breaks_ordered;
};

struct LexScopeAnalysisConfig {
    bool loop_body_is_nested{false};
};

[[nodiscard]] LC_XIR_API LexScopeInfo lex_scope_analysis_pass_run_on_function(
    const Function *function,
    const LexScopeAnalysisConfig &config) noexcept;

}// namespace luisa::compute::xir
