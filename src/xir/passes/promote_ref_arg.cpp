#include <luisa/core/logging.h>
#include <luisa/ast/type.h>
#include <luisa/xir/module.h>
#include <luisa/xir/function.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/passes/call_graph.h>
#include <luisa/xir/passes/promote_ref_arg.h>

namespace luisa::compute::xir {

namespace detail {

// checks if a function is a promotable callable, i.e., it is a callable function and all of its uses are call instructions
[[nodiscard]] static auto is_promotable_callable(FunctionDefinition *f) noexcept {
    // we may not process non-callable functions as their signatures might be imported/exported
    if (!f->isa<CallableFunction>()) { return false; }
    // otherwise, we check if all users of the callable are CallInst (non-call instructions
    // such as RayQueryPipelineInst may not allow changes to callee functions' signatures)
    for (auto &&use : f->use_list()) {
        if (auto user = use.user(); user != nullptr && !user->isa<CallInst>()) {
            return false;
        }
    }
    return true;
}

static void traverse_call_graph_post_order(Function *f, const CallGraph &call_graph,
                                           luisa::unordered_set<Function *> &visited,
                                           luisa::vector<CallableFunction *> &post_order) noexcept {
    if (visited.emplace(f).second) {
        if (auto def = f->definition()) {
            auto edges = call_graph.call_edges(def);
            for (auto &&call : edges) {
                traverse_call_graph_post_order(call->callee(), call_graph, visited, post_order);
            }
            if (is_promotable_callable(def)) {
                post_order.emplace_back(static_cast<CallableFunction *>(def));
            }
        }
    }
}

[[nodiscard]] static bool is_pointer_readonly(Value *p) noexcept {
    for (auto &&use : p->use_list()) {
        if (auto user = use.user()) {
            LUISA_DEBUG_ASSERT(user->isa<Instruction>(), "Invalid user.");
            switch (static_cast<Instruction *>(user)->derived_instruction_tag()) {
                case DerivedInstructionTag::LOAD: /* fine to check the next user */ break;
                case DerivedInstructionTag::RAY_QUERY_OBJECT_READ: /* fine to check the next user */ break;
                case DerivedInstructionTag::GEP: {
                    auto gep = static_cast<GEPInst *>(user);
                    LUISA_DEBUG_ASSERT(gep->base() == p, "Invalid GEP base.");
                    // if the pointer is used as the GEP base, we need to recursively check if
                    // the resulting element pointer is readonly
                    if (!is_pointer_readonly(gep)) { return false; }
                    break;
                }
                default: {
                    // be conservative and assume that the pointer is not readonly for other instructions
                    return false;
                }
            }
        }
    }
    // all uses of the pointer are either read-only
    return true;
}

[[nodiscard]] static ValueArgument *promote_ref_arg(CallableFunction *f, size_t index) noexcept {
    LUISA_DEBUG_ASSERT(index < f->arguments().size() && f->arguments()[index]->is_reference());
    auto arg = static_cast<ReferenceArgument *>(f->arguments()[index]);
    // we need to create a new value argument with the same type as the reference argument
    auto new_arg = f->create_value_argument(arg->type(), false);
    new_arg->add_comment("promoted reference argument");
    // we need to create a local variable to hold the value of the reference argument
    {
        XIRBuilder b;
        b.set_insertion_point(f->body_block()->instructions().head_sentinel());
        auto local = b.alloca_local(arg->type());
        b.store(local, new_arg);
        arg->replace_all_uses_with(local);
    }
    // we need to replace the reference argument with the new value argument
    f->replace_argument(index, new_arg);
    return new_arg;
}

static void promote_ref_args_in_function(CallableFunction *f, PromoteRefArgInfo &info) {
    luisa::fixed_vector<size_t, 16u> promoted_arg_indices;
    for (auto i = 0u; i < f->arguments().size(); i++) {
        // we may only promote non-custom-type reference arguments that are read-only
        if (auto arg = f->arguments()[i];
            arg->is_reference() && !arg->type()->is_custom() && is_pointer_readonly(arg)) {
            auto new_arg = promote_ref_arg(f, i);
            promoted_arg_indices.emplace_back(i);
            info.promoted_ref_args.emplace(static_cast<ReferenceArgument *>(arg), new_arg);
        }
    }
    // replace the call site arguments with load instructions
    for (auto &&use : f->use_list()) {
        if (auto user = use.user(); user != nullptr && user->isa<CallInst>()) {
            auto call = static_cast<CallInst *>(user);
            XIRBuilder b;
            b.set_insertion_point(call->prev());
            for (auto i : promoted_arg_indices) {
                auto loaded = b.load(f->arguments()[i]->type(), call->argument(i));
                call->set_argument(i, loaded);
            }
        }
    }
}

static void promote_ref_args_in_module(Module *m, PromoteRefArgInfo &info) noexcept {
    // we compute the call graph and collect the functions in post-order (i.e., leaves first)
    // so that we can process the callees before the callers to avoid reprocessing the same
    // function multiple times
    luisa::vector<CallableFunction *> post_order;
    {
        auto call_graph = compute_call_graph(m);
        luisa::unordered_set<Function *> visited;
        for (auto &&f : call_graph.root_functions()) {
            traverse_call_graph_post_order(f, call_graph, visited, post_order);
        }
    }
    // now we can iteratively promote the reference arguments of the functions in post-order
    for (auto f : post_order) {
        promote_ref_args_in_function(f, info);
    }
}

}// namespace detail

PromoteRefArgInfo promote_ref_arg_pass_run_on_module(Module *module) noexcept {
    PromoteRefArgInfo info;
    detail::promote_ref_args_in_module(module, info);
    return info;
}

}// namespace luisa::compute::xir
