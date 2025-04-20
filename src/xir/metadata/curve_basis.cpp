#include <luisa/xir/metadata/curve_basis.h>

namespace luisa::compute::xir {

CurveBasisMD::CurveBasisMD(Pool *pool, CurveBasisSet set) noexcept
    : Super{pool}, _curve_basis_set{set} {}

CurveBasisMD *CurveBasisMD::clone(Pool *pool) const noexcept {
    return pool->create<CurveBasisMD>(pool, this->curve_basis_set());
}

}// namespace luisa::compute::xir
