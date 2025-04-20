#pragma once

#include <luisa/runtime/rhi/curve_basis.h>
#include <luisa/xir/metadata.h>

namespace luisa::compute::xir {

class LC_XIR_API CurveBasisMD final : public DerivedMetadata<CurveBasisMD, DerivedMetadataTag::CURVE_BASIS> {

private:
    CurveBasisSet _curve_basis_set;

public:
    explicit CurveBasisMD(Pool *pool, CurveBasisSet set = {}) noexcept;
    [[nodiscard]] auto &curve_basis_set() noexcept { return _curve_basis_set; }
    [[nodiscard]] auto &curve_basis_set() const noexcept { return _curve_basis_set; }
    [[nodiscard]] CurveBasisMD *clone(Pool *pool) const noexcept override;
};

}// namespace luisa::compute::xir
