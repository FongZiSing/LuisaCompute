//
// Created by swfly on 2024/11/21.
//

#include <luisa/core/logging.h>
#include "fallback_buffer.h"

namespace luisa::compute::fallback {

FallbackBufferView FallbackBuffer::view(size_t offset, size_t size) noexcept {
    LUISA_DEBUG_ASSERT(offset + size <= _size, "Buffer view out of range.");
    return {_data + offset, size};
}

FallbackBufferView FallbackBuffer::view_with_offset(size_t offset) noexcept {
    LUISA_DEBUG_ASSERT(offset <= _size, "Buffer view out of range.");
    return {_data + offset, _size - offset};
}

FallbackBuffer::FallbackBuffer(size_t size_bytes) noexcept
    : _size{size_bytes}, _is_external{false} {
    _data = luisa::allocate_with_allocator<std::byte>(_size);
}

FallbackBuffer::FallbackBuffer(std::byte *data, size_t size) noexcept
    : _data{data}, _size{size}, _is_external{true} {}

FallbackBuffer::~FallbackBuffer() noexcept {
    if (!_is_external) {
        luisa::deallocate_with_allocator(_data);
    }
}

}// namespace luisa::compute::fallback
