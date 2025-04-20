#pragma once

#include <atomic>
#include <luisa/core/stl/memory.h>

namespace luisa {

template<typename T>
class ManagedPtr;

namespace detail {

struct ManagedPtrLowLevelOp;

class ManagedObject {

private:
    std::atomic<int32_t> _ref_count;
    uint32_t _managed_id;

public:
    explicit ManagedObject(uint32_t m_id = 0u) noexcept : _ref_count{1}, _managed_id{m_id} {}
    virtual ~ManagedObject() = default;
    ManagedObject(ManagedObject &&) = delete;
    ManagedObject(const ManagedObject &) = delete;
    ManagedObject &operator=(ManagedObject &&) = delete;
    ManagedObject &operator=(const ManagedObject &) = delete;

private:
    friend ManagedPtrLowLevelOp;
    ManagedObject *do_retain() noexcept {
        [[maybe_unused]] auto old_refcount = _ref_count.fetch_add(1, std::memory_order_relaxed);
        assert(old_refcount > 0 && "Retained object is likely already destroyed.");
        return this;
    }
    void do_release() noexcept {
        auto old_refcount = _ref_count.fetch_sub(1, std::memory_order_acq_rel);
        assert(old_refcount > 0 && "Releasing object is likely already destroyed.");
        if (old_refcount == 1) { luisa::delete_with_allocator(this); }
    }

public:
    [[nodiscard]] auto managed_id() const noexcept { return _managed_id; }
    void set_managed_id(uint32_t m_id) noexcept { _managed_id = m_id; }
};

struct ManagedPtrLowLevelOp {
    template<typename T>
        requires std::derived_from<T, ManagedObject>
    [[nodiscard]] static auto retain_nonnull(T *o) noexcept {
        assert(o != nullptr && "Null pointer dereferenced.");
        return static_cast<T *>(o->do_retain());
    }
    template<typename T>
        requires std::derived_from<T, ManagedObject>
    [[nodiscard]] static auto retain(T *o) noexcept {
        return o == nullptr ? nullptr : retain_nonnull(o);
    }
    template<typename T>
        requires std::derived_from<T, ManagedObject>
    static void release(T *o) noexcept {
        if (o != nullptr) { o->do_release(); }
    }
    template<typename T>
        requires std::derived_from<T, ManagedObject>
    static void reset(ManagedPtr<T> &m, T *ptr) noexcept {
        m.reset(ptr);
    }
    template<typename T>
        requires std::derived_from<T, ManagedObject>
    [[nodiscard]] static auto transfer(ManagedPtr<T> &m) noexcept {
        return m.transfer();
    }
};

}// namespace detail

template<typename T>
class ManagedPtr : public ManagedPtr<const T> {
public:
    ManagedPtr() noexcept = default;
    ~ManagedPtr() noexcept = default;

    ManagedPtr(const ManagedPtr &) noexcept = default;
    ManagedPtr(ManagedPtr &&) noexcept = default;
    ManagedPtr &operator=(ManagedPtr &&) noexcept = default;
    ManagedPtr &operator=(const ManagedPtr &) noexcept = default;

    ManagedPtr(const ManagedPtr<const T> &) noexcept = delete;
    ManagedPtr(ManagedPtr<const T> &&) noexcept = delete;
    ManagedPtr &operator=(ManagedPtr<const T> &&) noexcept = delete;
    ManagedPtr &operator=(const ManagedPtr<const T> &) noexcept = delete;

    [[nodiscard]] T *get() const noexcept { return const_cast<T *>(ManagedPtr<const T>::get()); }
    [[nodiscard]] T *operator->() const noexcept { return get(); }
    [[nodiscard]] T &operator*() const noexcept { return *get(); }
};

template<typename T>
class ManagedPtr<const T> {

private:
    static_assert(std::derived_from<T, detail::ManagedObject>);
    T *_object{nullptr};

private:
    friend detail::ManagedPtrLowLevelOp;
    [[nodiscard]] T *transfer() noexcept {
        return std::exchange(_object, nullptr);
    }
    void reset(T *new_object = nullptr) noexcept {
        luisa::detail::ManagedPtrLowLevelOp::release(
            std::exchange(_object, new_object));
    }

public:
    ManagedPtr() noexcept = default;
    ~ManagedPtr() noexcept { reset(); }
    ManagedPtr(ManagedPtr &&other) noexcept {
        reset(other.transfer());
    }
    ManagedPtr(const ManagedPtr &other) noexcept {
        reset(luisa::detail::ManagedPtrLowLevelOp::
                  retain(const_cast<T *>(other.get())));
    }
    ManagedPtr &operator=(ManagedPtr &&other) noexcept {
        if (&other != this) {
            reset(other.transfer());
        }
        return *this;
    }
    ManagedPtr &operator=(const ManagedPtr &other) noexcept {
        if (&other != this) {
            if (auto p = const_cast<T *>(other.get()); p != this->get()) {
                reset(luisa::detail::ManagedPtrLowLevelOp::retain(p));
            }
        }
        return *this;
    }

public:
    [[nodiscard]] const T *get() const noexcept { return _object; }
    [[nodiscard]] const T *operator->() const noexcept { return get(); }
    [[nodiscard]] const T &operator*() const noexcept { return *get(); }
    [[nodiscard]] explicit operator bool() const noexcept { return _object != nullptr; }
    [[nodiscard]] bool operator==(const T *rhs) const noexcept { return get() == rhs; }
    [[nodiscard]] bool operator==(const ManagedPtr &rhs) const noexcept { return get() == rhs.get(); }
};

template<typename T, typename Base = detail::ManagedObject>
    requires std::derived_from<Base, detail::ManagedObject>
class Managed : public Base {

public:
    static_assert(std::same_as<std::remove_cv_t<T>, T>);
    using derived_type = T;
    using base_type = Base;
    using super_type = Managed;
    using base_type::base_type;

public:
    [[nodiscard]] auto lock() noexcept {
        auto self = detail::ManagedPtrLowLevelOp::
            retain_nonnull(static_cast<T *>(this));
        ManagedPtr<T> p;
        detail::ManagedPtrLowLevelOp::reset(p, self);
        return p;
    }
    [[nodiscard]] auto lock() const noexcept {
        return static_cast<ManagedPtr<const T>>(
            const_cast<Managed *>(this)->lock());
    }
};

template<typename T, typename... Args>
    requires std::derived_from<T, detail::ManagedObject>
[[nodiscard]] ManagedPtr<T> make_managed(Args &&...args) noexcept {
    auto o = luisa::new_with_allocator<std::remove_const_t<T>>(std::forward<Args>(args)...);
    assert(std::addressof(*o) == std::addressof(*static_cast<detail::ManagedObject *>(o)) &&
           "ManagedObject should be the first non-empty base class of its derived classes.");
    ManagedPtr<T> p;
    detail::ManagedPtrLowLevelOp::reset(p, o);
    return p;
}

}// namespace luisa
