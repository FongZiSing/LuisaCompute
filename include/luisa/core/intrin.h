#pragma once

#if defined(__x86_64__) || defined(_M_X64)
#define LUISA_ARCH_X86_64
#elif defined(__aarch64__)
#define LUISA_ARCH_ARM64
#else
#error Unsupported architecture
#endif

#if defined(LUISA_ARCH_X86_64)

#include <immintrin.h>
#include <cstdint>
#include <cassert>

#define LUISA_INTRIN_PAUSE() _mm_pause()

#elif defined(LUISA_ARCH_ARM64)

#include <arm_neon.h>

#define LUISA_INTRIN_PAUSE() asm volatile("isb")

#else

#include <thread>
#define LUISA_INTRIN_PAUSE() std::this_thread::yield()

#endif

////////////// assume
#ifdef NDEBUG         // assume only enabled in non-debug mode.
#if defined(__clang__)// Clang
#define LUISA_ASSUME(x) (__builtin_assume(x))
#elif defined(_MSC_VER)// MSVC
#define LUISA_ASSUME(x) (__assume(x))
#else// GCC
#define LUISA_ASSUME(x)                        \
    do {                                       \
        if (!(x)) { __builtin_unreachable(); } \
    } while (false)
#endif
#else
#define LUISA_ASSUME(expression) assert(expression)
#endif

// debug trap
#if defined(_MSC_VER)
#define LUISA_DEBUG_TRAP() __debugbreak()
#else
#if defined(__has_builtin) && __has_builtin(__builtin_debugtrap)
#define LUISA_DEBUG_TRAP() __builtin_debugtrap()
#elif defined(__has_builtin) && __has_builtin(__builtin_break)
#define LUISA_DEBUG_TRAP() __builtin_break()
#else
#include <csignal>
#define LUISA_DEBUG_TRAP() std::raise(SIGTRAP)
#endif
#endif
