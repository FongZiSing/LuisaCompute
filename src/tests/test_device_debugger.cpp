#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;

struct MyStruct {
    float2 a;
    uint2 b;
};

LUISA_STRUCT(MyStruct, a, b) {};

// custom trap function with parameters
void my_trap(auto p, auto s, auto v, auto coord) {
    // You may do something with printing here
    LUISA_INFO("my_trap: p = {}, s = {}, v = {}, coord = {}", p, s, v, coord);
    // Please place a break point here with your IDE
}

// custom trap function without parameters
void foo() {
    // Please place a break point here with your IDE
}

int main(int argc, char *argv[]) {

    log_level_verbose();

    Context context{argv[0]};
    Device device = context.create_device("fallback");

    Kernel2D kernel = [&]() noexcept {
        UInt2 coord = dispatch_id().xy();
        $if (coord.x == 1) {
            $debug_break(coord);// programmable break point with __debugbreak-like intrinsics
        };
        $if (coord.x == coord.y) {
            Float2 v = make_float2(coord) / make_float2(dispatch_size().xy());
            Var<MyStruct> s;
            s.a = v;
            s.b = coord;
            $debug_break_on(s, v, coord, my_trap(dispatch_id, s, v, coord));// break point with custom trap function (note dispatch_id is always captured for convenience)
            $outline {
                device_log("s = {} at {}", s, dispatch_id());
            };
            $debug_break_on(foo());// custom trap function but without parameters (useful for use with interactive debuggers so you can control the break point during debugging)
        };
    };
    auto shader = device.compile(kernel);
    Stream stream = device.create_stream();
    stream << shader().dispatch(128u, 128u)
           << synchronize();
}
