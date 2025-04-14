#include <SPIRV/SpvBuilder.h>
#include <SPIRV/disassemble.h>
#include <SPIRV/Logger.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/format.h>
#include <luisa/ast/type.h>
#include <luisa/ast/usage.h>
#include <luisa/core/logging.h>
#include <luisa/ast/type_registry.h>
using namespace luisa;
using namespace luisa::compute;
class LCSpvBuilder : public spv::Builder {
public:
    LCSpvBuilder(unsigned int spvVersion, unsigned int userNumber, spv::SpvBuildLogger *logger) : spv::Builder(spvVersion, userNumber, logger) {}
    spv::Id make_type(Type const *type, Usage usage = Usage::NONE) {
        using namespace spv;
        switch (type->tag()) {
            case Type::Tag::BOOL:
                return makeBoolType();
            case Type::Tag::INT8:
                return makeIntType(8);
            case Type::Tag::INT16:
                return makeIntType(16);
            case Type::Tag::INT32:
                return makeIntType(32);
            case Type::Tag::INT64:
                return makeIntType(64);
            case Type::Tag::UINT8:
                return makeUintType(8);
            case Type::Tag::UINT16:
                return makeUintType(16);
            case Type::Tag::UINT32:
                return makeUintType(32);
            case Type::Tag::UINT64:
                return makeUintType(64);
            case Type::Tag::FLOAT16:
                return makeFloatType(16);
            case Type::Tag::FLOAT32:
                return makeFloatType(32);
            case Type::Tag::FLOAT64:
                return makeFloatType(64);
            case Type::Tag::VECTOR:
                return makeVectorType(make_type(type->element()), type->dimension());
            case Type::Tag::MATRIX:
                // TODO: float3x3 may have problem
                return makeMatrixType(make_type(type->element()), type->dimension(), type->dimension());
            case Type::Tag::ARRAY:
                return makeArrayType(make_type(type->element()), makeUintConstant(type->size()), type->element()->size());
            case Type::Tag::STRUCTURE: {
                size_t offset = 0;
                std::vector<Id> members;
                luisa::vector<int> offsets;
                auto str_name = luisa::format("S{}", type->hash());
                for (auto &mem : type->members()) {
                    offset = (offset + mem->alignment() - 1) & (~(mem->alignment() - 1));
                    members.emplace_back(make_type(mem));
                    offsets.emplace_back(offset);
                    offset += mem->size();
                }
                auto id = makeStructType(members, str_name.c_str());
                for (size_t idx = 0; idx < offsets.size(); ++idx) {
                    addMemberDecoration(id, idx, DecorationOffset, offsets[idx]);
                }
                return id;
            }
            case Type::Tag::BUFFER: {
                auto runtime_arr_name = luisa::format("RA{}", type->hash());
                auto runtime_arr_id = makeRuntimeArray(make_type(type->element()));
                addDecoration(runtime_arr_id, DecorationArrayStride, type->element()->alignment());
                std::vector<Id> member;
                member.emplace_back(runtime_arr_id);
                auto runtime_arr_struct_id = makeStructType(member, runtime_arr_name.c_str());
                auto ptr_id = makePointer(StorageClassUniform, runtime_arr_struct_id);
                addMemberDecoration(runtime_arr_struct_id, 0, DecorationOffset, 0);
                addDecoration(runtime_arr_struct_id, DecorationBufferBlock);
                if ((luisa::to_underlying(usage) & luisa::to_underlying(Usage::WRITE)) == 0) {
                    addMemberDecoration(runtime_arr_struct_id, 0, DecorationNonWritable);
                }
                return ptr_id;
            }
            case Type::Tag::TEXTURE: {
                Id type_2d;
                // Read sample type
                if ((luisa::to_underlying(usage) & luisa::to_underlying(Usage::WRITE)) == 0) {
                    type_2d = makeImageType(make_type(type->element()), (Dim)(type->dimension() - 1), false, 0, false, 1, ImageFormatUnknown);
                }
                // Read write storage type
                else {
                    ImageFormat format;
                    auto ele = type->element();
                    while (!ele->is_scalar()) {
                        ele = ele->element();
                    }
                    switch (ele->tag()) {
                        case Type::Tag::INT8:
                            format = spv::ImageFormatRgba8i;
                            break;
                        case Type::Tag::INT16:
                            format = spv::ImageFormatRgba16i;
                            break;
                        case Type::Tag::INT32:
                            format = spv::ImageFormatRgba32i;
                            break;
                        case Type::Tag::UINT8:
                            format = spv::ImageFormatRgba8ui;
                            break;
                        case Type::Tag::UINT16:
                            format = spv::ImageFormatRgba16ui;
                            break;
                        case Type::Tag::UINT32:
                            format = spv::ImageFormatRgba32ui;
                            break;
                        case Type::Tag::FLOAT16:
                            format = spv::ImageFormatRgba16f;
                            break;
                        case Type::Tag::FLOAT32:
                            format = spv::ImageFormatRgba32f;
                            break;
                        default:
                            LUISA_ERROR("Bad format.");
                    }
                    type_2d = makeImageType(make_type(type->element()), (Dim)(type->dimension() - 1), false, 0, false, 2, format);
                }
                return makePointer(StorageClassUniformConstant, type_2d);
            }
            case Type::Tag::BINDLESS_ARRAY:
            case Type::Tag::ACCEL:
                //TODO: implemented define
        }
        return makeVoidType();
    }
};
struct TestStruct {
    float a;
    int2 b;
    std::array<float3, 4> arr;
    int3 c;
    float2x2 mat0;
    float3x3 mat1;
    float4x4 mat2;
};
int main() {
    using namespace spv;
    SpvBuildLogger logger{};
    LCSpvBuilder builder{Spv_1_6, 0, &logger};
    std::basic_stringbuf<char, std::char_traits<char>, luisa::allocator<char>> stringbuf;
    std::ostream ostrm{&stringbuf};
    builder.make_type(Type::of<TestStruct>());
    builder.make_type(Type::texture(Type::of<float>(), 2), Usage::READ_WRITE);
    builder.make_type(Type::texture(Type::of<float>(), 2), Usage::READ);
    builder.make_type(Type::buffer(Type::of<float>()), Usage::READ_WRITE);
    builder.make_type(Type::buffer(Type::of<float>()), Usage::READ);

    std::vector<unsigned int> result;
    builder.dump(result);
    Disassemble(ostrm, result);
    std::cout << stringbuf.str() << "\n";
    return 0;
}