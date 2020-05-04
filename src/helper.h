#pragma once

#include <cassert>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <type_traits>

#include <d3dcompiler.h>

#include <agz/smaa/mlaa.h>

AGZ_SMAA_BEGIN

namespace detail
{

// ########## scope guard

class ScopeGuard
{
public:

    template<typename Func,
        typename = std::enable_if<!std::is_same<
        ScopeGuard, typename std::decay<Func>::type>::value>>
        explicit ScopeGuard(Func &&func)
        : func_(std::forward<Func>(func)) { }

    ScopeGuard(const ScopeGuard &) = delete;
    ScopeGuard &operator=(const ScopeGuard &) = delete;

    ~ScopeGuard() { func_(); }

private:

    std::function<void()> func_;
};

#define AGZ_SMAA_SCOPE_GUARD(X) \
    AGZ_SMAA_SCOPE_GUARD_IMPL0(X, __LINE__)
#define AGZ_SMAA_SCOPE_GUARD_IMPL0(X, LINE) \
    AGZ_SMAA_SCOPE_GUARD_IMPL1(X, LINE)
#define AGZ_SMAA_SCOPE_GUARD_IMPL1(X, LINE) \
    ::agz::smaa::ScopeGuard _autoSMAAScopeGuard##LINE([&] X)

// ########## compile shader

inline ComPtr<ID3D10Blob> CompileToByteCode(
    const char *source, const char *target,
    const D3D_SHADER_MACRO *macros = nullptr)
{
#ifdef _DEBUG
    constexpr UINT COMPILER_FLAGS = D3DCOMPILE_DEBUG |
                                    D3DCOMPILE_SKIP_OPTIMIZATION;
#else
    constexpr UINT COMPILER_FLAGS = 0;
#endif

    const size_t sourceLen = std::strlen(source);

    ComPtr<ID3D10Blob> ret, err;
    const HRESULT hr = D3DCompile(
        source, sourceLen,
        nullptr, macros, nullptr,
        "main", target,
        COMPILER_FLAGS, 0,
        ret.GetAddressOf(), err.GetAddressOf());

    if(FAILED(hr))
    {
        auto rawErrMsg = reinterpret_cast<const char *>(
            err->GetBufferPointer());
        throw std::runtime_error(rawErrMsg);
    }

    return ret;
}

// ########## create shader

inline ComPtr<ID3D11VertexShader> CreateVertexShader(
    ID3D11Device *device, void *byteCode, size_t len)
{
    ComPtr<ID3D11VertexShader> shader;
    const HRESULT hr = device->CreateVertexShader(
        byteCode, len, nullptr, shader.GetAddressOf());
    return FAILED(hr) ? nullptr : shader;
}

inline ComPtr<ID3D11PixelShader> CreatePixelShader(
    ID3D11Device *device, void *byteCode, size_t len)
{
    ComPtr<ID3D11PixelShader> shader;
    const HRESULT hr = device->CreatePixelShader(
        byteCode, len, nullptr, shader.GetAddressOf());
    return FAILED(hr) ? nullptr : shader;
}

} // namespace detail

AGZ_SMAA_END
