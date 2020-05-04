#include <iostream>

#include <wincodecsdk.h>
#include <wrl.h>

#include <ScreenGrab.h>

#include <agz/utility/d3d11/ImGui/imgui.h>
#include <agz/utility/d3d11.h>

#include <agz/smaa/mlaa.h>

using namespace agz::d3d11;

const int CLIENT_WIDTH  = 640;
const int CLIENT_HEIGHT = 640;

void run()
{
    WindowDesc window_desc;
    window_desc.clientWidth  = CLIENT_WIDTH;
    window_desc.clientHeight = CLIENT_HEIGHT;
    window_desc.windowTitle  = L"MLAA Demo by AirGuanZ";
    window_desc.resizable    = false;

    Window window;
    window.Initialize(window_desc);

#if (_WIN32_WINNT >= 0x0A00 /*_WIN32_WINNT_WIN10*/)
    Microsoft::WRL::Wrappers::RoInitializeWrapper roInitialize(RO_INIT_MULTITHREADED);
    if(FAILED(roInitialize))
        throw std::runtime_error("failed to initialize components for WIC");
#else
    const HRESULT CoHR = CoInitializeEx(nullptr, COINITBASE_MULTITHREADED);
    if(FAILED(CoHR))
        throw std::runtime_error("failed to initialize components for WIC");
    AGZ_SCOPE_GUARD({ CoUninitialize(); });
#endif

    agz::smaa::MLAA mlaa(
        window.Device(), window.DeviceContext(),
        8, 0.1f, agz::smaa::MLAA::Mode::Lum, CLIENT_WIDTH, CLIENT_HEIGHT);

    auto img = Texture2DLoader().LoadFromFile("input.png");

    Immediate2D imm2d;
    imm2d.SetFramebufferSize(
        { window.GetClientSizeX(), window.GetClientSizeY() });

    RenderTexture edgeTarget;
    edgeTarget.Initialize(
        window.GetClientSizeX(), window.GetClientSizeY(),
        DXGI_FORMAT_R32G32B32A32_FLOAT,
        DXGI_FORMAT_R32G32B32A32_FLOAT,
        DXGI_FORMAT_R32G32B32A32_FLOAT);

    RenderTexture weightTarget;
    weightTarget.Initialize(
        window.GetClientSizeX(), window.GetClientSizeY(),
        DXGI_FORMAT_R32G32B32A32_FLOAT,
        DXGI_FORMAT_R32G32B32A32_FLOAT,
        DXGI_FORMAT_R32G32B32A32_FLOAT);

    DepthState disableDepth;
    disableDepth.Initialize(false);

    RasterizerState rasterizerState;
    rasterizerState.Initialize(D3D11_FILL_SOLID, D3D11_CULL_NONE, false);

    bool displayInputTexture = false;

    while(!window.GetCloseFlag())
    {
        window.DoEvents();
        window.ImGuiNewFrame();

        window.ClearDefaultDepthStencil();
        window.ClearDefaultRenderTarget(0, 1, 1, 0);

        bool takeScreenshot = false;

        if(ImGui::Begin("SMAA", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
        {
            if(ImGui::Button("take screenshot"))
                takeScreenshot = true;

            ImGui::Checkbox("display input texture", &displayInputTexture);
        }
        ImGui::End();

        if(displayInputTexture)
        {
            imm2d.DrawTexture({ -1, -1 }, { 1, 1 }, img.Get());
        }
        else
        {
            // compute edge texture

            window.DeviceContext()->OMSetRenderTargets(
                1, edgeTarget.GetRenderTargetView().GetAddressOf(), nullptr);
            disableDepth.Bind();
            rasterizerState.Bind();

            mlaa.DetectEdgeWithLum(img.Get());

            rasterizerState.Unbind();
            disableDepth.Unbind();
            window.UseDefaultRenderTargetAndDepthStencil();

            // compute weight texture

            window.DeviceContext()->OMSetRenderTargets(
                1, weightTarget.GetRenderTargetView().GetAddressOf(), nullptr);
            disableDepth.Bind();
            rasterizerState.Bind();

            mlaa.ComputeBlendingWeight(edgeTarget.GetShaderResourceView().Get());

            rasterizerState.Unbind();
            disableDepth.Unbind();
            window.UseDefaultRenderTargetAndDepthStencil();

            // display edge texture

            //mlaa.ComputeBlendingWeight(edgeTarget.GetShaderResourceView().Get());

            //imm2d.DrawTexture(
            //    { -1, -1 }, { 1, 1 }, weightTarget.GetShaderResourceView().Get());

            //imm2d.DrawTexture(
            //    { -1, -1 }, { 1, 1 }, mlaa.GetInnerAreaTexture());

            mlaa.PerformBlending(
                img.Get(), weightTarget.GetShaderResourceView().Get());
        }

        if(takeScreenshot)
        {
            using namespace DirectX;
            using namespace Microsoft::WRL;

            ComPtr<ID3D11Texture2D> backBuffer;
            HRESULT hr = window.SwapChain()->GetBuffer(
                0, __uuidof(ID3D11Texture2D),
                reinterpret_cast<LPVOID *>(backBuffer.GetAddressOf()));
            if(SUCCEEDED(hr))
            {
                hr = SaveWICTextureToFile(
                    window.DeviceContext(), backBuffer.Get(),
                    GUID_ContainerFormatPng, L"./screenshot.png");
                if(FAILED(hr))
                    std::cerr << "failed to save screenshot" << std::endl;
            }
            else
                std::cerr << "failed to get backbuffer" << std::endl;
        }

        window.ImGuiRender();
        window.SwapBuffers();
    }
}

int main()
{
    try
    {
        run();
    }
    catch(const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return -1;
    }
}
