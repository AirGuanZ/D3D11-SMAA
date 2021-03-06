#include <iostream>

#include <wincodecsdk.h>
#include <wrl.h>

#include <ScreenGrab.h>

#include <agz/utility/d3d11/ImGui/imgui.h>
#include <agz/utility/d3d11/ImGui/imfilebrowser.h>
#include <agz/utility/d3d11.h>

#include <agz/smaa/mlaa.h>
#include <agz/smaa/smaa.h>

using namespace agz::d3d11;

void run()
{
    // initialize d3d11 window

    WindowDesc window_desc;
    window_desc.clientWidth  = 640;
    window_desc.clientHeight = 480;
    window_desc.windowTitle  = L"MLAA Demo by AirGuanZ";

    Window window;
    window.Initialize(window_desc);
    window.Maximize();

    // initialize WIC for screenshot saving

#if (_WIN32_WINNT >= 0x0A00 /*_WIN32_WINNT_WIN10*/)
    Microsoft::WRL::Wrappers::RoInitializeWrapper roInit(RO_INIT_MULTITHREADED);
    if(FAILED(roInit))
        throw std::runtime_error("failed to initialize components for WIC");
#else
    const HRESULT CoHR = CoInitializeEx(nullptr, COINITBASE_MULTITHREADED);
    if(FAILED(CoHR))
        throw std::runtime_error("failed to initialize components for WIC");
    AGZ_SCOPE_GUARD({ CoUninitialize(); });
#endif

    // for drawing img to screen

    Immediate2D imm2d;
    imm2d.SetFramebufferSize(
        { window.GetClientSizeX(), window.GetClientSizeY() });

    // intermediate rendertarget

    Vec2i targetSize = {
        window.GetClientSizeX(),
        window.GetClientSizeY()
    };

    auto edgeTarget = std::make_unique<RenderTexture >();
    edgeTarget->Initialize(
        window.GetClientSizeX(), window.GetClientSizeY(),
        DXGI_FORMAT_R32G32B32A32_FLOAT,
        DXGI_FORMAT_R32G32B32A32_FLOAT,
        DXGI_FORMAT_R32G32B32A32_FLOAT);

    auto weightTarget = std::make_unique<RenderTexture>();
    weightTarget->Initialize(
        window.GetClientSizeX(), window.GetClientSizeY(),
        DXGI_FORMAT_R32G32B32A32_FLOAT,
        DXGI_FORMAT_R32G32B32A32_FLOAT,
        DXGI_FORMAT_R32G32B32A32_FLOAT);

    auto outputTarget = std::make_unique<RenderTexture>();
    outputTarget->Initialize(
        window.GetClientSizeX(), window.GetClientSizeY(),
        DXGI_FORMAT_R32G32B32A32_FLOAT,
        DXGI_FORMAT_R32G32B32A32_FLOAT,
        DXGI_FORMAT_R32G32B32A32_FLOAT);

    auto useTargetViewport = [&]
    {
        D3D11_VIEWPORT vp;
        vp.TopLeftX = 0;
        vp.TopLeftY = 0;
        vp.Width    = static_cast<float>(targetSize.x);
        vp.Height   = static_cast<float>(targetSize.y);
        vp.MaxDepth = 1;
        vp.MinDepth = 0;
        window.setViewport(vp);
    };

    // mlaa & smaa

    float edgeThreshold           = 0.1f;
    float edgeLocalContrastFactor = 0.5f;
    float cornerAreaFactor        = 0.5f;
    int maxSearchDistanceLen      = 8;

    std::unique_ptr<agz::mlaa::MLAA> mlaa;
    std::unique_ptr<agz::smaa::SMAA> smaa;

    auto updateAA = [&]
    {
        mlaa = std::make_unique<agz::mlaa::MLAA>(
            window.Device(), window.DeviceContext(),
            targetSize.x, targetSize.y,
            agz::mlaa::EdgeDetectionMode::Lum,
            edgeThreshold, maxSearchDistanceLen);

        smaa = std::make_unique<agz::smaa::SMAA>(
            window.Device(), window.DeviceContext(),
            targetSize.x, targetSize.y,
            agz::smaa::EdgeDetectionMode::Lum,
            edgeThreshold, edgeLocalContrastFactor,
            maxSearchDistanceLen, cornerAreaFactor);
    };

    updateAA();

    // window resizing handler

    auto setTargetSize = [&](int w, int h)
    {
        targetSize = { w, h };

        updateAA();

        edgeTarget = std::make_unique<RenderTexture >();
        edgeTarget->Initialize(
            w, h,
            DXGI_FORMAT_R32G32B32A32_FLOAT,
            DXGI_FORMAT_R32G32B32A32_FLOAT,
            DXGI_FORMAT_R32G32B32A32_FLOAT);

        weightTarget = std::make_unique<RenderTexture>();
        weightTarget->Initialize(
            w, h,
            DXGI_FORMAT_R32G32B32A32_FLOAT,
            DXGI_FORMAT_R32G32B32A32_FLOAT,
            DXGI_FORMAT_R32G32B32A32_FLOAT);

        outputTarget = std::make_unique<RenderTexture>();
        outputTarget->Initialize(
            w, h,
            DXGI_FORMAT_R32G32B32A32_FLOAT,
            DXGI_FORMAT_R32G32B32A32_FLOAT,
            DXGI_FORMAT_R32G32B32A32_FLOAT);
    };

    FunctionalEventHandler<WindowResizeEvent> windowResizeHandler(
        [&](const WindowResizeEvent &e)
    {
        imm2d.SetFramebufferSize({ e.newClientWidth, e.newClientHeight });
    });
    window.Attach(&windowResizeHandler);

    // load image

    ComPtr<ID3D11ShaderResourceView> img;

    auto loadImage = [&](const std::string &filename)
    {
        auto imgData = agz::img::load_rgba_from_file(filename);
        if(!imgData.is_available())
            return;

        auto imgDataF = imgData.map([](const agz::math::color4b &c)
        {
            return agz::math::vec4f(
                c.r / 255.0f, c.g / 255.0f, c.b / 255.0f, c.a / 255.0f);
        });

        img = Texture2DLoader().LoadFromMemory(
            imgDataF.shape()[1], imgDataF.shape()[0], &imgDataF.raw_data()[0]);

        setTargetSize(imgDataF.shape()[1], imgDataF.shape()[0]);
    };

    loadImage("./1.png");

    ImGui::FileBrowser imgFileBrowser;

    // disable depth test

    DepthState disableDepth;
    disableDepth.Initialize(false);

    // disable backface culling

    RasterizerState rasterizerState;
    rasterizerState.Initialize(D3D11_FILL_SOLID, D3D11_CULL_NONE, false);

    // display mode

    enum DisplayMode : int
    {
        OriImage         = 0,
        AAImage          = 1,
        EdgeTexture      = 2,
        WeightTexture    = 3,
        InnerAreaTexture = 4,
    };

    bool useSMAA = false;

    int displayMode = AAImage;

    // mainloop

    while(!window.GetCloseFlag())
    {
        window.DoEvents();
        window.ImGuiNewFrame();

        window.ClearDefaultDepthStencil();
        window.ClearDefaultRenderTarget(0, 1, 1, 0);

        bool takeScreenshot = false;

        if(ImGui::Begin("MLAA"))
        {
            if(ImGui::Button("load image file"))
                imgFileBrowser.Open();

            ImGui::SameLine();

            if(ImGui::Button("take screenshot"))
                takeScreenshot = true;

            ImGui::Checkbox("use SMAA", &useSMAA);

            if(ImGui::InputInt(
                "search distance", &maxSearchDistanceLen))
            {
                maxSearchDistanceLen = (std::max)(0, maxSearchDistanceLen);
                updateAA();
            }

            if(useSMAA && ImGui::InputFloat(
                "local contrast factor", &edgeLocalContrastFactor))
            {
                edgeLocalContrastFactor = (std::max)(
                    0.0f, edgeLocalContrastFactor);
                updateAA();
            }

            if(useSMAA && ImGui::SliderFloat(
                "corner area factor", &cornerAreaFactor, 0, 1))
                updateAA();

            if(ImGui::SliderFloat("edge threshold", &edgeThreshold, 0, 1))
                updateAA();

            ImGui::RadioButton("Input  ", &displayMode, OriImage);
            ImGui::RadioButton("Output ", &displayMode, AAImage);
            ImGui::RadioButton("Edge   ", &displayMode, EdgeTexture);
            ImGui::RadioButton("Weight ", &displayMode, WeightTexture);
            ImGui::RadioButton("Area   ", &displayMode, InnerAreaTexture);
        }
        ImGui::End();

        imgFileBrowser.Display();

        if(imgFileBrowser.HasSelected())
        {
            auto selectedFilename = imgFileBrowser.GetSelected();
            imgFileBrowser.ClearSelected();
            loadImage(selectedFilename.string());
        }

        useTargetViewport();

        // compute edge texture

        const float TARGET_BKGD[4] = { 0, 0, 0, 0 };

        window.DeviceContext()->OMSetRenderTargets(
            1, edgeTarget->GetRenderTargetView().GetAddressOf(), nullptr);
        window.DeviceContext()->ClearRenderTargetView(
            edgeTarget->GetRenderTargetView().Get(), TARGET_BKGD);

        if(useSMAA)
            smaa->detectEdge(img.Get());
        else
            mlaa->detectEdge(img.Get());

        window.UseDefaultRenderTargetAndDepthStencil();

        // compute weight texture

        window.DeviceContext()->OMSetRenderTargets(
            1, weightTarget->GetRenderTargetView().GetAddressOf(), nullptr);
        window.DeviceContext()->ClearRenderTargetView(
            weightTarget->GetRenderTargetView().Get(), TARGET_BKGD);

        if(useSMAA)
            smaa->computeBlendingWeight(edgeTarget->GetShaderResourceView().Get());
        else
            mlaa->computeBlendingWeight(edgeTarget->GetShaderResourceView().Get());
        
        window.UseDefaultRenderTargetAndDepthStencil();

        // compute output image

        window.DeviceContext()->OMSetRenderTargets(
            1, outputTarget->GetRenderTargetView().GetAddressOf(), nullptr);
        window.DeviceContext()->ClearRenderTargetView(
            outputTarget->GetRenderTargetView().Get(), TARGET_BKGD);

        if(useSMAA)
            smaa->blend(weightTarget->GetShaderResourceView().Get(), img.Get());
        else
            mlaa->blend(weightTarget->GetShaderResourceView().Get(), img.Get());
        
        window.UseDefaultRenderTargetAndDepthStencil();

        // render to screen

        window.UseDefaultViewport();

        if(displayMode == OriImage)
        {
            imm2d.DrawTextureP2(
                {}, targetSize, img.Get());
        }
        else if(displayMode == AAImage)
        {
            imm2d.DrawTextureP2(
                {}, targetSize, outputTarget->GetShaderResourceView().Get());
        }
        else if(displayMode == EdgeTexture)
        {
            imm2d.DrawTextureP2(
                {}, targetSize, edgeTarget->GetShaderResourceView().Get());
        }
        else if(displayMode == WeightTexture)
        {
            imm2d.DrawTextureP2(
                {}, targetSize, weightTarget->GetShaderResourceView().Get());
        }
        else
        {
            const int size = targetSize.min_elem();
            imm2d.DrawTextureP2(
                { 0, 0 }, { size, size },
                mlaa->_blendingWeight().innerAreaTextureSRV.Get());
        }

        // screenshot

        if(takeScreenshot)
        {
            RenderTexture tempTarget;
            tempTarget.Initialize(
                targetSize.x, targetSize.y,
                DXGI_FORMAT_R8G8B8A8_UNORM,
                DXGI_FORMAT_R8G8B8A8_UNORM,
                DXGI_FORMAT_R8G8B8A8_UNORM);

            window.DeviceContext()->OMSetRenderTargets(
                1, tempTarget.GetRenderTargetView().GetAddressOf(), nullptr);

            useTargetViewport();
            
            switch(DisplayMode(displayMode))
            {
            case OriImage:
                imm2d.DrawTexture({ -1, -1 }, { 1, 1 }, img.Get());
                break;
            case AAImage:
                imm2d.DrawTexture(
                    { -1, -1 }, { 1, 1 },
                    outputTarget->GetShaderResourceView().Get());
                break;
            case EdgeTexture:
                imm2d.DrawTexture(
                    { -1, -1 }, { 1, 1 },
                    edgeTarget->GetShaderResourceView().Get());
                break;
            case WeightTexture:
                imm2d.DrawTexture(
                    { -1, -1 }, { 1, 1 },
                    weightTarget->GetShaderResourceView().Get());
                break;
            default:
                if(!useSMAA)
                {
                    imm2d.DrawTexture(
                        { -1, -1 }, { 1, 1 },
                        mlaa->_blendingWeight().innerAreaTextureSRV.Get());
                }
                break;
            }

            window.UseDefaultRenderTargetAndDepthStencil();

            const HRESULT hr = DirectX::SaveWICTextureToFile(
                window.DeviceContext(), tempTarget.GetRenderTarget().Get(),
                GUID_ContainerFormatPng, L"./screenshot.png",
                nullptr, nullptr, true);
            if(FAILED(hr))
                std::cerr << "failed to save screenshot" << std::endl;
        }

        window.UseDefaultViewport();
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
