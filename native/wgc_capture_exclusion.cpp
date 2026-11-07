#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <windows.ui.interop.h>
#include <inspectable.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Windows.UI.h>
#include <winrt/base.h>

#include <cstdint>
#include <vector>

MIDL_INTERFACE("BB91F61B-218A-587D-8580-2701A74C0525")
ID2SDisplayGraphicsCaptureSession : public IInspectable {
public:
    virtual HRESULT STDMETHODCALLTYPE SetWindowExclusionList(
        IUnknown* windows,
        UINT64* configuration_iteration
    ) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetWindowExclusionList(
        IUnknown** windows
    ) = 0;
};

MIDL_INTERFACE("71616DC8-FEA5-5741-A3D8-591ACC39A9EE")
ID2SDirect3D11CaptureFrame3 : public IInspectable {
public:
    virtual HRESULT STDMETHODCALLTYPE ConfigurationIteration(
        UINT64* configuration_iteration
    ) = 0;
};

using GetWindowIdFromWindowFn = HRESULT(WINAPI*)(
    HWND,
    ABI::Windows::UI::WindowId*
);

static HRESULT get_window_id_from_hwnd(
    UINT64 hwnd_value,
    winrt::Windows::UI::WindowId* window_id
) {
    if (!window_id) {
        return E_INVALIDARG;
    }
    static HMODULE windows_ui_module = LoadLibraryExW(
        L"Windows.UI.dll",
        nullptr,
        LOAD_LIBRARY_SEARCH_SYSTEM32
    );
    if (!windows_ui_module) {
        return HRESULT_FROM_WIN32(GetLastError());
    }
    static auto get_window_id_from_window = reinterpret_cast<GetWindowIdFromWindowFn>(
        GetProcAddress(windows_ui_module, "GetWindowIdFromWindow")
    );
    if (!get_window_id_from_window) {
        const DWORD error = GetLastError();
        return HRESULT_FROM_WIN32(error ? error : ERROR_PROC_NOT_FOUND);
    }

    ABI::Windows::UI::WindowId abi_window_id{};
    const HRESULT result = get_window_id_from_window(
        reinterpret_cast<HWND>(static_cast<std::uintptr_t>(hwnd_value)),
        &abi_window_id
    );
    if (SUCCEEDED(result)) {
        window_id->Value = abi_window_id.Value;
    }
    return result;
}

extern "C" __declspec(dllexport) HRESULT d2s_wgc_set_window_exclusion_list(
    void* session,
    const UINT64* hwnds,
    UINT32 count,
    UINT64* configuration_iteration
) {
    if (!session || (!hwnds && count != 0) || !configuration_iteration) {
        return E_INVALIDARG;
    }

    auto* unknown = static_cast<IUnknown*>(session);
    ID2SDisplayGraphicsCaptureSession* display_session = nullptr;
    HRESULT result = unknown->QueryInterface(
        __uuidof(ID2SDisplayGraphicsCaptureSession),
        reinterpret_cast<void**>(&display_session)
    );
    if (FAILED(result)) {
        return result;
    }

    try {
        auto values = winrt::single_threaded_vector<winrt::Windows::UI::WindowId>();
        std::vector<winrt::Windows::UI::WindowId> expected;
        expected.reserve(count);
        for (UINT32 index = 0; index < count; ++index) {
            winrt::Windows::UI::WindowId window_id{};
            result = get_window_id_from_hwnd(hwnds[index], &window_id);
            if (FAILED(result)) {
                display_session->Release();
                return result;
            }

            expected.push_back(window_id);
            values.Append(window_id);
        }
        auto iterable = values.as<
            winrt::Windows::Foundation::Collections::IIterable<winrt::Windows::UI::WindowId>
        >();
        result = display_session->SetWindowExclusionList(
            reinterpret_cast<IUnknown*>(winrt::get_abi(iterable)),
            configuration_iteration
        );
        if (FAILED(result)) {
            display_session->Release();
            return result;
        }

        IUnknown* windows_unknown = nullptr;
        result = display_session->GetWindowExclusionList(&windows_unknown);
        if (FAILED(result)) {
            display_session->Release();
            return result;
        }
        winrt::com_ptr<IUnknown> windows_owner;
        windows_owner.attach(windows_unknown);
        auto actual = windows_owner.as<
            winrt::Windows::Foundation::Collections::IVectorView<winrt::Windows::UI::WindowId>
        >();
        if (actual.Size() != count) {
            display_session->Release();
            return HRESULT_FROM_WIN32(ERROR_INVALID_DATA);
        }
        for (UINT32 index = 0; index < count; ++index) {
            if (actual.GetAt(index).Value != expected[index].Value) {
                display_session->Release();
                return HRESULT_FROM_WIN32(ERROR_INVALID_DATA);
            }
        }
    } catch (const winrt::hresult_error& error) {
        display_session->Release();
        return error.code();
    } catch (...) {
        display_session->Release();
        return E_FAIL;
    }

    display_session->Release();
    return S_OK;
}

extern "C" __declspec(dllexport) HRESULT d2s_wgc_frame_configuration_iteration(
    void* frame,
    UINT64* configuration_iteration
) {
    if (!frame || !configuration_iteration) {
        return E_INVALIDARG;
    }

    ID2SDirect3D11CaptureFrame3* frame3 = nullptr;
    const HRESULT result = static_cast<IUnknown*>(frame)->QueryInterface(
        __uuidof(ID2SDirect3D11CaptureFrame3),
        reinterpret_cast<void**>(&frame3)
    );
    if (FAILED(result)) {
        return result;
    }

    const HRESULT iteration_result = frame3->ConfigurationIteration(configuration_iteration);
    frame3->Release();
    return iteration_result;
}
