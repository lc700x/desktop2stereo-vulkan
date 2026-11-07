use parking_lot::{Condvar, Mutex};
use pyo3::prelude::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use windows::Win32::Foundation::HMODULE;
use windows::Win32::Foundation::{LPARAM, LUID, WPARAM};
use windows::Win32::Graphics::Direct3D::{D3D_DRIVER_TYPE_UNKNOWN, D3D_FEATURE_LEVEL_11_1};
use windows::Win32::Graphics::Direct3D11::{
    D3D11_BIND_SHADER_RESOURCE, D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_RESOURCE_MISC_SHARED,
    D3D11_SDK_VERSION, D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT, D3D11CreateDevice, ID3D11Device,
    ID3D11DeviceContext, ID3D11Texture2D,
};
use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIFactory1};
use windows::Win32::System::Threading::GetCurrentThreadId;
use windows::Win32::System::WinRT::{
    CreateDispatcherQueueController, DQTAT_COM_NONE, DQTYPE_THREAD_CURRENT, DispatcherQueueOptions,
};
use windows::Win32::UI::WindowsAndMessaging::{
    DispatchMessageW, GetMessageW, MSG, PostThreadMessageW, TranslateMessage, WM_QUIT,
};
use windows::core::Interface;

use windows_capture::capture::{Context, GraphicsCaptureApiHandler};
use windows_capture::frame::Frame;
use windows_capture::graphics_capture_api::{GraphicsCaptureApi, InternalCaptureControl};
use windows_capture::monitor::Monitor;
use windows_capture::settings::{
    ColorFormat, CursorCaptureSettings, DirtyRegionSettings, DrawBorderSettings,
    GraphicsCaptureItemType, MinimumUpdateIntervalSettings, SecondaryWindowSettings,
};
use windows_capture::window::Window;

#[path = "../../wgc_exclusion_control.rs"]
mod wgc_exclusion_control;

#[derive(thiserror::Error, Debug)]
pub enum WcCudaError {
    #[error("Windows API error: {0}")]
    Windows(#[from] windows::core::Error),
    #[error("AdapterNotFound")]
    AdapterNotFound,
    #[error("Python error: {0}")]
    Python(#[from] PyErr),
    #[error("Capture error: {0}")]
    Capture(String),
    #[error("Item conversion failed")]
    ItemConvertFailed,
}

impl From<WcCudaError> for PyErr {
    fn from(err: WcCudaError) -> PyErr {
        pyo3::exceptions::PyRuntimeError::new_err(err.to_string())
    }
}

pub fn create_d3d_device_with_luid(
    luid: Option<LUID>,
) -> Result<(ID3D11Device, ID3D11DeviceContext), WcCudaError> {
    let adapter = if let Some(luid) = luid {
        let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory1()? };
        let mut adapter = None;
        let mut i = 0;
        loop {
            let current_adapter = match unsafe { factory.EnumAdapters1(i) } {
                Ok(a) => a,
                Err(_) => break,
            };
            let desc = unsafe { current_adapter.GetDesc1()? };
            if desc.AdapterLuid.LowPart == luid.LowPart
                && desc.AdapterLuid.HighPart == luid.HighPart
            {
                adapter = Some(current_adapter);
                break;
            }
            i += 1;
        }
        adapter.ok_or(WcCudaError::AdapterNotFound)?
    } else {
        let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory1()? };
        unsafe { factory.EnumAdapters1(0)? }
    };

    let mut d3d_device = None;
    let mut d3d_device_context = None;
    let feature_levels = [D3D_FEATURE_LEVEL_11_1];

    unsafe {
        D3D11CreateDevice(
            &adapter,
            D3D_DRIVER_TYPE_UNKNOWN,
            HMODULE::default(),
            D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            Some(&feature_levels),
            D3D11_SDK_VERSION,
            Some(&mut d3d_device),
            None,
            Some(&mut d3d_device_context),
        )?;
    }

    Ok((d3d_device.unwrap(), d3d_device_context.unwrap()))
}

#[pyclass]
#[derive(Clone)]
pub struct WcGpuFrame {
    pub texture: ID3D11Texture2D,
    #[pyo3(get)]
    pub width: u32,
    #[pyo3(get)]
    pub height: u32,
    #[pyo3(get)]
    pub original_width: u32,
    #[pyo3(get)]
    pub original_height: u32,
    #[pyo3(get)]
    pub configuration_iteration: u64,
}

#[pymethods]
impl WcGpuFrame {
    #[getter]
    fn texture_ptr(&self) -> usize {
        self.texture.as_raw() as usize
    }
}

struct FrameSlot {
    data: Mutex<(Option<WcGpuFrame>, u64)>,
    condvar: Condvar,
}

struct InnerHandler {
    shared_slot: Arc<FrameSlot>,
    shared_texture: Option<ID3D11Texture2D>,
    current_aligned_width: u32,
    current_aligned_height: u32,
    thread_id: u32,
    track_configuration_iteration: bool,
}

impl GraphicsCaptureApiHandler for InnerHandler {
    type Flags = (Arc<FrameSlot>, u32, bool);
    type Error = WcCudaError;

    fn new(ctx: Context<Self::Flags>) -> Result<Self, Self::Error> {
        Ok(Self {
            shared_slot: ctx.flags.0,
            thread_id: ctx.flags.1,
            track_configuration_iteration: ctx.flags.2,
            shared_texture: None,
            current_aligned_width: 0,
            current_aligned_height: 0,
        })
    }

    fn on_frame_arrived(
        &mut self,
        frame: &mut Frame,
        _capture_control: InternalCaptureControl,
    ) -> Result<(), Self::Error> {
        let device = frame.device();
        let context = frame.device_context();
        let width = frame.width();
        let height = frame.height();
        let format = frame.desc().Format;
        let configuration_iteration = if self.track_configuration_iteration {
            frame
                .configuration_iteration()
                .map_err(|error| WcCudaError::Capture(error.to_string()))?
        } else {
            0
        };

        let aligned_width = (width + 63) & !63;
        let aligned_height = (height + 31) & !31;

        let recreate = self.shared_texture.is_none()
            || self.current_aligned_width != aligned_width
            || self.current_aligned_height != aligned_height;

        if recreate {
            let desc = D3D11_TEXTURE2D_DESC {
                Width: aligned_width,
                Height: aligned_height,
                MipLevels: 1,
                ArraySize: 1,
                Format: format,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Usage: D3D11_USAGE_DEFAULT,
                BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
                CPUAccessFlags: 0,
                MiscFlags: D3D11_RESOURCE_MISC_SHARED.0 as u32,
            };
            let mut tex = None;
            unsafe { device.CreateTexture2D(&desc, None, Some(&mut tex))? };
            self.shared_texture = Some(tex.unwrap());
            self.current_aligned_width = aligned_width;
            self.current_aligned_height = aligned_height;
        }

        if let Some(ref tex) = self.shared_texture {
            unsafe {
                context.CopySubresourceRegion(tex, 0, 0, 0, 0, frame.as_raw_texture(), 0, None);
                // Flush is a non-blocking submit: it pushes the copy into the GPU
                // queue so the shared texture is visible to the CUDA interop map.
                // It does NOT stall the CPU, so it is not an FPS bottleneck.
                context.Flush();
            }
            let gpu_frame = WcGpuFrame {
                texture: tex.clone(),
                width: aligned_width,
                height: aligned_height,
                original_width: width,
                original_height: height,
                configuration_iteration,
            };

            // Update the shared slot with the latest frame
            let mut data = self.shared_slot.data.lock();
            data.0 = Some(gpu_frame);
            data.1 += 1;
            self.shared_slot.condvar.notify_one();
        }
        Ok(())
    }

    fn on_closed(&mut self) -> Result<(), Self::Error> {
        unsafe {
            let _ = PostThreadMessageW(self.thread_id, WM_QUIT, WPARAM(0), LPARAM(0));
        }
        Ok(())
    }
}

#[pyclass]
pub struct WcCapture {
    luid: Option<(u32, i32)>,
    monitor_index: Option<usize>,
    window_hwnd: Option<isize>,
    window_title: Option<String>,
    initial_excluded_window_handles: Option<Vec<u64>>,
    shared_slot: Arc<FrameSlot>,
    thread_handle: Mutex<Option<thread::JoinHandle<()>>>,
    thread_id: Arc<Mutex<Option<u32>>>,
    should_stop: Arc<AtomicBool>,
    last_error: Arc<Mutex<Option<String>>>,
    window_exclusion_control: wgc_exclusion_control::Control,
    required_configuration_iteration: Arc<AtomicU64>,
}

#[pymethods]
impl WcCapture {
    #[new]
    #[pyo3(signature = (luid=None, monitor_index=None, window_hwnd=None, window_title=None, excluded_window_handles=None))]
    fn new(
        luid: Option<(u32, i32)>,
        monitor_index: Option<usize>,
        window_hwnd: Option<isize>,
        window_title: Option<String>,
        excluded_window_handles: Option<Vec<u64>>,
    ) -> Self {
        Self {
            luid,
            monitor_index,
            window_hwnd,
            window_title,
            initial_excluded_window_handles: excluded_window_handles,
            shared_slot: Arc::new(FrameSlot {
                data: Mutex::new((None, 0)),
                condvar: Condvar::new(),
            }),
            thread_handle: Mutex::new(None),
            thread_id: Arc::new(Mutex::new(Option::None)),
            should_stop: Arc::new(AtomicBool::new(false)),
            last_error: Arc::new(Mutex::new(None)),
            window_exclusion_control: wgc_exclusion_control::new_control(),
            required_configuration_iteration: Arc::new(AtomicU64::new(0)),
        }
    }

    fn start(&self) -> PyResult<()> {
        let mut handle_lock = self.thread_handle.lock();
        if handle_lock.is_some() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Capture already running",
            ));
        }

        let luid = self.luid.map(|(low, high)| LUID {
            LowPart: low,
            HighPart: high,
        });
        let monitor_index = self.monitor_index;
        let window_hwnd = self.window_hwnd;
        let window_title = self.window_title.clone();
        let excluded_window_handles = self.initial_excluded_window_handles.clone();
        let thread_id_ptr = self.thread_id.clone();
        let should_stop = self.should_stop.clone();
        let last_error = self.last_error.clone();
        let shared_slot = self.shared_slot.clone();
        let window_exclusion_control = self.window_exclusion_control.clone();
        let required_configuration_iteration = self.required_configuration_iteration.clone();

        should_stop.store(false, Ordering::Relaxed);
        required_configuration_iteration.store(0, Ordering::Release);
        *last_error.lock() = None;

        let handle = thread::spawn(move || {
            let res: Result<(), WcCudaError> = (|| {
                if should_stop.load(Ordering::Relaxed) {
                    return Ok(());
                }

                let (d3d_device, d3d_device_context) = create_d3d_device_with_luid(luid)?;

                let item_type: GraphicsCaptureItemType = if let Some(title) = window_title {
                    Window::from_contains_name(&title)
                        .map_err(|_| WcCudaError::Capture(format!("Window not found: {}", title)))?
                        .try_into()
                        .map_err(|_| WcCudaError::ItemConvertFailed)?
                } else if let Some(hwnd) = window_hwnd {
                    Window::from_raw_hwnd(hwnd as *mut std::ffi::c_void)
                        .try_into()
                        .map_err(|_| WcCudaError::ItemConvertFailed)?
                } else {
                    let idx = monitor_index.unwrap_or(1);
                    Monitor::from_index(idx)
                        .map_err(|_| WcCudaError::ItemConvertFailed)?
                        .try_into()
                        .map_err(|_| WcCudaError::ItemConvertFailed)?
                };

                let options = DispatcherQueueOptions {
                    dwSize: std::mem::size_of::<DispatcherQueueOptions>() as u32,
                    threadType: DQTYPE_THREAD_CURRENT,
                    apartmentType: DQTAT_COM_NONE,
                };
                let _controller = unsafe { CreateDispatcherQueueController(options)? };
                let tid = unsafe { GetCurrentThreadId() };
                *thread_id_ptr.lock() = Some(tid);

                if should_stop.load(Ordering::Relaxed) {
                    return Ok(());
                }

                let ctx = Context {
                    flags: (shared_slot, tid, excluded_window_handles.is_some()),
                    device: d3d_device.clone(),
                    device_context: d3d_device_context.clone(),
                };
                let handler = InnerHandler::new(ctx)?;
                let callback_arc = Arc::new(Mutex::new(handler));

                let result = Arc::new(Mutex::new(None));
                let mut capture = GraphicsCaptureApi::new(
                    d3d_device,
                    d3d_device_context,
                    item_type,
                    callback_arc,
                    CursorCaptureSettings::Default,
                    DrawBorderSettings::Default,
                    SecondaryWindowSettings::Default,
                    MinimumUpdateIntervalSettings::Default,
                    DirtyRegionSettings::Default,
                    ColorFormat::Bgra8,
                    tid,
                    result,
                )
                .map_err(|e| WcCudaError::Capture(e.to_string()))?;

                if let Some(hwnds) = excluded_window_handles.as_deref() {
                    let configuration_iteration =
                        capture.set_window_exclusion_list(hwnds).map_err(|error| {
                            WcCudaError::Capture(format!(
                                "WGC window exclusion is unavailable: {error}"
                            ))
                        })?;
                    if configuration_iteration == 0 {
                        return Err(WcCudaError::Capture(
                            "WGC window exclusion returned configuration iteration 0".to_owned(),
                        ));
                    }
                    required_configuration_iteration
                        .store(configuration_iteration, Ordering::Release);
                }

                capture
                    .start_capture()
                    .map_err(|e| WcCudaError::Capture(e.to_string()))?;

                let mut message = MSG::default();
                unsafe {
                    while GetMessageW(&mut message, None, 0, 0).as_bool() {
                        if message.message == wgc_exclusion_control::SET_WINDOW_EXCLUSIONS_MESSAGE {
                            wgc_exclusion_control::process_pending(
                                &capture,
                                &window_exclusion_control,
                            );
                            continue;
                        }
                        let _ = TranslateMessage(&message);
                        DispatchMessageW(&message);
                    }
                }
                wgc_exclusion_control::fail_pending(
                    &window_exclusion_control,
                    "WGC capture session stopped",
                );
                Ok(())
            })();

            if let Err(e) = res {
                *last_error.lock() = Some(e.to_string());
            }
            wgc_exclusion_control::fail_pending(
                &window_exclusion_control,
                "WGC capture session stopped",
            );
            *thread_id_ptr.lock() = None;
        });
        *handle_lock = Some(handle);
        Ok(())
    }

    fn stop(&self) {
        self.should_stop.store(true, Ordering::Relaxed);
        if let Some(tid) = *self.thread_id.lock() {
            unsafe {
                let _ = PostThreadMessageW(tid, WM_QUIT, WPARAM(0), LPARAM(0));
            }
        }

        let handle = self.thread_handle.lock().take();
        if let Some(h) = handle {
            if h.thread().id() != thread::current().id() {
                let _ = h.join();
            }
        }
        let mut data = self.shared_slot.data.lock();
        data.0 = None;
    }

    fn is_alive(&self) -> bool {
        let tid = self.thread_id.lock();
        let handle = self.thread_handle.lock();
        handle.is_some() && tid.is_some()
    }

    fn get_last_error(&self) -> Option<String> {
        self.last_error.lock().clone()
    }

    #[getter]
    fn required_configuration_iteration(&self) -> u64 {
        self.required_configuration_iteration.load(Ordering::Acquire)
    }

    fn set_excluded_windows(&self, py: Python<'_>, hwnds: Vec<u64>) -> PyResult<u64> {
        let thread_id = (*self.thread_id.lock()).ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Capture session is not running")
        })?;
        let iteration = wgc_exclusion_control::set_window_exclusions(
            &self.window_exclusion_control,
            thread_id,
            py,
            hwnds,
        )?;
        self.required_configuration_iteration
            .store(iteration, Ordering::Release);
        Ok(iteration)
    }

    #[pyo3(signature = (last_id, timeout=None))]
    fn get_frame(
        &self,
        py: Python,
        last_id: u64,
        timeout: Option<f32>,
    ) -> Option<(WcGpuFrame, u64)> {
        let slot = self.shared_slot.clone();
        let required_configuration_iteration = self.required_configuration_iteration.clone();
        py.allow_threads(move || {
            let mut data = slot.data.lock();
            let deadline = timeout.map(|seconds| Instant::now() + Duration::from_secs_f32(seconds));
            loop {
                let required_iteration =
                    required_configuration_iteration.load(Ordering::Acquire);
                if data.1 > last_id {
                    if let Some(frame) = data.0.as_ref() {
                        if frame.configuration_iteration >= required_iteration {
                            return Some((frame.clone(), data.1));
                        }
                    }
                }

                let Some(deadline) = deadline else {
                    return None;
                };
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    return None;
                }
                let _ = slot.condvar.wait_for(&mut data, remaining);
            }
        })
    }
}

#[pymodule]
fn _wc_cuda(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<WcCapture>()?;
    m.add_class::<WcGpuFrame>()?;
    Ok(())
}
