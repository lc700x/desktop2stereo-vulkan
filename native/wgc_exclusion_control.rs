use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};
use pyo3::prelude::*;
use windows::Win32::Foundation::{LPARAM, WPARAM};
use windows::Win32::UI::WindowsAndMessaging::PostThreadMessageW;
use windows_capture::graphics_capture_api::GraphicsCaptureApi;

pub const SET_WINDOW_EXCLUSIONS_MESSAGE: u32 = 0x8000 + 0x54;

#[derive(Default)]
pub struct State {
    next_id: u64,
    pending: VecDeque<(u64, Vec<u64>)>,
    waiting: HashSet<u64>,
    completed: HashMap<u64, Result<u64, String>>,
}

pub type Control = Arc<(Mutex<State>, Condvar)>;

pub fn new_control() -> Control {
    Arc::new((Mutex::new(State::default()), Condvar::new()))
}

pub fn set_window_exclusions(
    control: &Control,
    thread_id: u32,
    py: Python<'_>,
    mut hwnds: Vec<u64>,
) -> PyResult<u64> {
    if hwnds.iter().any(|hwnd| *hwnd == 0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Window handles must be nonzero",
        ));
    }
    hwnds.sort_unstable();
    hwnds.dedup();

    let (state_lock, changed) = &**control;
    let request_id = {
        let mut state = state_lock.lock();
        state.next_id = state.next_id.wrapping_add(1).max(1);
        let id = state.next_id;
        state.pending.push_back((id, hwnds));
        state.waiting.insert(id);
        id
    };
    unsafe {
        let _ = PostThreadMessageW(
            thread_id,
            SET_WINDOW_EXCLUSIONS_MESSAGE,
            WPARAM(request_id as usize),
            LPARAM(0),
        );
    }

    let result = py.allow_threads(|| {
        let mut state = state_lock.lock();
        let deadline = Instant::now() + Duration::from_secs(3);
        loop {
            if let Some(result) = state.completed.remove(&request_id) {
                return result;
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                state.waiting.remove(&request_id);
                state.pending.retain(|(id, _)| *id != request_id);
                return Err("Timed out applying WGC window exclusion".to_owned());
            }
            changed.wait_for(&mut state, remaining);
        }
    });
    result.map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

pub fn process_pending(capture: &GraphicsCaptureApi, control: &Control) {
    let (state_lock, changed) = &**control;
    loop {
        let request = state_lock.lock().pending.pop_front();
        let Some((id, hwnds)) = request else {
            return;
        };
        let result = capture
            .set_window_exclusion_list(&hwnds)
            .map_err(|error| format!("WGC window exclusion failed: {error}"));
        let mut state = state_lock.lock();
        if state.waiting.remove(&id) {
            state.completed.insert(id, result);
        }
        changed.notify_all();
    }
}

pub fn fail_pending(control: &Control, message: &str) {
    let (state_lock, changed) = &**control;
    let mut state = state_lock.lock();
    while let Some((id, _)) = state.pending.pop_front() {
        if state.waiting.remove(&id) {
            state.completed.insert(id, Err(message.to_owned()));
        }
    }
    changed.notify_all();
}
