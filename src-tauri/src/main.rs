#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use std::sync::{Arc, RwLock};

#[derive(Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct BackendState {
    pub ready: bool,
    pub port: u16,
    pub url: String,
    pub error: Option<String>,
}

#[tauri::command]
fn get_sidecar_url(
    state: tauri::State<'_, Arc<RwLock<BackendState>>>,
) -> Result<String, String> {
    let read_guard = state.read().map_err(|e| e.to_string())?;
    if read_guard.ready {
        Ok(read_guard.url.clone())
    } else {
        Err("Engine is still initializing".to_string())
    }
}

#[tauri::command]
fn get_backend_status(
    state: tauri::State<'_, Arc<RwLock<BackendState>>>,
) -> Result<BackendState, String> {
    let read_guard = state.read().map_err(|e| e.to_string())?;
    Ok(read_guard.clone())
}

#[cfg(target_os = "windows")]
mod job_object {
    use std::mem::size_of;
    use std::os::raw::c_void;

    type HANDLE = *mut c_void;
    type BOOL = i32;
    type DWORD = u32;

    const JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE: DWORD = 0x2000;
    const JOB_OBJECT_EXTENDED_LIMIT_INFORMATION: i32 = 9;
    const PROCESS_SET_QUOTA: DWORD = 0x0100;
    const PROCESS_TERMINATE: DWORD = 0x0001;

    #[repr(C)]
    struct IoCounters {
        read_operation_count: u64,
        write_operation_count: u64,
        other_operation_count: u64,
        read_transfer_count: u64,
        write_transfer_count: u64,
        other_transfer_count: u64,
    }

    #[repr(C)]
    struct JobobjectBasicLimitInformation {
        per_process_user_time_limit: i64,
        per_job_user_time_limit: i64,
        limit_flags: DWORD,
        minimum_working_set_size: usize,
        maximum_working_set_size: usize,
        active_process_limit: DWORD,
        affinity: usize,
        priority_class: DWORD,
        scheduling_class: DWORD,
    }

    #[repr(C)]
    struct JobobjectExtendedLimitInformation {
        basic_limit_information: JobobjectBasicLimitInformation,
        io_info: IoCounters,
        process_memory_limit: usize,
        job_memory_limit: usize,
        peak_process_memory_used: usize,
        peak_job_memory_used: usize,
    }

    extern "system" {
        fn CreateJobObjectW(lp_job_attributes: *mut c_void, lp_name: *const u16) -> HANDLE;
        fn SetInformationJobObject(
            h_job: HANDLE,
            job_object_info_class: i32,
            lp_job_object_info: *const c_void,
            cb_job_object_info_length: DWORD,
        ) -> BOOL;
        fn AssignProcessToJobObject(h_job: HANDLE, h_process: HANDLE) -> BOOL;
        fn OpenProcess(dw_desired_access: DWORD, b_inherit_handle: BOOL, dw_process_id: DWORD) -> HANDLE;
        fn CloseHandle(h_object: HANDLE) -> BOOL;
    }

    pub struct ProcessJob {
        job_handle: HANDLE,
    }

    unsafe impl Send for ProcessJob {}
    unsafe impl Sync for ProcessJob {}

    impl ProcessJob {
        pub fn new() -> Option<Self> {
            unsafe {
                let job = CreateJobObjectW(std::ptr::null_mut(), std::ptr::null());
                if job.is_null() {
                    return None;
                }

                let mut info: JobobjectExtendedLimitInformation = std::mem::zeroed();
                info.basic_limit_information.limit_flags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;

                let success = SetInformationJobObject(
                    job,
                    JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                    &info as *const _ as *const c_void,
                    size_of::<JobobjectExtendedLimitInformation>() as DWORD,
                );

                if success == 0 {
                    CloseHandle(job);
                    return None;
                }

                Some(ProcessJob { job_handle: job })
            }
        }

        pub fn assign_pid(&self, pid: u32) -> bool {
            unsafe {
                let proc_handle = OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, 0, pid);
                if proc_handle.is_null() {
                    return false;
                }
                let success = AssignProcessToJobObject(self.job_handle, proc_handle);
                CloseHandle(proc_handle);
                success != 0
            }
        }
    }

    impl Drop for ProcessJob {
        fn drop(&mut self) {
            unsafe {
                if !self.job_handle.is_null() {
                    CloseHandle(self.job_handle);
                }
            }
        }
    }
}

#[cfg(target_os = "windows")]
mod single_instance {
    use std::os::raw::c_void;
    type HANDLE = *mut c_void;
    type BOOL = i32;
    type DWORD = u32;

    const ERROR_ALREADY_EXISTS: DWORD = 183;

    extern "system" {
        fn CreateMutexW(lp_mutex_attributes: *mut c_void, b_initial_owner: BOOL, lp_name: *const u16) -> HANDLE;
        fn GetLastError() -> DWORD;
        fn CloseHandle(h_object: HANDLE) -> BOOL;
    }

    pub struct SingleInstanceLock {
        handle: HANDLE,
    }

    impl SingleInstanceLock {
        pub fn acquire(name: &str) -> Result<Self, ()> {
            let wide_name: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
            unsafe {
                let handle = CreateMutexW(std::ptr::null_mut(), 1, wide_name.as_ptr());
                if handle.is_null() || GetLastError() == ERROR_ALREADY_EXISTS {
                    if !handle.is_null() {
                        CloseHandle(handle);
                    }
                    return Err(());
                }
                Ok(SingleInstanceLock { handle })
            }
        }
    }

    impl Drop for SingleInstanceLock {
        fn drop(&mut self) {
            unsafe {
                if !self.handle.is_null() {
                    CloseHandle(self.handle);
                }
            }
        }
    }
}

fn main() {
    #[cfg(target_os = "windows")]
    let _instance_lock = match single_instance::SingleInstanceLock::acquire("Local\\PhotoVault_App_Instance_Mutex") {
        Ok(lock) => lock,
        Err(_) => {
            eprintln!("[PhotoVault] Another instance is already running. Exiting.");
            std::process::exit(0);
        }
    };

    let backend_state = Arc::new(RwLock::new(BackendState::default()));
    let backend_state_clone = Arc::clone(&backend_state);

    #[cfg(target_os = "windows")]
    let job_mgr = Arc::new(job_object::ProcessJob::new());
    #[cfg(target_os = "windows")]
    let job_mgr_for_setup = Arc::clone(&job_mgr);

    let builder = tauri::Builder::default()
        .manage(backend_state);

    #[cfg(target_os = "windows")]
    let builder = builder.manage(job_mgr);

    builder
        .setup(move |_app| {
            use tauri::api::process::Command;
            
            // Spawn photovault-engine sidecar on startup with dynamic ephemeral port
            match Command::new_sidecar("photovault-engine") {
                Ok(cmd) => match cmd.args(["--port", "0"]).spawn() {
                    Ok((mut rx, child)) => {
                        let pid = child.pid();
                        println!("[PhotoVault] Sidecar spawned with PID: {}", pid);

                        #[cfg(target_os = "windows")]
                        if let Some(ref job) = *job_mgr_for_setup {
                            if job.assign_pid(pid) {
                                println!("[PhotoVault] Child process assigned to Windows Job Object.");
                            }
                        }

                        let state_ref = Arc::clone(&backend_state_clone);
                        tauri::async_runtime::spawn(async move {
                            while let Some(event) = rx.recv().await {
                                match event {
                                    tauri::api::process::CommandEvent::Stdout(line) => {
                                        println!("[engine]: {}", line);
                                        // Look for JSON handshake: {"status": "ready", "port": ..., "url": ...}
                                        if line.contains("\"status\": \"ready\"") || line.contains("\"status\":\"ready\"") {
                                            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) {
                                                if let (Some(port), Some(url)) = (v.get("port").and_then(|p| p.as_u64()), v.get("url").and_then(|u| u.as_str())) {
                                                    if let Ok(mut state) = state_ref.write() {
                                                        state.ready = true;
                                                        state.port = port as u16;
                                                        state.url = url.to_string();
                                                        state.error = None;
                                                        println!("[PhotoVault] Engine is READY at: {}", url);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    tauri::api::process::CommandEvent::Stderr(line) => {
                                        eprintln!("[engine err]: {}", line);
                                    }
                                    tauri::api::process::CommandEvent::Error(err) => {
                                        eprintln!("[engine error event]: {}", err);
                                        if let Ok(mut state) = state_ref.write() {
                                            state.error = Some(err);
                                        }
                                    }
                                    tauri::api::process::CommandEvent::Terminated(payload) => {
                                        println!("[engine terminated]: {:?}", payload);
                                        if let Ok(mut state) = state_ref.write() {
                                            state.ready = false;
                                            state.error = Some("Engine process terminated".to_string());
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        });
                    }
                    Err(e) => {
                        eprintln!("[error] Failed to spawn sidecar: {}", e);
                        if let Ok(mut state) = backend_state_clone.write() {
                            state.error = Some(e.to_string());
                        }
                    }
                },
                Err(e) => {
                    eprintln!("[error] Failed to setup sidecar: {}", e);
                    if let Ok(mut state) = backend_state_clone.write() {
                        state.error = Some(e.to_string());
                    }
                }
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![get_sidecar_url, get_backend_status])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
