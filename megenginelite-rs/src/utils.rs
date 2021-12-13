use std::ffi::CString;
use std::path::Path;

pub fn path_to_cstr(path: &Path) -> CString {
    #[cfg(unix)]
    let bytes = {
        use std::os::unix::ffi::OsStrExt;

        path.as_os_str().as_bytes()
    };

    #[cfg(not(unix))]
    let bytes = { path.to_string_lossy().to_string().into_bytes() };

    CString::new(bytes).unwrap()
}
