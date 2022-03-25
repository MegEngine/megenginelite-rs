use crate::types::*;
use megenginelite_sys::MgeLiteDynLib;
use std::ffi::CStr;
use std::sync::{Mutex, Once};

#[doc(hidden)]
pub trait IntoLiteRst {
    fn into_rst(self) -> LiteResult<()>;
}

impl IntoLiteRst for i32 {
    fn into_rst(self) -> LiteResult<()> {
        match self {
            0 => Ok(()),
            _ => {
                let descp = unsafe {
                    let api = API
                        .as_ref()
                        .expect("dynamic library [megenginelite] is not found");
                    CStr::from_ptr(api.LITE_get_last_error())
                }
                .to_str()
                .unwrap()
                .to_owned();
                Err(LiteError::MGELiteError(descp))
            }
        }
    }
}

#[doc(hidden)]
pub static mut API: Option<MgeLiteDynLib> = None;

#[cfg(feature = "auto-load")]
fn auto_load() -> Option<()> {
    use std::path::PathBuf;
    use std::process::Command;
    if let Ok(output) = Command::new("python3")
        .args(["-c", "import megenginelite;print(megenginelite.__file__)"])
        .output()
    {
        let output = String::from_utf8(output.stdout).ok()?;
        let mut dir = PathBuf::from(output);
        dir.pop();
        dir.push("libs");
        for name in std::fs::read_dir(&dir).ok()? {
            if let Some(path) = name.ok() {
                let path = path.path();
                if let Some(ext) = path.extension() {
                    if ext == "so" {
                        unsafe { load(path) }.ok();
                    }
                }
            }
        }
    }
    None
}

lazy_static::lazy_static! {
    static ref INIT: Mutex<()> = Mutex::new(());
}

#[cfg(feature = "auto-load")]
static INIT_ONCE: Once = Once::new();

#[doc(hidden)]
pub fn api() -> &'static MgeLiteDynLib {
    #[cfg(feature = "auto-load")]
    INIT_ONCE.call_once(|| {
        auto_load();
    });
    unsafe {
        API.as_ref()
            .expect("dynamic library [megenginelite] is not found")
    }
}

/// Find and load megenginelite dynamic library.
///
/// The `path` argument may be either:
/// - A library filename;
/// - The absolute path to the library;
/// - A relative (to the current working directory) path to the library.
/// # Safety
/// see [libloading](https://docs.rs/libloading/latest/libloading/struct.Library.html#method.new)
pub unsafe fn load<P>(path: P) -> LiteResult<()>
where
    P: AsRef<std::ffi::OsStr>,
{
    let mut err = None;
    let _l = INIT.lock().unwrap();
    match MgeLiteDynLib::new(&path) {
        Ok(lib) => {
            API = Some(lib);
        }
        Err(e) => {
            err = Some(e);
        }
    };

    if err.is_some() {
        return Err(LiteError::LoadingFault);
    }

    check_version()
}

fn check_version() -> LiteResult<()> {
    let mut major = 0i32;
    let mut minor = 0i32;
    let mut patch = 0i32;
    unsafe {
        let api = API
            .as_ref()
            .expect("dynamic library [megenginelite] is not found");
        api.LITE_get_version(&mut major, &mut minor, &mut patch)
    };

    let current_version = version(major, minor, patch);
    let min_version = version(
        megenginelite_sys::MAJOR,
        megenginelite_sys::MINOR,
        megenginelite_sys::PATCH,
    );

    if current_version < min_version {
        return Err(LiteError::VersionNotMatch(format!(
            "This version is not compatible, [expected version >= {}, but get {}]",
            min_version, current_version
        )));
    }
    Ok(())
}

fn version(major: i32, minor: i32, patch: i32) -> i32 {
    const UNIT: i32 = 10000;
    major * UNIT * UNIT + minor * UNIT + patch
}
