use anyhow::{anyhow, Result};
use megenginelite_sys::MgeLiteDynLib;
use std::sync::Once;

static mut API: Option<MgeLiteDynLib> = None;
static INIT: Once = Once::new();

fn version(major: i32, minor: i32, patch: i32) -> i32 {
    const UNIT: i32 = 10000;
    major * UNIT * UNIT + minor * UNIT + patch
}

pub(crate) fn api() -> &'static MgeLiteDynLib {
    unsafe {
        API.as_ref()
            .expect("dynamic library [megenginelite] is not found")
    }
}

/// Find and load liblite_shared.so dynamic library.
/// The `path` argument may be either:
/// - A library filename;
/// - The absolute path to the library;
/// - A relative (to the current working directory) path to the library.
/// # Safety
/// see [libloading](https://docs.rs/libloading/latest/libloading/struct.Library.html#method.new)
pub unsafe fn load<P>(path: P) -> Result<()>
where
    P: AsRef<std::ffi::OsStr>,
{
    let mut err = None;
    INIT.call_once(|| match MgeLiteDynLib::new(&path) {
        Ok(lib) => {
            API = Some(lib);
        }
        Err(e) => {
            err = Some(e);
        }
    });

    if let Some(e) = err {
        return Err(anyhow::Error::from(e));
    }

    let mut major = 0i32;
    let mut minor = 0i32;
    let mut patch = 0i32;
    api().LITE_get_version(&mut major, &mut minor, &mut patch);

    let current_version = version(major, minor, patch);
    let min_version = version(
        *megenginelite_sys::MAJOR,
        *megenginelite_sys::MINOR,
        *megenginelite_sys::PATCH,
    );

    if current_version < min_version {
        return Err(anyhow!(
            "This version is not compatible, [expected version >= {}, but get {}]",
            min_version,
            current_version
        ));
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::*;

    #[test]
    fn test_basis() {
        let path = lib_path();
        unsafe { assert!(load(path).is_ok()) };
        api();
    }

    #[test]
    #[should_panic]
    fn test_panic() {
        api();
    }
}
