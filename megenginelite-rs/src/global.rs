//! Some global functions

use super::{api, utils};
use crate::ffi::*;
use std::ffi::CString;
use std::path::Path;

/// Get device count
pub fn device_count(ty: LiteDeviceType) -> usize {
    let mut count = 0;
    unsafe { api().LITE_get_device_count(ty, &mut count) };
    count
}

/// Try to coalesce all free memory in megenine
pub fn try_coalesce_all_free_memory() {
    unsafe {
        api().LITE_try_coalesce_all_free_memory();
    }
}

/// Update decryption key by name.
///
/// `decrypt_name` the name of the decryption, which will act as the
/// hash key to find the decryption method.
/// `key` the decryption key of the method, if the size of key is zero,
/// it will not be updated
pub fn update_decryption(name: &str, key: &[u8]) {
    let name = CString::new(name).unwrap();
    unsafe { api().LITE_update_decryption_or_key(name.as_ptr(), None, key.as_ptr(), key.len()) };
}

/// Set the algo policy cache file for CPU/CUDA ...
///
/// `path` is the file path which store the cache
/// `always_sync` sync the cache when cache updated
pub fn set_persistent_cache(path: impl AsRef<Path>, always_sync: bool) {
    let path = utils::path_to_cstr(path.as_ref());
    unsafe { api().LITE_set_persistent_cache(path.as_ptr(), always_sync as i32) };
}

/// Dump the algo policy cache to file, if the network is set to profile
///
/// when forward, though this the algo policy will dump to file
/// `cache_path` is the file path which store the cache
pub fn dump_persistent_cache(path: impl AsRef<Path>) {
    let path = utils::path_to_cstr(path.as_ref());
    unsafe { api().LITE_dump_persistent_cache(path.as_ptr()) };
}

/// Register the physical and virtual address pair to the mge, some device
/// need the map from physical to virtual.
pub fn register_memory_pair(vir_ptr: usize, phy_ptr: usize, length: usize, device: LiteDeviceType) {
    unsafe {
        api().LITE_register_memory_pair(
            vir_ptr as *mut _,
            phy_ptr as *mut _,
            length,
            device,
            LiteBackend_LITE_DEFAULT,
        )
    };
}

/// Clear the physical and virtual address pair in mge.
pub fn clear_memory_pair(vir_ptr: usize, phy_ptr: usize, device: LiteDeviceType) {
    unsafe {
        api().LITE_clear_memory_pair(
            vir_ptr as *mut _,
            phy_ptr as *mut _,
            device,
            LiteBackend_LITE_DEFAULT,
        )
    };
}
