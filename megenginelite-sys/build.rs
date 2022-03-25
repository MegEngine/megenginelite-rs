#![allow(dead_code)]

use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

fn major() -> i32 {
    env::var("CARGO_PKG_VERSION_MAJOR")
        .unwrap()
        .parse()
        .unwrap()
}

fn minor() -> i32 {
    env::var("CARGO_PKG_VERSION_MINOR")
        .unwrap()
        .parse()
        .unwrap()
}

fn patch() -> i32 {
    env::var("CARGO_PKG_VERSION_PATCH")
        .unwrap()
        .parse()
        .unwrap()
}

fn version() -> String {
    format!("v{}.{}.{}", major(), minor(), patch())
}

fn output() -> PathBuf {
    PathBuf::from(env::var("OUT_DIR").unwrap())
}

fn megbrain() -> PathBuf {
    let mut path = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    path.push("MegEngine");
    path
}

fn lite_header() -> PathBuf {
    megbrain().join("lite/lite-c/include/lite-c/global_c.h")
}

fn lite_c_include_dir() -> PathBuf {
    megbrain().join("lite/lite-c/include/lite-c")
}

fn lite_include_dir() -> PathBuf {
    megbrain().join("lite/include/lite")
}

fn bindgen(path: &Path) -> io::Result<()> {
    let b = bindgen::builder()
        .header(lite_header().to_str().unwrap())
        .dynamic_library_name("MgeLiteDynLib")
        .size_t_is_usize(true)
        .clang_arg(format!("-I{}", lite_c_include_dir().to_str().unwrap()))
        .clang_arg(format!("-I{}", lite_include_dir().to_str().unwrap()))
        .generate()
        .expect("Unable to generate bindings");
    b.write_to_file(path)
}

fn main() {
    bindgen(&output().join("bindings.rs")).unwrap();

    let version = version();
    fs::write(&output().join("version.rs"), {
        let mut vs: Vec<i32> = version[1..]
            .split(".")
            .map(|x| x.parse().unwrap())
            .collect();
        while vs.len() < 3 {
            vs.push(0);
        }
        format!(
            r#"
pub static MAJOR: i32 = {};
pub static MINOR: i32 = {};
pub static PATCH: i32 = {};
                "#,
            vs[0], vs[1], vs[2]
        )
    })
    .unwrap();
}
