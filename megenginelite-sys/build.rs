use cmd_lib::*;
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
    output().join(format!("megbrain-{}", version()))
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

fn fetch() -> io::Result<()> {
    const REPO: &str = "https://github.com/MegEngine/MegEngine.git";
    let version = version();
    let base_path = output();
    let dest = megbrain();
    let err_header = lite_include_dir().join("common_enum_c.h"); // workaround for https://jira.megvii-inc.com/browse/MGE-3115
    let _ = std::fs::remove_dir_all(&dest);

    let tag = run_fun!(
        git ls-remote --tags $REPO | grep -o -E "v.*?[^\\^{}]$" | grep "$version" | tail -n 1
    )?;

    run_cmd!(
        cd $base_path;
        git clone --depth=1 -b $tag $REPO $dest;
        sed -i "s/LiteCDataType/LiteDataType/g" $err_header;
    )?;

    Ok(())
}

fn bindgen(path: &Path) -> io::Result<()> {
    let b = bindgen::builder()
        .header(lite_header().to_str().unwrap())
        .dynamic_library_name("MgeLiteDynLib")
        .clang_arg(format!("-I{}", lite_c_include_dir().to_str().unwrap()))
        .clang_arg(format!("-I{}", lite_include_dir().to_str().unwrap()))
        .generate()
        .expect("Unable to generate bindings");
    b.write_to_file(path)
}

fn main() {
    if fs::metadata(&lite_c_include_dir()).is_err() || fs::metadata(&lite_include_dir()).is_err() {
        fetch().unwrap();
    }
    bindgen(&output().join("bindings.rs")).unwrap();
}
