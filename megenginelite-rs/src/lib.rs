/*!

`megenginelite-rs` provides the safe megenginelite wrapper in Rust.

See more in [megenginelite](https://github.com/MegEngine/MegEngine/tree/master/lite).

# Examples

```no_run
use megenginelite_rs::*;
// The dynamic library version needs to be greater than or equal to the compiled version
unsafe {
    load("dynamic_library_path").unwrap();
}

// set some options, and load model
let mut network = Network::builder()
        .dev_id(0)
        .stream_id(0)
        // ...
        .build("model_path");

// get an input of the model by name
let mut input = network.io_tensor("input_name").unwrap();
let data = Tensor::host();
input.copy_from(&data);

// exec, and wait
network.exec_wait();

// get an output of the model by name
let output = network.io_tensor("output_name").unwrap();
println!("{:?}", output.as_slice::<f32>());
```

!*/

extern crate self as megenginelite_rs;

mod api;
mod builder;
mod global;
mod network;
mod tensor;
mod types;
mod utils;

pub use api::*;
pub use builder::*;
pub use global::*;
pub use network::*;
pub use tensor::*;
pub use types::*;

pub use megenginelite_derive::*;
pub use megenginelite_sys::*;

#[cfg(test)]
fn lib_path() -> std::path::PathBuf {
    let mut root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    root.push("liblite_shared.so");
    root
}

#[cfg(test)]
fn model_path() -> std::path::PathBuf {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("../resources/shufflenet.mge");
    path
}
