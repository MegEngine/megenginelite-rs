/*!

`megenginelite-rs` provides the safe megenginelite wrapper in Rust.

See more in [megenginelite](https://github.com/MegEngine/MegEngine/tree/master/lite).

# Examples

```no_run
# use megenginelite_rs::*;
# #[tokio::main]
# async fn main() -> LiteResult<()> {

// The dynamic library version needs to be greater than or equal to the compiled version.
// It is needless if the feature `auto-load` is enable (default enable).
unsafe {
    load("dynamic_library_path")?;
}

// set some options, and load model
let mut network = Network::builder()
        .dev_id(0)
        .stream_id(0)
        // ...
        .build("model_path")?;

// get an input of the model by name
let mut input = network.io_tensor("input_name").unwrap();
let data = Tensor::host()?;
input.copy_from(&data);

// exec, and wait
network.exec_wait()?;
// exec, async
network.exec().await?;

// get an output of the model by name
let output = network.io_tensor("output_name").unwrap();
println!("{:?}", output.as_slice::<f32>());
# Ok(())
# }
```

# Default feature flags
The following features are turned on by default:

- `auto-load`: automatically load megenginelite dynamic library from the megenginelite python package, and find that by `python3 -c "import megenginelite;print(megenginelite.__file__)"`.

# Optional feature flags
The following features is optional.

- `ndarray-basis`: enable ndarray support.
- `ndarray-rayon`: enable ndarray/rayon feature.

*/

extern crate self as megenginelite_rs;

mod api;
mod builder;
mod global;
mod network;
mod pool;
mod tensor;
mod types;
mod utils;

pub use api::*;
pub use builder::*;
pub use global::*;
pub use network::*;
pub use pool::*;
pub use tensor::*;
pub use types::*;

pub use megenginelite_derive::*;

pub mod ffi {
    pub use megenginelite_sys::*;
}

#[cfg(test)]
fn model_path() -> std::path::PathBuf {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("../resources/shufflenet.mge");
    path
}
