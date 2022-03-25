megenginelite-rs
--------------
[![Crates.io](https://img.shields.io/crates/v/megenginelite-rs.svg)](https://crates.io/crates/megenginelite-rs)
[![libs.rs](https://img.shields.io/badge/libs.rs-gray.svg)](https://lib.rs/crates/megenginelite-rs)
[![Documentation](https://docs.rs/megenginelite-rs/badge.svg)](https://docs.rs/megenginelite-rs)

A safe megenginelite wrapper in Rust

<strong>⚠️ The project is still in early development, expect bugs, safety issues, and things that don't work ⚠️</strong>

### Install

```toml
[dependencies]
megenginelite-rs = "1.8.2"
```

### How to use

```rust
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
```

see more in [megenginelite](https://github.com/MegEngine/MegEngine/tree/master/lite).
