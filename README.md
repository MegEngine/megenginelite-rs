megenginelite-rs
--------------
[![Crates.io](https://img.shields.io/crates/v/minifb.svg)](https://crates.io/crates/megenginelite-rs)
[![Documentation](https://docs.rs/minifb/badge.svg)](https://docs.rs/megenginelite-rs)

A safe megenginelite wrapper in Rust

<strong>⚠️ The project is still in early development, expect bugs, safety issues, and things that don't work ⚠️</strong>

### Install

```toml
[dependencies]
megenginelite-rs = "1.7.0"
```

### How to use

```rust
// The dynamic library version needs to be greater than or equal to the compiled version 
load(dynamic_library_path)?; 

// set some options, and load model
let mut network = Network::builder() 
        .config(config)
        .io(network_io)
        .dev_id(dev_id)
        .stream_id(stream_id)
        ...
        .build(model_path);

// get an input of the model by name
let mut input = network.io_tensor("input_name").unwrap(); 
input.copy_from(data);

// exec, and wait
network.exec_wait(); 

// get an output of the model by name
let output = network.io_tensor("output_name").unwrap(); 
println!("{:?}", output.as_slice::<f32>());
```

see more in [megenginelite](https://github.com/MegEngine/MegEngine/tree/master/lite).

### FAQ

1. how to change the supported min version ?

just modify the `version` in `megenginelite-sys/Cargo.toml`.
