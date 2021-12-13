#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(deref_nullptr)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

use lazy_static::lazy_static;

lazy_static! {
    pub static ref MAJOR: i32 = env!("CARGO_PKG_VERSION_MAJOR").parse().unwrap();
}

lazy_static! {
    pub static ref MINOR: i32 = env!("CARGO_PKG_VERSION_MINOR").parse().unwrap();
}

lazy_static! {
    pub static ref PATCH: i32 = env!("CARGO_PKG_VERSION_PATCH").parse().unwrap();
}
