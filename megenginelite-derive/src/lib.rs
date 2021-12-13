mod index;
mod shape;

use proc_macro::TokenStream;
use syn::parse_macro_input;

#[proc_macro]
pub fn idx(input: TokenStream) -> TokenStream {
    let sequence = parse_macro_input!(input as index::IndexSequence);
    index::expand(sequence).into()
}

#[proc_macro]
pub fn shape(input: TokenStream) -> TokenStream {
    let sequence = parse_macro_input!(input as shape::ShapeSequence);
    shape::expand(sequence).into()
}
