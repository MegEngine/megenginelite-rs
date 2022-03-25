mod index;

use proc_macro::TokenStream;
use syn::parse_macro_input;

/// A helper macro used to slice tensor.
///
/// The syntax is `idx![elem[,elem[,...]]]`, where elem is any of the following:
/// - index: an index to use for taking a subview with respect to that axis.
/// - range: a range with step size 1 to use for slicing that axis.
/// - range;step: a range with step size step to use for slicing that axis. (step >= 1)
///
/// # Example
/// ```no_run
/// # use megenginelite_derive::idx;
/// idx!(0..2, 1, ..3;5);
/// idx!(0..2, 2..3;5);
/// idx!(0.., ..3;5, ..3, .., ..;4, 0..;4);
/// ```
#[proc_macro]
pub fn idx(input: TokenStream) -> TokenStream {
    let sequence = parse_macro_input!(input as index::IndexSequence);
    index::expand(sequence).into()
}
