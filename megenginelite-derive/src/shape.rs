use proc_macro2::TokenStream;
use quote::quote;
use syn::{
    parse::{Parse, ParseStream, Result},
    punctuated::Punctuated,
    spanned::Spanned,
    Error, LitInt, Token,
};

static LAYOUT_MAX_DIM: usize = 7;

pub struct ShapeSequence {
    seq: Punctuated<LitInt, Token![,]>,
}

impl Parse for ShapeSequence {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(ShapeSequence {
            seq: input.parse_terminated(LitInt::parse).unwrap_or_default(),
        })
    }
}

pub fn expand(shape: ShapeSequence) -> TokenStream {
    if shape.seq.len() > LAYOUT_MAX_DIM {
        return Error::new(
            shape.seq.span(),
            format!(
                "The maximum dim supported does not exceed {}",
                LAYOUT_MAX_DIM
            ),
        )
        .to_compile_error();
    }
    let rest = vec![quote!(0); LAYOUT_MAX_DIM - shape.seq.len()];
    let shape = &shape.seq;
    if shape.is_empty() {
        quote! {
            [#(#rest),*]
        }
    } else {
        quote! {
            [#shape, #(#rest),*]
        }
    }
}
