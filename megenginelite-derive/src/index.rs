use proc_macro2::TokenStream;
use quote::quote;
use syn::{
    parse::{Parse, ParseStream, Result},
    punctuated::Punctuated,
    Expr, ExprRange, Token,
};

struct Index {
    start: Option<Box<Expr>>,
    end: Option<Box<Expr>>,
    step: Option<Expr>,
    one: bool,
}

pub struct IndexSequence {
    seq: Punctuated<Index, Token![,]>,
}

impl Parse for Index {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut start: Option<_> = None;
        let mut end: Option<_> = None;
        let mut step: Option<Expr> = None;

        let mut one = false;
        if input.fork().parse::<ExprRange>().is_ok() {
            if let Ok(ExprRange {
                from, limits, to, ..
            }) = input.parse::<ExprRange>()
            {
                if matches!(limits, syn::RangeLimits::Closed(_)) {
                    one = true;
                }
                start = from;
                end = to;
            }
        } else {
            start = Some(Box::new(input.parse::<Expr>()?));
            one = true;
        }

        if input.peek(Token![;]) {
            input.parse::<Token![;]>()?;
            step = Some(input.parse()?);
        }

        Ok(Index {
            start,
            end,
            step,
            one,
        })
    }
}

impl Parse for IndexSequence {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(IndexSequence {
            seq: input.parse_terminated(Index::parse).unwrap_or_default(),
        })
    }
}

pub fn expand(index_seq: IndexSequence) -> TokenStream {
    let start: Vec<_> = index_seq
        .seq
        .iter()
        .map(|index| {
            if let Some(start) = index.start.as_ref() {
                quote! (#start)
            } else {
                quote!(0)
            }
        })
        .collect();
    let end: Vec<_> = index_seq
        .seq
        .iter()
        .map(|index| {
            let end = &index.end;
            if let Some(end) = end {
                if index.one {
                    quote! ( Some((#end) + 1) )
                } else {
                    quote! ( Some(#end) )
                }
            } else {
                if index.one {
                    let start = index.start.as_ref().unwrap();
                    quote!(Some((#start) + 1))
                } else {
                    quote!(None)
                }
            }
        })
        .collect();
    let step: Vec<_> = index_seq
        .seq
        .iter()
        .map(|index| {
            if let Some(step) = &index.step {
                quote!(#step)
            } else {
                quote!(1)
            }
        })
        .collect();
    quote! {
        megenginelite_rs::SliceInfo {
            start: &[#(#start),*],
            end: &[#(#end),*],
            step: &[#(#step),*],
        }
    }
}
