use proc_macro2::TokenStream;
use quote::quote;
use syn::{
    parse::{Error, Parse, ParseStream, Result},
    punctuated::Punctuated,
    LitInt, Token,
};

struct Index {
    start: Option<u64>,
    end: Option<u64>,
    step: Option<u64>,
}

pub struct IndexSequence {
    seq: Punctuated<Index, Token![,]>,
}

impl Parse for Index {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut start: Option<u64> = None;
        let mut end: Option<u64> = None;
        let mut step: Option<u64> = None;
        if !input.peek(Token![..]) {
            let lit: LitInt = input.parse()?;
            start = Some(lit.base10_parse()?);
        }

        if input.peek(Token![..]) {
            input.parse::<Token![..]>()?;
            if input.peek(LitInt) {
                let lit: LitInt = input.parse()?;
                end = Some(lit.base10_parse()?);
            }
            if input.peek(Token![;]) {
                input.parse::<Token![;]>()?;
                let lit: LitInt = input.parse()?;
                step = Some(lit.base10_parse()?);
            }
        } else {
            if start.is_none() {
                return Err(Error::new(input.span(), "expected a `start` [number]"));
            }
            end = start.map(|x| x + 1);
        }

        Ok(Index { start, end, step })
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
            let start = index.start.unwrap_or(0);
            quote! (#start)
        })
        .collect();
    let end: Vec<_> = index_seq
        .seq
        .iter()
        .map(|index| {
            let end = &index.end;
            if let Some(end) = end {
                quote! ( Some(#end) )
            } else {
                quote!(None)
            }
        })
        .collect();
    let step: Vec<_> = index_seq
        .seq
        .iter()
        .map(|index| {
            let step = index.step.unwrap_or(1);
            quote!(#step)
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
