use megenginelite_derive::idx;

macro_rules! p {
    ($any:expr) => {
        println!("{} => {:?}", stringify!($any), $any);
    };
}

fn main() {
    let n = 2;
    p!(idx!(0..n, n-1, ..3;5));
    p!(idx!(0..1+1, 2..3;5));
    p!(idx!(0.., ..3;5, ..n+1, .., ..;n*2, 0..;n*2));
}
