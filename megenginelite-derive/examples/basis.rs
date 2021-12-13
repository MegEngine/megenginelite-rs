use megenginelite_derive::{idx, shape};

macro_rules! p {
    ($any:expr) => {
        println!("{} => {:?}", stringify!($any), $any);
    };
}

fn main() {
    p!(idx!(0..2, 1, ..3;5));
    p!(idx!(0..2, 2..3;5));
    p!(idx!(0.., ..3;5, ..3, .., ..;4, 0..;4));

    p!(shape!(1, 2));
    p!(shape!());
    p!(shape!(1, 2, 3, 4, 5, 6, 7));
    // will report error
    // p!(shape!(1, 2, 3, 4, 5, 6, 7, 8));
}
