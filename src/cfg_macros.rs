#![allow(unused_macros)]

#[macro_export]
macro_rules! cfg_grpc {
     ($($item:item)*) => {
        $(
            #[cfg(feature = "grpc-server")]
            $item
        )*
    }
}
