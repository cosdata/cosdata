#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum Precedence {
    Lowest = 0,
    Or = 1,
    And = 2,
    Equals = 3,
    Compare = 4,
    Add = 5,
    Multiply = 6,
    Exponentiation = 7,
}

impl Precedence {
    pub fn is_right_associative(&self) -> bool {
        self == &Self::Exponentiation
    }
}
