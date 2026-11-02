use super::Color;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Label {
    Nothing,
    Check(Color),
    Checkmate(Color),
    Stalemate,
}
