mod color;
mod index;
mod text;

#[cfg(test)]
mod tests;

pub use color::Color;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Label {
    Nothing,
    Check(Color),
    Checkmate(Color),
    Stalemate,
}

impl Label {
    pub const ALL: [Label; 6] = [
        Label::Nothing,
        Label::Check(Color::White),
        Label::Check(Color::Black),
        Label::Checkmate(Color::White),
        Label::Checkmate(Color::Black),
        Label::Stalemate,
    ];
}
