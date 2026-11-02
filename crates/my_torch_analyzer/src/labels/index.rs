use super::{Color, Label};

impl Label {
    pub const ALL: [Label; 6] = [
        Label::Nothing,
        Label::Check(Color::White),
        Label::Check(Color::Black),
        Label::Checkmate(Color::White),
        Label::Checkmate(Color::Black),
        Label::Stalemate,
    ];

    pub const COUNT: usize = Label::ALL.len();

    pub fn index(self) -> usize {
        Label::ALL
            .iter()
            .position(|&label| label == self)
            .expect("Label::ALL lists every label")
    }

    pub fn from_index(index: usize) -> Option<Label> {
        Label::ALL.get(index).copied()
    }
}
