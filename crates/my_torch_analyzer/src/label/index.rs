use super::Label;

impl Label {
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
