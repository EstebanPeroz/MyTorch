use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Color {
    White,
    Black,
}

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

impl fmt::Display for Color {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Color::White => formatter.write_str("White"),
            Color::Black => formatter.write_str("Black"),
        }
    }
}

impl fmt::Display for Label {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Label::Nothing => formatter.write_str("Nothing"),
            Label::Check(color) => write!(formatter, "Check {color}"),
            Label::Checkmate(color) => write!(formatter, "Checkmate {color}"),
            Label::Stalemate => formatter.write_str("Stalemate"),
        }
    }
}

impl FromStr for Label {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        Label::ALL
            .into_iter()
            .find(|label| label.to_string() == text)
            .ok_or_else(|| format!("unknown label '{text}'"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn displays_subject_wording() {
        assert_eq!(Label::Nothing.to_string(), "Nothing");
        assert_eq!(Label::Check(Color::White).to_string(), "Check White");
        assert_eq!(
            Label::Checkmate(Color::Black).to_string(),
            "Checkmate Black"
        );
        assert_eq!(Label::Stalemate.to_string(), "Stalemate");
    }

    #[test]
    fn text_round_trips_for_every_label() {
        for label in Label::ALL {
            assert_eq!(label.to_string().parse::<Label>(), Ok(label));
        }
    }

    #[test]
    fn indices_cover_every_class_once() {
        for (expected, label) in Label::ALL.into_iter().enumerate() {
            assert_eq!(label.index(), expected);
            assert_eq!(Label::from_index(expected), Some(label));
        }
        assert_eq!(Label::from_index(Label::COUNT), None);
    }

    #[test]
    fn rejects_unknown_labels() {
        for text in [
            "",
            "Check",
            "check White",
            "Checkmate Red",
            "Nothing else",
            " Nothing",
        ] {
            assert!(text.parse::<Label>().is_err(), "{text:?}");
        }
    }
}
