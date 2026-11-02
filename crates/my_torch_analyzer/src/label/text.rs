use std::fmt;
use std::str::FromStr;

use super::Label;

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
