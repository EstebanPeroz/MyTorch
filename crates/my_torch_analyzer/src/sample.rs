use std::str::FromStr;

use crate::board::Board;
use crate::label::Label;

const FEN_FIELDS: usize = 6;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sample {
    board: Board,
    label: Option<Label>,
}

impl Sample {
    pub fn board(&self) -> &Board {
        &self.board
    }

    pub fn label(&self) -> Option<Label> {
        self.label
    }
}

impl FromStr for Sample {
    type Err = String;

    fn from_str(line: &str) -> Result<Self, Self::Err> {
        let words: Vec<&str> = line.split_whitespace().collect();
        if words.len() < FEN_FIELDS {
            return Err(format!(
                "expected a FEN with {FEN_FIELDS} fields, found {}",
                words.len()
            ));
        }
        let (fen, label) = words.split_at(FEN_FIELDS);
        let board = Board::from_fen(&fen.join(" "))?;
        let label = match label {
            [] => None,
            words => Some(words.join(" ").parse()?),
        };
        Ok(Sample { board, label })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Color;

    const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    fn parse(line: &str) -> Result<Sample, String> {
        line.parse()
    }

    #[test]
    fn reads_a_fen_without_label() {
        let sample = parse(START).unwrap();
        assert_eq!(sample.board(), &Board::from_fen(START).unwrap());
        assert_eq!(sample.label(), None);
    }

    #[test]
    fn reads_one_word_and_two_word_labels() {
        assert_eq!(
            parse(&format!("{START} Nothing")).unwrap().label(),
            Some(Label::Nothing)
        );
        assert_eq!(
            parse(&format!("{START} Checkmate Black")).unwrap().label(),
            Some(Label::Checkmate(Color::Black))
        );
    }

    #[test]
    fn tolerates_extra_whitespace() {
        let line =
            "  rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR \tw  KQkq - 0 1   Check \t White \r";
        let sample = parse(line).unwrap();
        assert_eq!(sample.board(), &Board::from_fen(START).unwrap());
        assert_eq!(sample.label(), Some(Label::Check(Color::White)));
    }

    #[test]
    fn rejects_malformed_lines() {
        let cases = [
            ("", "expected a FEN with 6 fields, found 0".to_string()),
            (
                "8/8/8/8/8/8/8/k1K5 w - -",
                "expected a FEN with 6 fields, found 4".to_string(),
            ),
            (&format!("{START} Chek"), "unknown label 'Chek'".to_string()),
            (
                &format!("{START} Check  Whit"),
                "unknown label 'Check Whit'".to_string(),
            ),
            (
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 Nothing",
                "invalid FEN 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 Nothing'"
                    .to_string(),
            ),
        ];
        for (line, message) in cases {
            assert_eq!(parse(line), Err(message), "{line:?}");
        }
    }
}
