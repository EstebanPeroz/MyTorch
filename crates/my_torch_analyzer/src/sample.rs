use std::ops::Range;
use std::str::FromStr;

use crate::board::Board;
use crate::label::Label;

const FEN_FIELDS: usize = 6;
const COUNTERS: Range<usize> = 4..FEN_FIELDS;

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

    fn is_counter(word: &str) -> bool {
        !word.is_empty() && word.bytes().all(|byte| byte.is_ascii_digit())
    }
}

impl FromStr for Sample {
    type Err = String;

    fn from_str(line: &str) -> Result<Self, Self::Err> {
        let words: Vec<&str> = line.split_whitespace().collect();
        if words.len() < COUNTERS.start {
            return Err(format!(
                "expected a FEN with {FEN_FIELDS} fields, found {}",
                words.len()
            ));
        }
        let has_counters = words
            .get(COUNTERS)
            .is_some_and(|counters| counters.iter().all(|word| Sample::is_counter(word)));
        if !has_counters {
            return Err(
                "expected halfmove and fullmove counters after the en passant field".into(),
            );
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
    use crate::test_fens::START;

    const COUNTERS_MISSING: &str =
        "expected halfmove and fullmove counters after the en passant field";

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
                "8/8/8/8/8/8/8/k1K5 w",
                "expected a FEN with 6 fields, found 2".to_string(),
            ),
            ("8/8/8/8/8/8/8/k1K5 w - -", COUNTERS_MISSING.to_string()),
            (
                "8/8/8/8/8/8/8/k1K5 w - - Nothing",
                COUNTERS_MISSING.to_string(),
            ),
            (
                "8/8/8/8/8/8/8/k1K5 w - - Check White",
                COUNTERS_MISSING.to_string(),
            ),
            (
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 Nothing",
                COUNTERS_MISSING.to_string(),
            ),
            (&format!("{START} Chek"), "unknown label 'Chek'".to_string()),
            (
                &format!("{START} Check  Whit"),
                "unknown label 'Check Whit'".to_string(),
            ),
            (
                "8/8/8/8/8/8/8/8 x - - 0 1 Nothing",
                "invalid FEN '8/8/8/8/8/8/8/8 x - - 0 1'".to_string(),
            ),
        ];
        for (line, message) in cases {
            assert_eq!(parse(line), Err(message), "{line:?}");
        }
    }
}
