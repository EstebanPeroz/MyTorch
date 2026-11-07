use std::fs::File;
use std::io::{self, BufRead, BufReader, ErrorKind};
use std::path::Path;

use rayon::prelude::*;

use crate::sample::Sample;

const BYTE_ORDER_MARK: char = '\u{feff}';

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Labels {
    Required,
    Optional,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChessFile {
    samples: Vec<Sample>,
}

impl ChessFile {
    pub fn read(path: &Path, labels: Labels) -> Result<Self, String> {
        let name = path.display().to_string();
        let file = File::open(path).map_err(|error| ChessFile::unreadable(&name, &error))?;
        ChessFile::parse(&name, BufReader::new(file), labels)
    }

    pub fn samples(&self) -> &[Sample] {
        &self.samples
    }

    fn parse(name: &str, reader: impl BufRead, labels: Labels) -> Result<Self, String> {
        let located =
            |index: usize, message: String| format!("error: {name}:{}: {message}", index + 1);
        // Reading lines is cheap, parsing them is not: read everything up to
        // the first unreadable line, parse the lines in parallel, then report
        // whichever problem comes first in the file.
        let mut lines = Vec::new();
        let mut unreadable = None;
        for (index, line) in reader.lines().enumerate() {
            match line {
                Ok(line) => lines.push((index, line)),
                Err(error) => {
                    unreadable = Some((index, error));
                    break;
                }
            }
        }
        let parsed: Vec<Result<Sample, (usize, String)>> = lines
            .par_iter()
            .filter_map(|(index, line)| {
                let line = match index {
                    0 => line.strip_prefix(BYTE_ORDER_MARK).unwrap_or(line),
                    _ => line,
                };
                (!line.trim().is_empty()).then(|| {
                    ChessFile::parse_line(line, labels).map_err(|message| (*index, message))
                })
            })
            .collect();
        let mut samples = Vec::with_capacity(parsed.len());
        for sample in parsed {
            samples.push(sample.map_err(|(index, message)| located(index, message))?);
        }
        if let Some((index, error)) = unreadable {
            return Err(match error.kind() {
                ErrorKind::InvalidData => located(index, error.to_string()),
                _ => ChessFile::unreadable(name, &error),
            });
        }
        if samples.is_empty() {
            return Err(format!("error: {name}: no chessboards"));
        }
        Ok(ChessFile { samples })
    }

    fn unreadable(name: &str, error: &io::Error) -> String {
        format!("error: cannot read '{name}': {error}")
    }

    fn parse_line(line: &str, labels: Labels) -> Result<Sample, String> {
        let sample: Sample = line.parse()?;
        if labels == Labels::Required && sample.label().is_none() {
            return Err("missing label, required for training".into());
        }
        Ok(sample)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Color;
    use crate::label::Label;
    use crate::test_fens::{FOOLS_MATE, START};

    fn parse(content: impl AsRef<[u8]>, labels: Labels) -> Result<ChessFile, String> {
        ChessFile::parse("boards.txt", content.as_ref(), labels)
    }

    fn labels_of(file: &ChessFile) -> Vec<Option<Label>> {
        file.samples().iter().map(Sample::label).collect()
    }

    #[test]
    fn reads_every_non_blank_line_in_order() {
        let content = format!("{FOOLS_MATE} Checkmate Black\n\n   \n{START} Nothing\r\n{START}\n");
        let file = parse(&content, Labels::Optional).unwrap();
        assert_eq!(
            labels_of(&file),
            [
                Some(Label::Checkmate(Color::Black)),
                Some(Label::Nothing),
                None
            ]
        );
    }

    #[test]
    fn errors_point_at_the_first_bad_line() {
        let content = format!("{START} Nothing\n\n\n{START} Chek\n{START} Nope\n");
        assert_eq!(
            parse(&content, Labels::Optional),
            Err("error: boards.txt:4: unknown label 'Chek'".to_string())
        );
    }

    #[test]
    fn ignores_a_byte_order_mark_at_the_start() {
        let content = format!("{BYTE_ORDER_MARK}{START} Nothing\n{START} Check White\n");
        let file = parse(&content, Labels::Required).unwrap();
        assert_eq!(
            labels_of(&file),
            [Some(Label::Nothing), Some(Label::Check(Color::White))]
        );
        let content = format!("{START} Nothing\n{BYTE_ORDER_MARK}{START} Nothing\n");
        assert!(parse(&content, Labels::Optional).is_err());
    }

    #[test]
    fn a_bad_line_is_reported_before_a_later_unreadable_one() {
        let content = [
            format!("{START} Nothing\n{START} Chek\n").as_bytes(),
            b"\xa0\n",
        ]
        .concat();
        assert_eq!(
            parse(content, Labels::Optional),
            Err("error: boards.txt:2: unknown label 'Chek'".to_string())
        );
    }

    #[test]
    fn invalid_utf8_points_at_its_line() {
        let content = [format!("{START} Nothing\n").as_bytes(), b"\xa0\n"].concat();
        assert_eq!(
            parse(content, Labels::Optional),
            Err("error: boards.txt:2: stream did not contain valid UTF-8".to_string())
        );
    }

    #[test]
    fn labels_are_required_only_when_asked() {
        let content = format!("{START} Nothing\n{START}\n");
        assert!(parse(&content, Labels::Optional).is_ok());
        assert_eq!(
            parse(&content, Labels::Required),
            Err("error: boards.txt:2: missing label, required for training".to_string())
        );
    }

    #[test]
    fn rejects_files_without_chessboards() {
        for content in ["", "\n\n", "  \n\t\n"] {
            assert_eq!(
                parse(content, Labels::Optional),
                Err("error: boards.txt: no chessboards".to_string()),
                "{content:?}"
            );
        }
    }

    #[test]
    fn reports_unreadable_files() {
        let error = ChessFile::read(Path::new("does/not/exist.txt"), Labels::Optional).unwrap_err();
        assert!(
            error.starts_with("error: cannot read 'does/not/exist.txt': "),
            "{error}"
        );
        let error = ChessFile::read(Path::new("."), Labels::Optional).unwrap_err();
        assert!(error.starts_with("error: cannot read '.': "), "{error}");
    }
}
