use std::fs;
use std::path::Path;

use crate::sample::Sample;

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
        let content = fs::read_to_string(path)
            .map_err(|error| format!("error: cannot read '{}': {error}", path.display()))?;
        ChessFile::parse(&path.display().to_string(), &content, labels)
    }

    pub fn samples(&self) -> &[Sample] {
        &self.samples
    }

    fn parse(name: &str, content: &str, labels: Labels) -> Result<Self, String> {
        let samples = content
            .lines()
            .enumerate()
            .filter(|(_, line)| !line.trim().is_empty())
            .map(|(index, line)| {
                ChessFile::parse_line(line, labels)
                    .map_err(|message| format!("error: {name}:{}: {message}", index + 1))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if samples.is_empty() {
            return Err(format!("error: {name}: no chessboards"));
        }
        Ok(ChessFile { samples })
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

    fn parse(content: &str, labels: Labels) -> Result<ChessFile, String> {
        ChessFile::parse("boards.txt", content, labels)
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
    fn errors_point_at_the_line_in_the_file() {
        let content = format!("{START} Nothing\n\n\n{START} Chek\n");
        assert_eq!(
            parse(&content, Labels::Optional),
            Err("error: boards.txt:4: unknown label 'Chek'".to_string())
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
    }
}
