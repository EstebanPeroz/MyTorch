use std::collections::{HashMap, HashSet};

use my_torch::{Dataset, Matrix, Network, Scalar};

use crate::board::{Board, PieceKind};
use crate::color::Color;
use crate::label::Label;
use crate::sample::Sample;

const KINDS: usize = 6;
const PLANES: usize = KINDS * 2;
const SIDE_TO_MOVE: usize = PLANES * Board::SQUARES;
const PREDICTION_BATCH: usize = 4096;

pub const INPUTS: usize = SIDE_TO_MOVE + 1;
pub const OUTPUTS: usize = Label::COUNT;

fn plane(kind: PieceKind, color: Color) -> usize {
    let kind = match kind {
        PieceKind::Pawn => 0,
        PieceKind::Knight => 1,
        PieceKind::Bishop => 2,
        PieceKind::Rook => 3,
        PieceKind::Queen => 4,
        PieceKind::King => 5,
    };
    match color {
        Color::White => kind,
        Color::Black => KINDS + kind,
    }
}

pub fn encode(board: &Board) -> [Scalar; INPUTS] {
    let mut inputs = [0.0; INPUTS];
    for (square, piece) in board.squares().iter().enumerate() {
        if let Some(piece) = piece {
            inputs[plane(piece.kind(), piece.color()) * Board::SQUARES + square] = 1.0;
        }
    }
    if board.side_to_move() == Color::White {
        inputs[SIDE_TO_MOVE] = 1.0;
    }
    inputs
}

pub fn inputs(samples: &[Sample]) -> Matrix {
    let data = samples
        .iter()
        .flat_map(|sample| encode(sample.board()))
        .collect();
    Matrix::from_vec(samples.len(), INPUTS, data).expect("every board is encoded to INPUTS values")
}

pub fn dataset(samples: &[Sample]) -> Result<Dataset, String> {
    let mut targets = vec![0.0; samples.len() * OUTPUTS];
    for (row, sample) in samples.iter().enumerate() {
        let label = sample.label().ok_or("a training sample has no label")?;
        targets[row * OUTPUTS + label.index()] = 1.0;
    }
    Dataset::new(
        inputs(samples),
        Matrix::from_vec(samples.len(), OUTPUTS, targets)?,
    )
}

pub fn unique(samples: Vec<Sample>) -> Vec<Sample> {
    let mut seen = HashSet::new();
    samples
        .into_iter()
        .filter(|sample| seen.insert(sample.board().clone()))
        .collect()
}

/// Counts the chessboards that appear with more than one label. `unique` keeps only the first label of each
/// of them, so the result of a training depends on the order of the file.
pub fn conflicts(samples: &[Sample]) -> usize {
    let mut labels: HashMap<&Board, Option<Label>> = HashMap::new();
    let mut conflicting = HashSet::new();
    for sample in samples {
        let first = labels.entry(sample.board()).or_insert(sample.label());
        if *first != sample.label() {
            conflicting.insert(sample.board());
        }
    }
    conflicting.len()
}

pub fn check_shape(network: &Network) -> Result<(), String> {
    if network.input_size() != INPUTS || network.output_size() != OUTPUTS {
        return Err(format!(
            "the network has {} inputs and {} outputs, but a chessboard needs {INPUTS} and {OUTPUTS}",
            network.input_size(),
            network.output_size()
        ));
    }
    Ok(())
}

/// # Panics
/// Panics if `scores` contains fewer than `OUTPUTS` values.
pub fn classify(scores: &[Scalar], board: &Board) -> Label {
    let attacker = board.side_to_move().opposite();
    Label::ALL
        .into_iter()
        .filter(|label| match label {
            Label::Check(color) | Label::Checkmate(color) => *color == attacker,
            Label::Nothing | Label::Stalemate => true,
        })
        .max_by(|a, b| scores[a.index()].total_cmp(&scores[b.index()]))
        .expect("Nothing is always possible")
}

fn check_finite(scores: &Matrix) -> Result<(), String> {
    if scores.data().iter().any(|score| !score.is_finite()) {
        return Err("the network produced a score that is not finite".into());
    }
    Ok(())
}

pub fn predict(network: &Network, samples: &[Sample]) -> Result<Vec<Label>, String> {
    if !samples.is_empty() {
        check_shape(network)?;
    }
    let mut labels = Vec::with_capacity(samples.len());
    for chunk in samples.chunks(PREDICTION_BATCH) {
        let scores = network.predict(&inputs(chunk))?;
        check_finite(&scores)?;
        labels.extend(
            chunk
                .iter()
                .enumerate()
                .map(|(row, sample)| classify(scores.row(row), sample.board())),
        );
    }
    Ok(labels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fens::{FOOLS_MATE, START};

    fn board(fen: &str) -> Board {
        Board::from_fen(fen).unwrap()
    }

    #[test]
    fn marks_each_piece_in_its_plane_and_square() {
        let inputs = encode(&board(START));
        assert_eq!(inputs.iter().filter(|&&value| value == 1.0).count(), 33);
        // White pawn on a2: plane 0, square index 8.
        assert_eq!(inputs[8], 1.0);
        // Black king on e8: plane 11, square index 60.
        assert_eq!(inputs[11 * Board::SQUARES + 60], 1.0);
        assert_eq!(inputs[SIDE_TO_MOVE], 1.0);
    }

    #[test]
    fn tells_who_is_to_move() {
        assert_eq!(encode(&board(START))[SIDE_TO_MOVE], 1.0);
        let black = "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1";
        assert_eq!(encode(&board(black))[SIDE_TO_MOVE], 0.0);
    }

    #[test]
    fn targets_are_one_hot() {
        let samples: Vec<Sample> = [
            format!("{FOOLS_MATE} Checkmate Black"),
            format!("{START} Nothing"),
        ]
        .iter()
        .map(|line| line.parse().unwrap())
        .collect();
        let data = dataset(&samples).unwrap();
        assert_eq!(data.sample_count(), 2);
        assert_eq!(data.targets().row(0), [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        assert_eq!(data.targets().row(1), [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn unlabeled_samples_cannot_make_a_dataset() {
        let sample: Sample = START.parse().unwrap();
        assert!(dataset(&[sample]).is_err());
    }

    #[test]
    fn keeps_the_first_copy_of_each_board() {
        let lines = [
            format!("{START} Nothing"),
            format!("{FOOLS_MATE} Checkmate Black"),
            format!("{START} Nothing"),
        ];
        let samples = lines.iter().map(|line| line.parse().unwrap()).collect();
        let kept = unique(samples);
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0].board(), &board(START));
    }

    #[test]
    fn counts_the_boards_with_conflicting_labels() {
        let lines = [
            format!("{START} Nothing"),
            format!("{FOOLS_MATE} Checkmate Black"),
            format!("{START} Stalemate"),
            format!("{START} Nothing"),
            format!("{FOOLS_MATE} Checkmate Black"),
        ];
        let samples: Vec<Sample> = lines.iter().map(|line| line.parse().unwrap()).collect();
        assert_eq!(conflicts(&samples), 1);
        assert_eq!(conflicts(&samples[..2]), 0);
    }

    #[test]
    fn rejects_scores_that_are_not_finite() {
        let finite = Matrix::from_rows(&[[0.0, 1.0]]).unwrap();
        assert!(check_finite(&finite).is_ok());
        for bad in [Scalar::NAN, Scalar::INFINITY, Scalar::NEG_INFINITY] {
            assert!(check_finite(&Matrix::from_rows(&[[0.0, bad]]).unwrap()).is_err());
        }
    }

    #[test]
    fn only_the_attacking_color_can_be_predicted() {
        let mut scores = [0.0; OUTPUTS];
        scores[Label::Check(Color::White).index()] = 5.0;
        scores[Label::Check(Color::Black).index()] = 4.0;
        // White to move: only Black can be giving check.
        assert_eq!(classify(&scores, &board(START)), Label::Check(Color::Black));
        let black = board("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1");
        assert_eq!(classify(&scores, &black), Label::Check(Color::White));
    }

    #[test]
    fn picks_the_strongest_output() {
        let mut scores = [0.0; OUTPUTS];
        scores[Label::Stalemate.index()] = 1.0;
        assert_eq!(classify(&scores, &board(START)), Label::Stalemate);
    }
}
