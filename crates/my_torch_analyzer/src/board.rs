use chess_fen_parser::{FenError, FenRecord};

use crate::color::Color;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PieceKind {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Piece {
    kind: PieceKind,
    color: Color,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Board {
    squares: [Option<Piece>; Board::SQUARES],
    side_to_move: Color,
}

impl Piece {
    pub fn new(kind: PieceKind, color: Color) -> Self {
        Piece { kind, color }
    }

    pub fn kind(self) -> PieceKind {
        self.kind
    }

    pub fn color(self) -> Color {
        self.color
    }
}

impl Board {
    pub const SIZE: usize = 8;
    pub const SQUARES: usize = Board::SIZE * Board::SIZE;

    pub fn from_fen(fen: &str) -> Result<Self, String> {
        chess_fen_parser::parse_fen(fen)
            .map(|record| Board::from_record(&record))
            .map_err(|error| Board::describe(fen, error))
    }

    pub const fn square_index(file: usize, rank: usize) -> usize {
        rank * Board::SIZE + file
    }

    pub fn squares(&self) -> &[Option<Piece>; Board::SQUARES] {
        &self.squares
    }

    pub fn side_to_move(&self) -> Color {
        self.side_to_move
    }

    fn from_record(record: &FenRecord) -> Self {
        let mut squares = [None; Board::SQUARES];
        for (row, rank) in record.board.iter().enumerate() {
            for (file, square) in rank.iter().enumerate() {
                squares[Board::square_index(file, Board::SIZE - 1 - row)] =
                    square.as_ref().map(Piece::from);
            }
        }
        Board {
            squares,
            side_to_move: Color::from(&record.active_color),
        }
    }

    fn describe(fen: &str, error: FenError) -> String {
        match error {
            FenError::ParseError(_) => format!("invalid FEN '{fen}'"),
            error => format!("invalid FEN '{fen}': {error}"),
        }
    }
}

impl From<&chess_fen_parser::Piece> for Piece {
    fn from(piece: &chess_fen_parser::Piece) -> Self {
        let kind = match piece.kind {
            chess_fen_parser::PieceKind::Pawn => PieceKind::Pawn,
            chess_fen_parser::PieceKind::Knight => PieceKind::Knight,
            chess_fen_parser::PieceKind::Bishop => PieceKind::Bishop,
            chess_fen_parser::PieceKind::Rook => PieceKind::Rook,
            chess_fen_parser::PieceKind::Queen => PieceKind::Queen,
            chess_fen_parser::PieceKind::King => PieceKind::King,
        };
        Piece::new(kind, Color::from(&piece.color))
    }
}

impl From<&chess_fen_parser::Color> for Color {
    fn from(color: &chess_fen_parser::Color) -> Self {
        match color {
            chess_fen_parser::Color::White => Color::White,
            chess_fen_parser::Color::Black => Color::Black,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fens::{FOOLS_MATE, START};

    fn piece_at(board: &Board, square: &str) -> Option<Piece> {
        let [file, rank] = square.as_bytes() else {
            panic!("invalid square name '{square}'");
        };
        board.squares()[Board::square_index(usize::from(file - b'a'), usize::from(rank - b'1'))]
    }

    #[test]
    fn places_every_piece_of_the_start_position() {
        let board = Board::from_fen(START).unwrap();
        let back_rank = [
            PieceKind::Rook,
            PieceKind::Knight,
            PieceKind::Bishop,
            PieceKind::Queen,
            PieceKind::King,
            PieceKind::Bishop,
            PieceKind::Knight,
            PieceKind::Rook,
        ];
        for (file, kind) in back_rank.into_iter().enumerate() {
            let expected = [
                Some(Piece::new(kind, Color::White)),
                Some(Piece::new(PieceKind::Pawn, Color::White)),
                None,
                None,
                None,
                None,
                Some(Piece::new(PieceKind::Pawn, Color::Black)),
                Some(Piece::new(kind, Color::Black)),
            ];
            for (rank, piece) in expected.into_iter().enumerate() {
                let square = board.squares()[Board::square_index(file, rank)];
                assert_eq!(square, piece, "file {file}, rank {rank}");
            }
        }
        assert_eq!(Board::square_index(0, 0), 0);
        assert_eq!(Board::square_index(7, 7), Board::SQUARES - 1);
    }

    #[test]
    fn reads_a_mid_game_position() {
        let board = Board::from_fen(FOOLS_MATE).unwrap();
        assert_eq!(
            piece_at(&board, "h4"),
            Some(Piece::new(PieceKind::Queen, Color::Black))
        );
        assert_eq!(piece_at(&board, "d8"), None);
        assert_eq!(board.side_to_move(), Color::White);
    }

    #[test]
    fn reads_the_side_to_move() {
        let board = Board::from_fen("8/8/8/8/8/8/8/k1K5 b - - 0 1").unwrap();
        assert_eq!(board.side_to_move(), Color::Black);
    }

    #[test]
    fn rejects_malformed_fens_with_one_line_messages() {
        let cases = [
            (
                "rnbqkbnrr/8/8/8/8/8/8/8 w - - 0 1",
                "invalid FEN 'rnbqkbnrr/8/8/8/8/8/8/8 w - - 0 1': invalid rank length: 9",
            ),
            (
                "8/8/8/8/8/8/PPPPPPPPP/8 w - - 0 1",
                "invalid FEN '8/8/8/8/8/8/PPPPPPPPP/8 w - - 0 1': invalid rank length: 9",
            ),
            (
                "7/8/8/8/8/8/8/8 w - - 0 1",
                "invalid FEN '7/8/8/8/8/8/8/8 w - - 0 1': invalid rank length: 7",
            ),
            (
                "8/8/8/8/8/8/8/8 w - é 0 1",
                "invalid FEN '8/8/8/8/8/8/8/8 w - é 0 1'",
            ),
            (
                "8/8/8/8/8/8/8 w - - 0 1",
                "invalid FEN '8/8/8/8/8/8/8 w - - 0 1'",
            ),
            (
                "8/8/8/8/8/8/8/8 x - - 0 1",
                "invalid FEN '8/8/8/8/8/8/8/8 x - - 0 1'",
            ),
            (
                "8/8/8/8/8/8/8/8 w - - 0 70000",
                "invalid FEN '8/8/8/8/8/8/8/8 w - - 0 70000': invalid number: number too large to fit in target type",
            ),
        ];
        for (fen, message) in cases {
            assert_eq!(Board::from_fen(fen), Err(message.to_string()), "{fen}");
        }
    }
}
