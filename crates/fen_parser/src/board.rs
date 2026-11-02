use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Color {
    White,
    Black,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PieceType {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Piece {
    pub kind: PieceType,
    pub color: Color,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CastlingRights {
    pub white_king_side: bool,
    pub white_queen_side: bool,
    pub black_king_side: bool,
    pub black_queen_side: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Board {
    pub squares: [Option<Piece>; 64],
    pub side_to_move: Color,
    pub castling: CastlingRights,
    pub en_passant: Option<u8>,
    pub halfmove_clock: u32,
    pub fullmove_number: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FenError {
    FieldCount(usize),
    RankCount(usize),
    RankLength(usize),
    InvalidPiece(char),
    InvalidSideToMove(String),
    InvalidCastling(String),
    InvalidEnPassant(String),
    InvalidNumber(String),
}

impl FromStr for Board {
    type Err = FenError;

    fn from_str(_fen: &str) -> Result<Self, Self::Err> {
        todo!()
    }
}
