#![doc = include_str!("../README.md")]
use pest::Parser as PestParser;
use pest::iterators::Pair;
use pest_derive::Parser;
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Parser)]
#[grammar = "fen.pest"]
pub struct FenParser;

#[derive(Debug)]
pub enum Color {
    White,
    Black,
}

#[derive(Debug)]
pub enum PieceKind {
    Pawn,
    Rook,
    Knight,
    Bishop,
    Queen,
    King,
}

#[derive(Debug)]
pub struct Piece {
    pub kind: PieceKind,
    pub color: Color,
}

#[derive(Debug)]
pub struct Castlings {
    pub white_short: bool,
    pub white_long: bool,
    pub black_short: bool,
    pub black_long: bool,
}

#[derive(Debug)]
pub struct FenRecord {
    pub event: Option<String>,
    pub white: Option<String>,
    pub black: Option<String>,
    pub comment: Option<String>,
    pub board: Vec<Vec<Option<Piece>>>,
    pub active_color: Color,
    pub castling: Castlings,
    pub en_passant: Option<String>,
    pub halfmove_clock: u16,
    pub fullmove_number: u16,
}

#[derive(Error, Debug)]
pub enum FenError {
    #[error("parse error: {0}")]
    ParseError(#[from] Box<pest::error::Error<Rule>>),
    #[error("invalid rank length: {0}")]
    InvalidRankLength(usize),
    #[error("invalid rank count: {0}")]
    InvalidRankCount(usize),
    #[error("invalid castling rights: {0}")]
    InvalidCastling(String),
    #[error("invalid number: {0}")]
    InvalidNumber(#[from] std::num::ParseIntError),
}

impl From<pest::error::Error<Rule>> for FenError {
    fn from(e: pest::error::Error<Rule>) -> Self {
        FenError::ParseError(Box::new(e))
    }
}

fn parse_piece(pair: Pair<Rule>) -> Result<Piece, FenError> {
    let ch = pair.as_str().chars().next().unwrap();
    let color = if ch.is_uppercase() {
        Color::White
    } else {
        Color::Black
    };
    let kind = match ch.to_ascii_lowercase() {
        'p' => PieceKind::Pawn,
        'r' => PieceKind::Rook,
        'n' => PieceKind::Knight,
        'b' => PieceKind::Bishop,
        'q' => PieceKind::Queen,
        _ => PieceKind::King,
    };
    Ok(Piece { kind, color })
}

fn expand_rank(pair: Pair<Rule>) -> Result<Vec<Option<Piece>>, FenError> {
    let mut squares = Vec::with_capacity(8);
    for p in pair.into_inner() {
        match p.as_rule() {
            Rule::piece => squares.push(Some(parse_piece(p)?)),
            Rule::rank_digit => {
                let cnt = p.as_str().parse::<usize>()?;
                for _ in 0..cnt {
                    squares.push(None);
                }
            }
            _ => {}
        }
    }
    if squares.len() != 8 {
        return Err(FenError::InvalidRankLength(squares.len()));
    }
    Ok(squares)
}

fn parse_castling(s: &str) -> Result<Castlings, FenError> {
    let mut rights = Castlings {
        white_short: false,
        white_long: false,
        black_short: false,
        black_long: false,
    };
    if s == "-" {
        return Ok(rights);
    }
    for ch in s.chars() {
        match ch {
            'K' => rights.white_short = true,
            'Q' => rights.white_long = true,
            'k' => rights.black_short = true,
            'q' => rights.black_long = true,
            _ => return Err(FenError::InvalidCastling(s.to_string())),
        }
    }
    Ok(rights)
}

pub fn parse_fen(input: &str) -> Result<FenRecord, FenError> {
    let mut pairs = FenParser::parse(Rule::fen_record, input)?;
    let record_pair = pairs.next().unwrap();

    let mut event = None;
    let mut white = None;
    let mut black = None;
    let mut comment = None;

    let mut board = Vec::with_capacity(8);
    let mut active_color = Color::White;
    let mut castling = Castlings {
        white_short: false,
        white_long: false,
        black_short: false,
        black_long: false,
    };
    let mut en_passant = None;
    let mut halfmove_clock = 0;
    let mut fullmove_number = 1;

    for pair in record_pair.into_inner() {
        match pair.as_rule() {
            Rule::event_tag => event = Some(extract_tag_value(pair.as_str())),
            Rule::white_tag => white = Some(extract_tag_value(pair.as_str())),
            Rule::black_tag => black = Some(extract_tag_value(pair.as_str())),
            Rule::comment => {
                let c = pair.as_str();
                comment = Some(c.trim_matches(|c| c == '{' || c == '}').trim().to_string());
            }
            Rule::fen_core => {
                let mut inner = pair.into_inner();
                let piece_placement_pair = inner.next().unwrap();
                active_color = match inner.next().unwrap().as_str() {
                    "w" => Color::White,
                    _ => Color::Black,
                };
                castling = parse_castling(inner.next().unwrap().as_str())?;
                en_passant = {
                    let ep = inner.next().unwrap().as_str();
                    if ep == "-" {
                        None
                    } else {
                        Some(ep.to_string())
                    }
                };
                halfmove_clock = inner.next().unwrap().as_str().parse::<u16>()?;
                fullmove_number = inner.next().unwrap().as_str().parse::<u16>()?;
                for rank_pair in piece_placement_pair.into_inner() {
                    board.push(expand_rank(rank_pair)?);
                }
                if board.len() != 8 {
                    return Err(FenError::InvalidRankCount(board.len()));
                }
            }
            _ => {}
        }
    }

    Ok(FenRecord {
        event,
        white,
        black,
        comment,
        board,
        active_color,
        castling,
        en_passant,
        halfmove_clock,
        fullmove_number,
    })
}

fn extract_tag_value(s: &str) -> String {
    s.split_once('\"')
        .map(|x| x.1)
        .and_then(|rest| rest.split('\"').next())
        .unwrap_or("")
        .trim()
        .to_string()
}

pub fn coord_from_rf(r: usize, f: usize) -> String {
    let file = (b'a' + (f as u8)) as char;
    let rank = 8 - r;
    format!("{}{}", file, rank)
}

pub fn collect_positions(
    fen: &FenRecord,
) -> (BTreeMap<String, Vec<String>>, BTreeMap<String, Vec<String>>) {
    let mut white: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut black: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for r in 0..8 {
        for f in 0..8 {
            if let Some(p) = &fen.board[r][f] {
                let name = match p.kind {
                    PieceKind::Pawn => "pawn",
                    PieceKind::Rook => "rook",
                    PieceKind::Knight => "knight",
                    PieceKind::Bishop => "bishop",
                    PieceKind::Queen => "queen",
                    PieceKind::King => "king",
                };
                let coord = coord_from_rf(r, f);
                match p.color {
                    Color::White => white.entry(name.to_string()).or_default().push(coord),
                    Color::Black => black.entry(name.to_string()).or_default().push(coord),
                }
            }
        }
    }
    for v in white.values_mut() {
        v.sort();
    }
    for v in black.values_mut() {
        v.sort();
    }
    (white, black)
}

pub fn castling_string(c: &Castlings) -> String {
    let mut s = String::new();
    if c.white_short {
        s.push('K');
    }
    if c.white_long {
        s.push('Q');
    }
    if c.black_short {
        s.push('k');
    }
    if c.black_long {
        s.push('q');
    }
    if s.is_empty() {
        s.push('-');
    }
    s
}

pub fn castling_for_side(c: &Castlings, color: Color) -> String {
    let mut parts: Vec<&str> = Vec::new();
    match color {
        Color::White => {
            if c.white_long {
                parts.push("long");
            }
            if c.white_short {
                parts.push("short");
            }
        }
        Color::Black => {
            if c.black_long {
                parts.push("long");
            }
            if c.black_short {
                parts.push("short");
            }
        }
    }
    if parts.is_empty() {
        "-".to_string()
    } else {
        parts.join(", ")
    }
}

pub fn print_parsing_results(fen: &FenRecord) {
    let divider = "------------------------------------------";
    println!("\n{}", divider);
    println!("CHESS GAME INFO");
    println!("{}", divider);
    if let Some(e) = &fen.event {
        println!("Event: {}", e);
    }
    if let Some(w) = &fen.white {
        println!("White: {}", w);
    }
    if let Some(b) = &fen.black {
        println!("Black: {}", b);
    }
    println!("{}", divider);
    let (white, black) = collect_positions(fen);
    print_side("White:", &white, &fen.castling, Color::White, divider);
    print_side("Black:", &black, &fen.castling, Color::Black, divider);
    println!(
        "Active color: {}",
        match fen.active_color {
            Color::White => "White",
            Color::Black => "Black",
        }
    );
    println!(
        "En passant: {}",
        fen.en_passant.clone().unwrap_or_else(|| "-".to_string())
    );
    println!("Halfmove clock: {}", fen.halfmove_clock);
    println!("Fullmove number: {}", fen.fullmove_number);
    if let Some(c) = &fen.comment {
        println!("Comment: {}", c);
    }
    println!("{}\n", divider);
}

fn print_side(
    header: &str,
    map: &BTreeMap<String, Vec<String>>,
    castlings: &Castlings,
    color: Color,
    divider: &str,
) {
    println!("{}", header);
    for piece in ["pawn", "rook", "knight", "bishop", "queen", "king"] {
        if let Some(vec) = map.get(piece) {
            println!("    {}: {}", piece, vec.join(", "));
        }
    }
    println!("    castling: {}", castling_for_side(castlings, color));
    println!("{}", divider);
}
