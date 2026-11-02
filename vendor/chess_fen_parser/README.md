# chess_fen_parser

A library to parse [Forsyth–Edwards Notation](https://www.chess.com/terms/fen-chess) (FEN) strings for chess positions.
The library provides a parser that converts a FEN string into a structured Fen value to present the parsed information in an expanded form (piece lists for each side, castling rights, target on passage, move counters).

Crates.io: https://crates.io/crates/chess_fen_parser

Docs: https://docs.rs/chess_fen_parser

## Description
This project uses the pest parser and a custom grammar in src/fen.pest to parse complete FEN strings according to the standard field structure. The main parsing logic is implemented in src/lib.rs.

1. The input string is first parsed by Pest using the fen rule defined in the grammar. The parse_fen function takes a &str, runs the parser, and extracts the six required fields in order: piece placement, active color, castling rights, en-passant square, halfmove counter, and fullmove number.

2. Piece placement is handled rank by rank. Each rank is expanded using expand_rank, which reads piece tokens or digits indicating empty squares. Piece tokens are converted into Piece structures with a type and color, and each rank becomes a Vec<Option<Piece>> of length 8. The function checks that exactly 8 ranks are produced.

3. Castling rights are converted into a Castlings struct with boolean fields for each allowed castling type. A dash means no castling rights. The en-passant field is stored as Option<String>.

4. The halfmove and fullmove fields are parsed into integers (u16).

## Produced data structures
- Fen (pub):
  - board: Vec<Vec<Option<Piece>>> (8 ranks, each rank is 8 files, Option<Piece> for empty squares)
  - active_color: Color (White | Black)
  - castling: Castlings { white_short/white_long/black_short/black_long }
  - en_passant: Option<String>
  - halfmove_clock: u16
  - fullmove_number: u16

- Piece: { kind: PieceKind, color: Color }
  - PieceKind: Pawn, Rook, Knight, Bishop, Queen, King

## Grammar (src/fen.pest)
```pest
piece = { "p" | "r" | "n" | "b" | "q" | "k" | "P" | "R" | "N" | "B" | "Q" | "K" }

active_color = { "w" | "b" }

castling = { "-" | ("K" | "Q" | "k" | "q")+ }

rank_digit = { '1'..'8' }

rank = { ( piece | rank_digit )+ }

piece_placement = { rank ~ ("/" ~ rank){7} }

square = { 'a'..'h' ~ rank_digit }

en_passant = { "-" | square }

halfmove = @{ ASCII_DIGIT+ }

fullmove = @{ ASCII_DIGIT+ }

event_tag = { "[Event" ~ " \"" ~ (!"\"" ~ ANY)* ~ "\"]" }

white_tag = { "[White" ~ " \"" ~ (!"\"" ~ ANY)* ~ "\"]" }

black_tag = { "[Black" ~ " \"" ~ (!"\"" ~ ANY)* ~ "\"]" }

comment = { "{" ~ (!"}" ~ ANY)* ~ "}" }

fen_core = {
    piece_placement ~ " "
    ~ active_color ~ " "
    ~ castling ~ " "
    ~ en_passant ~ " "
    ~ halfmove ~ " "
    ~ fullmove
}

fen_record = {
    SOI
    ~ event_tag? ~ NEWLINE*
    ~ white_tag? ~ NEWLINE*
    ~ black_tag? ~ NEWLINE*
    ~ fen_core
    ~ (NEWLINE* ~ comment)?
    ~ EOI
}
```

## Usage
- Library: call parse_fen("<fen string>") to obtain a Fen value.

- CLI:
  - cargo run -- parse example_fen.txt

## Examples
- The starting position:
  rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1

- A middlegame example:
  r1bq1rk1/ppp2ppp/2n2n2/2bp4/4P3/2N2N2/PPP2PPP/R1BQ1RK1 b - - 4 9

- An en passant example:
  8/8/8/3pP3/8/8/8/8 b - e3 0 1

- FEN with tags and comment:
  [Event "Casual Game"]
  [White "Alice"]
  [Black "Bob"]
  rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
  {This is the starting position}

## How parsed results can be used
The parsed FEN is returned as a Fen object that can be displayed, serialized, or processed.
It provides structured board data for UIs, logging, ASCII or SVG output, and conversion to other formats like JSON or binary.
Engines and move generators can use it directly for piece placement, turn, and other data.
