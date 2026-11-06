use anyhow::{Result, anyhow};
use chess_fen_parser::{FenParser, Rule};
use pest::Parser;

#[test]
fn piece_rule() -> Result<()> {
    let inputs = ["K", "B", "q", "n", "r", "p"];
    for input in inputs {
        let res = FenParser::parse(Rule::piece, input)?
            .next()
            .ok_or(anyhow!("No pair"))?;
        assert_eq!(res.as_str(), input);
    }
    Ok(())
}

#[test]
fn active_color_rule() -> Result<()> {
    let inputs = ["b", "w"];
    for input in inputs {
        let res = FenParser::parse(Rule::active_color, input)?
            .next()
            .ok_or(anyhow!("No pair"))?;
        assert_eq!(res.as_str(), input);
    }
    Ok(())
}

#[test]
fn castling_rule() -> Result<()> {
    let inputs = ["KQkq", "KQ", "Qq", "-"];
    for input in inputs {
        let res = FenParser::parse(Rule::castling, input)?
            .next()
            .ok_or(anyhow!("No pair"))?;
        assert_eq!(res.as_str(), input);
    }
    Ok(())
}

#[test]
fn rank_digit_rule() -> Result<()> {
    let inputs = ["1", "5", "8"];
    for input in inputs {
        let res = FenParser::parse(Rule::rank_digit, input)?
            .next()
            .ok_or(anyhow!("No pair"))?;
        assert_eq!(res.as_str(), input);
    }
    Ok(())
}

#[test]
fn rank_rule() -> Result<()> {
    let inputs = ["rnbqkbnr", "PPPPPPPP", "R3K2r", "3p4", "4B3", "8"];
    for input in inputs {
        let res = FenParser::parse(Rule::rank, input)?
            .next()
            .ok_or(anyhow!("No pair"))?;
        assert_eq!(res.as_str(), input);
    }
    Ok(())
}

#[test]
fn piece_placement_rule() -> Result<()> {
    let inputs = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
        "8/8/8/3pP3/8/8/8/8",
        "r3k2r/8/8/8/8/8/8/R3K2R",
    ];
    for input in inputs {
        let res = FenParser::parse(Rule::piece_placement, input)?
            .next()
            .ok_or(anyhow!("No pair"))?;
        assert_eq!(res.as_str(), input);
    }
    Ok(())
}

#[test]
fn square_rule() -> Result<()> {
    let inputs = ["e3", "b6", "h1"];
    for input in inputs {
        let res = FenParser::parse(Rule::square, input)?
            .next()
            .ok_or(anyhow!("No pair"))?;
        assert_eq!(res.as_str(), input);
    }
    Ok(())
}

#[test]
fn en_passant_rule() -> Result<()> {
    let inputs = ["-", "e3", "a6"];
    for input in inputs {
        let res = FenParser::parse(Rule::en_passant, input)?
            .next()
            .ok_or(anyhow!("No pair"))?;
        assert_eq!(res.as_str(), input);
    }
    Ok(())
}

#[test]
fn halfmove_rule() -> Result<()> {
    let inputs = ["0", "5", "27", "100"];
    for input in inputs {
        let res = FenParser::parse(Rule::halfmove, input)?
            .next()
            .ok_or(anyhow!("No pair"))?;
        assert_eq!(res.as_str(), input);
    }
    Ok(())
}

#[test]
fn fullmove_rule() -> Result<()> {
    let inputs = ["0", "1", "35", "100"];
    for input in inputs {
        let res = FenParser::parse(Rule::fullmove, input)?
            .next()
            .ok_or(anyhow!("No pair"))?;
        assert_eq!(res.as_str(), input);
    }
    Ok(())
}

#[test]
fn fen_core_rule() -> Result<()> {
    let inputs = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R b Qk - 5 20",
        "8/8/8/3pP3/8/8/8/8 w - d6 0 10",
    ];
    for input in inputs {
        let res = FenParser::parse(Rule::fen_core, input)?
            .next()
            .ok_or(anyhow!("No pair"))?;
        assert_eq!(res.as_str(), input);
    }
    Ok(())
}

#[test]
fn event_tag_rule() -> Result<()> {
    let inputs = [
        "[Event \"World Championship\"]",
        "[Event \"Rapid Cup 2025\"]",
    ];
    for input in inputs {
        let res = FenParser::parse(Rule::event_tag, input)?
            .next()
            .ok_or(anyhow!("No pair"))?;
        assert_eq!(res.as_str(), input);
    }
    Ok(())
}

#[test]
fn white_tag_rule() -> Result<()> {
    let inputs = ["[White \"Magnus Carlsen\"]", "[White \"Hikaru Nakamura\"]"];
    for input in inputs {
        let res = FenParser::parse(Rule::white_tag, input)?
            .next()
            .ok_or(anyhow!("No pair"))?;
        assert_eq!(res.as_str(), input);
    }
    Ok(())
}

#[test]
fn black_tag_rule() -> Result<()> {
    let inputs = [
        "[Black \"Ian Nepomniachtchi\"]",
        "[Black \"Fabiano Caruana\"]",
    ];
    for input in inputs {
        let res = FenParser::parse(Rule::black_tag, input)?
            .next()
            .ok_or(anyhow!("No pair"))?;
        assert_eq!(res.as_str(), input);
    }
    Ok(())
}

#[test]
fn comment_rule() -> Result<()> {
    let inputs = [
        "{This position is from round 3}",
        "{Opening preparation line}",
        "{White has a strong center here.}",
    ];
    for input in inputs {
        let res = FenParser::parse(Rule::comment, input)?
            .next()
            .ok_or(anyhow!("No pair"))?;
        assert_eq!(res.as_str(), input);
    }
    Ok(())
}

#[test]
fn fen_record_rule() -> Result<()> {
    let inputs = [
        "[Event \"World Championship\"]\n[White \"Magnus Carlsen\"]\n[Black \"Ian Nepomniachtchi\"]\nrnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\n{Opening position}",
        "[Event \"Rapid Cup 2025\"]\nr3k2r/8/8/8/8/8/8/R3K2R b Qk - 5 20",
    ];
    for input in inputs {
        let res = FenParser::parse(Rule::fen_record, input)?
            .next()
            .ok_or(anyhow!("No pair"))?;
        assert_eq!(res.as_str(), input);
    }
    Ok(())
}
