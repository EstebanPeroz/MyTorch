pub(crate) fn find_by_name<T>(
    kind: &str,
    name: &str,
    candidates: Vec<T>,
    name_of: impl Fn(&T) -> &'static str,
) -> Result<T, String> {
    let known: Vec<&str> = candidates.iter().map(&name_of).collect();
    candidates
        .into_iter()
        .find(|candidate| name_of(candidate) == name)
        .ok_or_else(|| {
            format!(
                "unknown {kind} \"{name}\", expected one of: {}",
                known.join(", ")
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn colors() -> Vec<(&'static str, u8)> {
        vec![("red", 1), ("green", 2)]
    }

    #[test]
    fn finds_the_candidate_with_that_name() {
        assert_eq!(
            find_by_name("color", "green", colors(), |c| c.0),
            Ok(("green", 2))
        );
    }

    #[test]
    fn lists_the_known_names_when_none_matches() {
        assert_eq!(
            find_by_name("color", "blue", colors(), |c| c.0),
            Err("unknown color \"blue\", expected one of: red, green".to_string())
        );
    }
}
