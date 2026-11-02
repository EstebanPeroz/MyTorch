use super::{Color, Label};

#[test]
fn displays_subject_wording() {
    assert_eq!(Label::Nothing.to_string(), "Nothing");
    assert_eq!(Label::Check(Color::White).to_string(), "Check White");
    assert_eq!(
        Label::Checkmate(Color::Black).to_string(),
        "Checkmate Black"
    );
    assert_eq!(Label::Stalemate.to_string(), "Stalemate");
}

#[test]
fn text_round_trips_for_every_label() {
    for label in Label::ALL {
        assert_eq!(label.to_string().parse::<Label>(), Ok(label));
    }
}

#[test]
fn indices_cover_every_class_once() {
    for (expected, label) in Label::ALL.into_iter().enumerate() {
        assert_eq!(label.index(), expected);
        assert_eq!(Label::from_index(expected), Some(label));
    }
    assert_eq!(Label::from_index(Label::COUNT), None);
}

#[test]
fn rejects_unknown_labels() {
    for text in [
        "",
        "Check",
        "check White",
        "Checkmate Red",
        "Nothing else",
        " Nothing",
    ] {
        assert!(text.parse::<Label>().is_err(), "{text:?}");
    }
}
