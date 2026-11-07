use my_torch::Scalar;

use crate::label::Label;

/// How often each label was predicted for each expected label: `count(expected, predicted)`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Confusion {
    counts: [[usize; Label::COUNT]; Label::COUNT],
}

impl Confusion {
    pub fn record(&mut self, expected: Label, predicted: Label) {
        self.counts[expected.index()][predicted.index()] += 1;
    }

    pub fn count(&self, expected: Label, predicted: Label) -> usize {
        self.counts[expected.index()][predicted.index()]
    }

    pub fn counts(&self) -> &[[usize; Label::COUNT]; Label::COUNT] {
        &self.counts
    }

    pub fn total(&self) -> usize {
        self.counts.iter().flatten().sum()
    }

    /// The share of predictions that are right, between 0 and 1; 0 when nothing was recorded.
    pub fn accuracy(&self) -> Scalar {
        let correct: usize = (0..Label::COUNT)
            .map(|index| self.counts[index][index])
            .sum();
        ratio(correct, self.total())
    }

    /// How many samples were expected to have this label.
    pub fn support(&self, label: Label) -> usize {
        self.counts[label.index()].iter().sum()
    }

    /// The accuracy of always predicting the most frequent label, which a network has to beat.
    pub fn baseline(&self) -> Scalar {
        let most = Label::ALL
            .into_iter()
            .map(|label| self.support(label))
            .max()
            .unwrap_or(0);
        ratio(most, self.total())
    }
}

fn ratio(part: usize, whole: usize) -> Scalar {
    if whole == 0 {
        0.0
    } else {
        part as Scalar / whole as Scalar
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Color;

    const CHECK: Label = Label::Check(Color::White);

    #[test]
    fn counts_each_pair_of_labels() {
        let mut confusion = Confusion::default();
        confusion.record(Label::Nothing, Label::Nothing);
        confusion.record(Label::Nothing, CHECK);
        confusion.record(Label::Nothing, CHECK);
        confusion.record(CHECK, CHECK);
        assert_eq!(confusion.count(Label::Nothing, CHECK), 2);
        assert_eq!(confusion.count(CHECK, Label::Nothing), 0);
        assert_eq!(confusion.total(), 4);
        assert_eq!(confusion.support(Label::Nothing), 3);
        assert_eq!(confusion.support(Label::Stalemate), 0);
        assert_eq!(confusion.counts()[Label::Nothing.index()][CHECK.index()], 2);
    }

    #[test]
    fn accuracy_counts_the_diagonal() {
        let mut confusion = Confusion::default();
        assert_eq!(confusion.accuracy(), 0.0);
        confusion.record(Label::Nothing, Label::Nothing);
        confusion.record(CHECK, CHECK);
        confusion.record(CHECK, Label::Nothing);
        confusion.record(Label::Stalemate, CHECK);
        assert_eq!(confusion.accuracy(), 0.5);
    }

    #[test]
    fn the_baseline_is_the_share_of_the_most_frequent_label() {
        let mut confusion = Confusion::default();
        assert_eq!(confusion.baseline(), 0.0);
        for _ in 0..3 {
            confusion.record(Label::Nothing, CHECK);
        }
        confusion.record(CHECK, CHECK);
        assert_eq!(confusion.baseline(), 0.75);
    }
}
