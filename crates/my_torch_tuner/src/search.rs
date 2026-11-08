use std::cmp::Ordering;
use std::time::Instant;

use rand::SeedableRng;
use rand::rngs::StdRng;
use serde_json::{Value, json};

use crate::space::{Params, SearchSpace};

/// How well a trial did on the validation split.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Score {
    pub accuracy: f64,
    pub loss: f64,
}

impl Score {
    /// Orders from the worst to the best score: a higher accuracy wins, and the lower loss breaks ties.
    fn compare(&self, other: &Score) -> Ordering {
        self.accuracy
            .total_cmp(&other.accuracy)
            .then(other.loss.total_cmp(&self.loss))
    }
}

/// The size of the search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Budget {
    /// Trials of the first round.
    pub trials: usize,
    /// Rounds of halving, at least 1.
    pub rounds: usize,
    /// Epochs of the first round. Each round doubles them.
    pub epochs: usize,
    pub seed: u64,
}

/// One network trained and scored.
#[derive(Debug, Clone, PartialEq)]
pub struct Trial {
    pub round: usize,
    /// The position of the trial in the first round, kept by the trials that go on to the next ones.
    pub id: usize,
    pub epochs: usize,
    pub params: Params,
    pub result: Result<Score, String>,
    pub seconds: f64,
}

impl Trial {
    pub fn to_json(&self) -> Value {
        let (accuracy, loss, error) = match &self.result {
            Ok(score) => (json!(score.accuracy), json!(score.loss), Value::Null),
            Err(error) => (Value::Null, Value::Null, json!(error)),
        };
        json!({
            "round": self.round + 1,
            "trial": self.id + 1,
            "epochs": self.epochs,
            "params": self.params,
            "validation_accuracy": accuracy,
            "validation_loss": loss,
            "error": error,
            "seconds": (self.seconds * 100.0).round() / 100.0,
        })
    }

    /// A trial that failed (the network diverged, for example) ranks below every trial that worked.
    fn compare(&self, other: &Trial) -> Ordering {
        match (&self.result, &other.result) {
            (Ok(a), Ok(b)) => a.compare(b),
            (Ok(_), Err(_)) => Ordering::Greater,
            (Err(_), Ok(_)) => Ordering::Less,
            (Err(_), Err(_)) => Ordering::Equal,
        }
    }
}

/// The best trial of the last round, and every trial that was run.
#[derive(Debug)]
pub struct Found {
    pub best: Trial,
    pub trials: Vec<Trial>,
}

/// Random search with successive halving. The first round draws `budget.trials` random sets of
/// hyperparameters and scores each with `evaluate(params, epochs)`. Every next round keeps the better half and
/// scores it again with twice the epochs, so the budget goes to the promising trials. With one round, this is a
/// plain random search. `on_trial` sees each trial as soon as it is scored.
pub fn search(
    space: &SearchSpace,
    budget: Budget,
    mut evaluate: impl FnMut(&Params, usize) -> Result<Score, String>,
    mut on_trial: impl FnMut(&Trial) -> Result<(), String>,
) -> Result<Found, String> {
    let mut rng = StdRng::seed_from_u64(budget.seed);
    let mut candidates: Vec<(usize, Params)> = (0..budget.trials)
        .map(|id| (id, space.sample(&mut rng)))
        .collect();
    let mut trials = Vec::new();
    let mut epochs = budget.epochs;
    for round in 0..budget.rounds {
        let mut scored = Vec::with_capacity(candidates.len());
        for (id, params) in &candidates {
            let start = Instant::now();
            let result = evaluate(params, epochs);
            let trial = Trial {
                round,
                id: *id,
                epochs,
                params: params.clone(),
                result,
                seconds: start.elapsed().as_secs_f64(),
            };
            on_trial(&trial)?;
            scored.push(trial.clone());
            trials.push(trial);
        }
        // The sort is stable and puts the best first, so equal trials keep their order of draw.
        scored.sort_by(|a, b| b.compare(a));
        let last = round + 1 == budget.rounds || scored.len() == 1;
        if last {
            let best = scored.swap_remove(0);
            return match &best.result {
                Ok(_) => Ok(Found { best, trials }),
                Err(error) => Err(format!("every trial of the last round failed: {error}")),
            };
        }
        scored.truncate(scored.len().div_ceil(2));
        candidates = scored.into_iter().map(|t| (t.id, t.params)).collect();
        epochs *= 2;
    }
    unreachable!("the budget has at least one round")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn space() -> SearchSpace {
        SearchSpace::from_json(r#"{"learning_rate":{"log_uniform":[0.001,1.0]}}"#).unwrap()
    }

    fn budget(trials: usize, rounds: usize) -> Budget {
        Budget {
            trials,
            rounds,
            epochs: 2,
            seed: 7,
        }
    }

    /// The closer the learning rate is to 0.1 (on a log scale), the better.
    fn fake(params: &Params, epochs: usize) -> Result<Score, String> {
        let rate = params["learning_rate"].as_f64().unwrap();
        Ok(Score {
            accuracy: 1.0 - (rate / 0.1).ln().abs() / 10.0 + epochs as f64 * 1e-6,
            loss: 0.0,
        })
    }

    #[test]
    fn halves_the_trials_and_doubles_the_epochs_each_round() {
        let found = search(&space(), budget(8, 3), fake, |_| Ok(())).unwrap();
        let shape: Vec<_> = (0..3)
            .map(|round| {
                let trials: Vec<_> = found.trials.iter().filter(|t| t.round == round).collect();
                (trials.len(), trials[0].epochs)
            })
            .collect();
        assert_eq!(shape, [(8, 2), (4, 4), (2, 8)]);
        assert_eq!(found.best.round, 2);
    }

    #[test]
    fn survivors_are_the_best_of_the_round_before() {
        let found = search(&space(), budget(8, 2), fake, |_| Ok(())).unwrap();
        let score = |trial: &Trial| trial.result.as_ref().unwrap().accuracy;
        let mut first: Vec<_> = found.trials.iter().filter(|t| t.round == 0).collect();
        first.sort_by(|a, b| score(b).total_cmp(&score(a)));
        let mut kept: Vec<_> = found
            .trials
            .iter()
            .filter(|t| t.round == 1)
            .map(|t| t.id)
            .collect();
        let mut expected: Vec<_> = first[..4].iter().map(|t| t.id).collect();
        kept.sort();
        expected.sort();
        assert_eq!(kept, expected);
    }

    #[test]
    fn finds_a_better_configuration_than_the_average_draw() {
        let found = search(&space(), budget(16, 1), fake, |_| Ok(())).unwrap();
        let scores: Vec<f64> = found
            .trials
            .iter()
            .map(|t| t.result.as_ref().unwrap().accuracy)
            .collect();
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let best = found.best.result.as_ref().unwrap().accuracy;
        assert!(scores.iter().all(|score| *score <= best));
        assert!(best > mean);
    }

    #[test]
    fn the_same_seed_gives_the_same_search() {
        let run = |seed| {
            let budget = Budget {
                seed,
                ..budget(6, 2)
            };
            let found = search(&space(), budget, fake, |_| Ok(())).unwrap();
            let drawn: Vec<_> = found.trials.into_iter().map(|t| (t.id, t.params)).collect();
            drawn
        };
        assert_eq!(run(1), run(1));
        assert_ne!(run(1), run(2));
    }

    #[test]
    fn a_failed_trial_never_beats_a_working_one() {
        let evaluate = |params: &Params, epochs: usize| {
            if params["learning_rate"].as_f64().unwrap() > 0.1 {
                Err("diverged".to_string())
            } else {
                fake(params, epochs)
            }
        };
        let found = search(&space(), budget(12, 2), evaluate, |_| Ok(())).unwrap();
        assert!(found.trials.iter().any(|t| t.result.is_err()));
        assert!(found.best.result.is_ok());
        for trial in found.trials.iter().filter(|t| t.round == 1) {
            assert!(trial.result.is_ok());
        }
    }

    #[test]
    fn fails_when_every_trial_fails() {
        let error = search(
            &space(),
            budget(3, 2),
            |_, _| Err("boom".into()),
            |_| Ok(()),
        )
        .unwrap_err();
        assert_eq!(error, "every trial of the last round failed: boom");
    }

    #[test]
    fn a_single_trial_stops_the_halving() {
        let found = search(&space(), budget(1, 4), fake, |_| Ok(())).unwrap();
        assert_eq!(found.trials.len(), 1);
    }

    #[test]
    fn reports_every_trial_and_stops_on_a_logging_error() {
        let mut seen = 0;
        search(&space(), budget(4, 2), fake, |_| {
            seen += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(seen, 6);
        let error = search(&space(), budget(4, 2), fake, |_| Err("disk full".into())).unwrap_err();
        assert_eq!(error, "disk full");
    }

    #[test]
    fn ties_in_accuracy_go_to_the_lower_loss() {
        let score = |loss| Score {
            accuracy: 0.5,
            loss,
        };
        assert_eq!(score(0.2).compare(&score(0.3)), Ordering::Greater);
        assert_eq!(score(f64::NAN).compare(&score(0.3)), Ordering::Less);
    }

    #[test]
    fn a_trial_is_logged_as_json() {
        let trial = Trial {
            round: 0,
            id: 2,
            epochs: 4,
            params: Params::from([("batch_size".to_string(), json!(64))]),
            result: Ok(Score {
                accuracy: 0.75,
                loss: 0.5,
            }),
            seconds: 1.234,
        };
        assert_eq!(
            trial.to_json(),
            json!({
                "round": 1, "trial": 3, "epochs": 4, "params": {"batch_size": 64},
                "validation_accuracy": 0.75, "validation_loss": 0.5, "error": null, "seconds": 1.23
            })
        );
        let failed = Trial {
            result: Err("diverged".into()),
            ..trial
        };
        assert_eq!(failed.to_json()["error"], "diverged");
        assert_eq!(failed.to_json()["validation_accuracy"], Value::Null);
    }
}
