use std::path::Path;

use my_torch::{Dataset, EpochReport, History, Model, Scalar};
use my_torch_analyzer::chess_file::{ChessFile, Labels};
use my_torch_analyzer::confusion::Confusion;
use my_torch_analyzer::dashboard::{Dashboard, Point, Set};
use my_torch_analyzer::encoding;
use my_torch_analyzer::sample::Sample;

pub fn predict(load_file: &Path, chess_file: &Path) -> Result<(), String> {
    let file = ChessFile::read(chess_file, Labels::Optional)?;
    let model = load(load_file)?;
    let labels = encoding::predict(model.network(), file.samples())?;
    for label in labels {
        println!("{label}");
    }
    Ok(())
}

pub fn train(
    load_file: &Path,
    chess_file: &Path,
    save_file: &Path,
    dashboard: bool,
) -> Result<(), String> {
    let file = ChessFile::read(chess_file, Labels::Required)?;
    let mut model = load(load_file)?;
    let conflicts = encoding::conflicts(file.samples());
    if conflicts > 0 {
        eprintln!(
            "warning: {conflicts} chessboards have conflicting labels, only the first one is kept"
        );
    }
    let samples = encoding::unique(file.samples().to_vec());
    println!(
        "training on {} chessboards ({} duplicates removed)",
        samples.len(),
        file.samples().len() - samples.len()
    );
    let dataset = encoding::dataset(&samples)?;
    let history = if dashboard {
        train_watched(&mut model, &dataset, &samples)?
    } else {
        model.train(&dataset)
    }
    .map_err(|error| format!("error: {error}"))?;
    report(&history);
    println!("accuracy: {:.2}%", accuracy(&model, &samples)? * 100.0);
    model
        .save(save_file)
        .map_err(|error| format!("error: {error}"))
}

/// Trains while serving the live dashboard, which tells the page how the training ended.
fn train_watched(
    model: &mut Model,
    dataset: &Dataset,
    samples: &[Sample],
) -> Result<Result<History, String>, String> {
    let dashboard = Dashboard::start(model.trainer().epochs(), model.network())
        .map_err(|error| format!("error: {error}"))?;
    eprintln!("dashboard: {}", dashboard.url());
    let trained = model.train_observed(dataset, |report| {
        let (point, confusion) = point(report, samples);
        dashboard.record(point, confusion);
    });
    dashboard.finish(trained.as_ref().map(|_| ()).map_err(String::as_str));
    Ok(trained)
}

fn point(report: &EpochReport, samples: &[Sample]) -> (Point, Option<(Set, Confusion)>) {
    // A matrix that cannot be measured (the network diverged) leaves gaps instead of stopping the training.
    let measure = |rows: &[usize]| encoding::confusion(report.network(), samples, rows).ok();
    let training = measure(report.training_samples());
    let validation = report
        .validation_loss()
        .and_then(|_| measure(report.validation_samples()));
    let accuracy = |confusion: Option<Confusion>| confusion.map_or(Scalar::NAN, |c| c.accuracy());
    let point = Point {
        epoch: report.epoch(),
        training_loss: report.training_loss(),
        validation_loss: report.validation_loss(),
        training_accuracy: accuracy(training),
        validation_accuracy: report.validation_loss().map(|_| accuracy(validation)),
    };
    let shown = match (validation, training) {
        (Some(confusion), _) => Some((Set::Validation, confusion)),
        (None, Some(confusion)) => Some((Set::Training, confusion)),
        (None, None) => None,
    };
    (point, shown)
}

fn load(path: &Path) -> Result<Model, String> {
    let model = Model::load(path).map_err(|error| format!("error: {error}"))?;
    encoding::check_shape(model.network())
        .map_err(|error| format!("error: {}: {error}", path.display()))?;
    Ok(model)
}

fn report(history: &History) {
    for epoch in 0..history.epochs() {
        match history.validation().get(epoch) {
            Some(validation) => println!(
                "epoch {}: loss {:.4}, validation loss {validation:.4}",
                epoch + 1,
                history.training()[epoch]
            ),
            None => println!("epoch {}: loss {:.4}", epoch + 1, history.training()[epoch]),
        }
    }
    if let Some(best) = history.best_epoch() {
        println!("best epoch: {}", best + 1);
    }
}

fn accuracy(model: &Model, samples: &[Sample]) -> Result<f32, String> {
    let labels = encoding::predict(model.network(), samples)?;
    let correct = labels
        .iter()
        .zip(samples)
        .filter(|(label, sample)| Some(**label) == sample.label())
        .count();
    Ok(correct as f32 / samples.len() as f32)
}
