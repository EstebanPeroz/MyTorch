use my_torch::{
    Activation, Architecture, Dataset, LayerSpec, Loss, Matrix, MeanSquaredError, Model, Rng,
    Scalar, Sgd, SoftmaxCrossEntropy, Trainer, XavierNormal,
};

const SEED: u64 = 42;
const INPUTS: [[Scalar; 2]; 4] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
const EXPECTED: [usize; 4] = [0, 1, 1, 0];

fn inputs() -> Matrix {
    Matrix::from_rows(&INPUTS).unwrap()
}

fn one_output() -> Dataset {
    Dataset::new(
        inputs(),
        Matrix::from_rows(&[[0.0], [1.0], [1.0], [0.0]]).unwrap(),
    )
    .unwrap()
}

fn two_classes() -> Dataset {
    Dataset::new(
        inputs(),
        Matrix::from_rows(&[[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]).unwrap(),
    )
    .unwrap()
}

fn train(
    output: (usize, Activation),
    xor: &Dataset,
    loss: impl Loss + 'static,
    learning_rate: Scalar,
) -> Model {
    let (outputs, activation) = output;
    let architecture = Architecture::new(
        2,
        vec![
            LayerSpec::new(8, Activation::Tanh, XavierNormal).unwrap(),
            LayerSpec::new(outputs, activation, XavierNormal).unwrap(),
        ],
    )
    .unwrap();
    let network = architecture.build(&mut Rng::new(SEED));
    let trainer = Trainer::new(loss, Sgd::new(learning_rate).unwrap(), 2, 2000).unwrap();
    let mut model = Model::new(network, trainer, SEED).unwrap();
    let history = model.train(xor).unwrap();
    let last = history[history.len() - 1];
    assert!(last < 0.05, "final loss {last}");
    model
}

#[test]
fn learns_xor_with_one_sigmoid_output() {
    let model = train(
        (1, Activation::Sigmoid),
        &one_output(),
        MeanSquaredError,
        0.5,
    );
    let outputs = model.network().predict(&inputs()).unwrap();
    for (row, expected) in EXPECTED.into_iter().enumerate() {
        let output = outputs.get(row, 0);
        assert_eq!(
            usize::from(output > 0.5),
            expected,
            "{:?} gave {output}",
            INPUTS[row]
        );
    }
}

#[test]
fn learns_xor_as_two_classes() {
    let model = train(
        (2, Activation::Identity),
        &two_classes(),
        SoftmaxCrossEntropy,
        0.2,
    );
    let outputs = model.network().predict(&inputs()).unwrap();
    assert_eq!(outputs.argmax_rows(), EXPECTED);
}

#[test]
fn a_saved_network_predicts_the_same_and_keeps_training_after_loading() {
    let xor = two_classes();
    let model = train((2, Activation::Identity), &xor, SoftmaxCrossEntropy, 0.2);
    let path = std::env::temp_dir().join(format!("my_torch_xor_{}.nn", std::process::id()));
    model.save(&path).unwrap();
    let loaded = Model::load(&path);
    std::fs::remove_file(&path).unwrap();
    let mut loaded = loaded.unwrap();
    assert_eq!(
        loaded.network().predict(&inputs()),
        model.network().predict(&inputs())
    );
    let history = loaded.train(&xor).unwrap();
    assert_eq!(history.len(), 2000);
    assert!(history[history.len() - 1] < 0.05);
}
