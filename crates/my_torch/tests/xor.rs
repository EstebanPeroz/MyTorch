use my_torch::{
    Activation, Dataset, Dense, Loss, Matrix, MeanSquaredError, Network, Rng, Scalar, Sgd,
    SoftmaxCrossEntropy, Trainer, XavierNormal,
};

const SEED: u64 = 42;
const INPUTS: [[Scalar; 2]; 4] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
const EXPECTED: [usize; 4] = [0, 1, 1, 0];

fn train(
    output: (usize, Activation),
    targets: Matrix,
    loss: impl Loss + 'static,
    learning_rate: Scalar,
) -> Network {
    let mut rng = Rng::new(SEED);
    let (outputs, activation) = output;
    let mut network = Network::new(vec![
        Dense::random(2, 8, Activation::Tanh, &XavierNormal, &mut rng),
        Dense::random(8, outputs, activation, &XavierNormal, &mut rng),
    ])
    .unwrap();
    let xor = Dataset::new(Matrix::from_rows(&INPUTS).unwrap(), targets).unwrap();
    let mut trainer = Trainer::new(loss, Sgd::new(learning_rate).unwrap(), 2, 2000).unwrap();
    let history = trainer.fit(&mut network, &xor, &mut rng).unwrap();
    let last = history[history.len() - 1];
    assert!(last < 0.05, "final loss {last}");
    network
}

#[test]
fn learns_xor_with_one_sigmoid_output() {
    let targets = Matrix::from_rows(&[[0.0], [1.0], [1.0], [0.0]]).unwrap();
    let network = train((1, Activation::Sigmoid), targets, MeanSquaredError, 0.5);
    let outputs = network
        .predict(&Matrix::from_rows(&INPUTS).unwrap())
        .unwrap();
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
    let targets = Matrix::from_rows(&[[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]).unwrap();
    let network = train((2, Activation::Identity), targets, SoftmaxCrossEntropy, 0.2);
    let outputs = network
        .predict(&Matrix::from_rows(&INPUTS).unwrap())
        .unwrap();
    assert_eq!(outputs.argmax_rows(), EXPECTED);
}
