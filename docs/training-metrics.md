# Training metrics

What to measure while training the chess network, how to read it, and what to try when the accuracy is too low.
`History` (`crates/my_torch/src/history.rs`) already records the training and validation loss of each epoch, and
`configs/chess.conf` enables early stopping on a 10 % validation split. This document lists what to add on top.

## Metrics to track during training

### Loss curves (training and validation, every epoch)

The main signal to detect overfitting.

| Pattern | Diagnosis | What to do |
|---|---|---|
| Both losses high and flat | Underfitting | Larger network, higher learning rate, more epochs |
| Training loss falls, validation loss rises | Overfitting | Early stopping, regularization, more data |
| Both low and close to each other | Good fit | Keep this model |
| Loss oscillates or explodes | Learning rate too high | Lower the learning rate |
| Loss is `NaN` | Divergence | Lower the learning rate, check the data and the initialization |

The best epoch is the one with the lowest validation loss (`History::best_epoch`), and it is the point early
stopping aims for.

### Accuracy on the training and the validation sets

Track both separately. The gap between them is a quick overfitting indicator. Loss alone can hide a model that
becomes more confident without becoming more correct.

### Metrics beyond accuracy

The labels are likely imbalanced: `Nothing` probably dominates, while checkmate and stalemate are rare. A model
that always predicts `Nothing` can then look very accurate. Also track:

- **A confusion matrix** over the 6 labels, to see which labels get confused (for example check and checkmate).
- **Per-class precision, recall and F1**, which matter most for the rare labels.
- **Macro-F1** (the unweighted mean of the per-class F1) as the single headline number.
- **The label distribution** of the dataset. It gives the baseline: the accuracy of always predicting the majority
  label. The network has to beat it.

### Training-health diagnostics

- **The gradient norm of each layer.** Vanishing or exploding gradients show up here.
- **The fraction of dead ReLUs**, meaning units that always output 0.
- **The weight norm**, which can grow when the network overfits.
- **A sanity check**: overfit a tiny subset (about 50 boards). If the loss cannot reach about 0, there is a bug or
  the capacity is too low.

## Levers to raise the accuracy

These are hypotheses to test one at a time, not measured results.

1. **Check the data first**: label distribution, label noise, size of the deduplicated dataset. It is often the
   biggest lever.
2. **Handle the imbalance** with a class-weighted loss or by oversampling the rare labels.
3. **Optimizer**: plain SGD at a small learning rate is slow. Momentum or Adam usually helps, as does a decaying
   learning rate.
4. **Capacity**: 769 inputs into 128 then 64 neurons is small for chess. Try wider or deeper networks, once the
   validation gap is visible.
5. **Regularization** if the network overfits: L2 weight decay, more data. Early stopping and dropout (`"dropout"` on a layer of the configuration) already exist.
6. **Evaluate on a held-out test set** that is never used for tuning. Do not choose hyperparameters on the
   validation split that drives early stopping.

## Possible next step

Record the training and validation accuracy per epoch in `History`, and add an evaluation step that prints the
confusion matrix and the per-class precision, recall and F1 after training.

## Sources

- [How to use Learning Curves to Diagnose Machine Learning Model Performance](https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/)
- [Learning curve to identify overfitting/underfitting problems](https://towardsdatascience.com/learning-curve-to-identify-overfitting-underfitting-problems-133177f38df5)
- [Why an Increasing Validation Loss and Validation Accuracy Signifies Overfitting?](https://www.geeksforgeeks.org/?p=1168755)
- [Practice: visualizing overfitting](https://apxml.com/courses/deep-learning-regularization-optimization/chapter-1-generalization-challenge/practice-visualizing-overfitting)

The loss-curve reading comes from these sources. The imbalance, per-class metrics and training-health diagnostics
are general machine-learning practice, not taken from them.
