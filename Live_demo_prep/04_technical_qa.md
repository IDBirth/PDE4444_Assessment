# Technical Q&A

## Activation function selection

### Q: Why did you compare multiple activation functions?

Because activation functions control non-linearity, gradient flow, and
training stability. The assessment specifically requires comparing at least
two activations, and in practice they can change convergence behaviour and
final accuracy.

### Q: What did you compare?

For the scratch CNN and MobileNet transfer-learning experiments, the main
activations were:

- ReLU
- ELU
- GELU
- SELU
- LeakyReLU

### Q: What did you observe?

For the scratch CNN in `runs5`, `GELU` was best:

- `cnn_gelu` F1 = `0.7582`

But the whole scratch-CNN family remained weak compared with transfer
learning, which shows that activation choice alone cannot overcome limited
data when the feature extractor is trained from scratch.

For the fine-tuned MobileNetV2 family in `runs5`, the best models were:

- `mobilenet_relu` F1 = `0.9545`
- `mobilenet_leaky_relu` F1 = `0.9545`
- `mobilenet_elu` F1 = `0.9385`

So the answer is not "one activation is always best". The best activation
depends on the model family and optimisation setting.

### Q: Why not just use ReLU everywhere?

ReLU is simple and efficient, but it can produce dead neurons when many
activations stay negative. Alternatives such as ELU or GELU can provide
smoother gradients and sometimes more stable optimisation. That is why the
comparison was worth running rather than assuming one default choice.

## Optimisation method comparison

### Q: Which optimisation methods did you compare?

The optimiser comparison experiment covered:

- Nelder-Mead: zero-order
- SGD: first-order
- Adam: first-order adaptive
- L-BFGS: second-order / quasi-Newton

Results in `runs5`:

- Adam: F1 `0.8472`
- L-BFGS: F1 `0.7727`
- SGD: F1 `0.7586`
- Nelder-Mead: F1 `0.6203`

### Q: Why was Adam the best?

Adam combines gradient descent with adaptive learning rates per parameter.
That usually helps on noisy small-dataset problems because training is more
stable and less sensitive to a single global step size.

### Q: Why was Nelder-Mead poor?

Nelder-Mead is derivative-free, which makes it attractive conceptually for
zero-order comparison, but it scales badly with parameter count. Even with
the deliberately small MLP used here, it was much less effective than
gradient-based methods.

### Q: Why did L-BFGS not win despite using second-order information?

L-BFGS can fit fast, but on a small dataset it also overfits quickly. So it
optimises the training objective well without necessarily giving the best
generalisation.

## Model limitations

### Q: What are the main limitations of your best model?

1. The dataset is relatively small.
2. The test set is balanced and controlled, but real factory variation can
   be larger than the dataset variation.
3. Out-of-distribution conditions such as new lighting, camera angle, or a
   different cup family are not fully stress-tested.
4. The model predicts a class and confidence, but it is still a learned
   statistical model, not a rule-based defect explanation engine.

### Q: Why did the baseline sklearn models struggle?

There are two levels to that answer:

1. Before balancing, the dataset was dominated by one class, so the
   classical models tended to collapse toward majority-class behaviour.
2. Even after balancing, HOG-based classical models compress the image into
   hand-crafted edge descriptors. That works reasonably well, but it does
   not adapt feature extraction to subtle surface defects as effectively as
   pretrained deep models.

### Q: Why did the MLP random search work so well?

Because it searched over:

- hidden layer depth/width
- activation
- dropout
- learning rate

In `runs5`, the best setting was:

- hidden sizes: `(256, 128)`
- activation: `gelu`
- dropout: `0.0`
- learning rate: `0.0003`

This gave F1 `0.9545`, which shows that a well-tuned dense model can be
very strong when the input pipeline and optimisation are chosen carefully.

### Q: Why is YOLO26n-cls better?

Because it starts from a pretrained visual backbone and is optimised for
fast, strong image classification. In this project it achieved:

- `top1 = 0.9928`
- `top5 = 1.0`

It combines high accuracy, compact size, and practical real-time behaviour,
which makes it the best model for the live demo and the most realistic
deployment candidate.
