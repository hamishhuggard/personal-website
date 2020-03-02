* Following the success of back-propagation, neural network research gained popularity and reached a peak in the early 1990s. Afterwards, other machine learning
techniques became more popular until the modern deep learning renaissance that began in 2006.
* Regularization of an estimator works by trading increased bias for reduced variance.
* in neural networks, typically only the weight and not the biases are used in normalisation penalties
* Effect of weight decay: small eigenvalue directions of the Hessian are reduced more than large eigenvalues.
 * Linear hidden units can be useful: if rather than g(Wx+b) we have g(UVx+b) then we have effectively factorised W, which can save parameters (at cost of constraining W to lower rank).
 * "The softplus demonstrates that the performance of hidden unit types canbe very counterintuitive—one might expect it to have an advantage overthe rectiﬁer due to being diﬀerentiable everywhere or due to saturating lesscompletely, but empirically it does not."
 * Hard tanh
 * L2 regularisation is comes from Gaussian prior over weights + MAP
 * L1 regularisation comes from Laplacian prior
 * L-norms are equivalent to constrained optimisation problems: constraining to an L-n ball whose radius depends on the form of the loss
 * With early stopping, after you've finished on the training set, you can now also train on the validation data
   * You can either train again from scratch with the val data added in
     * You can do the same number of parameter updates
     * Or same number of passes through the data
   * Or also train on the validation data after the first round of training
    * Perhaps until the objective function on the validation set reaches the same level as the training set
 * Early stopping is in a sense equivalent to L2 regularization in that it limits the length of the optimisation trajectory. It is superior in that it automatically fine tunes the eta hyperparameter1
 * Bagging = bootstrap aggregating
 * Dropout: Typically, an input unit is included with probability 0.8, and a hidden unit is included with probability 0.5.
 * Although the extra cost of dropout per step is negligable, it does require longer training and a larger model. If the dataset is large then this probably isn't worthwhile.
 * Wang and Manning (2013) showed that deterministic dropout can converge faster
 * Regularising noise has to be multiplicative rather than additive because otherwise the outputs could just be made very large
 * Virtual adversarial training: generate example x which is far from any real examples and make sure that the model is smooth around x
 * Smaller batch sizes are better for regularisation, but often underutilise hardware
 * Second order optimisation techniques like Newtons method appear to have not taken off due to them getting stuck in saddle points
 * Cliffs are common in objective landscapes because of "a multiplication of many factors"
 * Exploding/vanishing gradient illustration: if you are multiplying by M repeatedly, then eigenvalues > 1 will explode and < 1 will vanish. This can cause cliffs.
 * A sufficient condition for convergence of SGD is $\sum_{k=1}^\infty \epsilon_k = \infty$ and $\sum_{k=1}^\infty \epsilon_k^2 < \infty$
 * Often a learning rate is decayed such that $\epsilon_t = (1-\alpha) \epsilon_{t-1} + \alpha \epsilon_\tau
 * Newton's method isn't commonly used because calculating the inverse Hessian is O(k^n) with number of parameters
 * coordinate descent: optimise one parameter at a time
 * block coordinate descent: optimise a subset of parameters at a time
 * Three ways to decide the length of a output sequence of a RNN:
   * Have an <END> token
   * Have a binary output saying whether the sequence is done or not
   * Have a countdown to the end as one of the outputs
 * Reservoir computing: hard code the weights for the recurrent and input connections, only train output
 * You can only accept predictions above a given confidence. Then the metric you might use is coverage.
  * If the error rate on the training set is low, try to get this up by increasing model size, more data, better hyperparameter, etc.
 * If the error rate on the test set is low, try adding regularizers, more data, etc.
 * Hyperparameter vs loss tends to be U-shaped
 * One common learning rate regime is to wait for plateaus then reduce the learning rate 2-10x
 * Learning rate is probably the most important hyperparameter
 * Learning rate vs training error: apparently exponential decay then a sharp jump upwards (p. 430)
 * In conv nets, there are three schemes (TODO: look up names):
   * no padding
   * pad enough to preserve image size
   * pad enough so that all pixels get equal convolutions (increase image size)
 * Grid hyperparameter search is normally iterative: if you search [-1, 0, 1] and find 1 is the best, then you should expand the window to the right
 * To debug: look at most confident mistakes
 * Test for bprop bug: manually estimate the gradient (see p. 439)
 * Monitor outputs/entropy, activations (for dead units), updates to weight magnitude ratio (should be ~1%)
 * Your metric can be coverage: hold accuracy constant and try improve coverage
 * Groups of GPU threads are called warps
 * Mixture of experts: one model predicts weighting of expert predictions.
 * Hard mixture of experts: one model chooses a single expert predictor
 * Combinatorial gaters: choose a subset of experts to vote
 * Switch: model receives subset of inputs (similar to attention)
 * To save on compute use cascades: use a cheap model for most instances, and an expensive model when some tricky feature is present. Use a cascade of models to detect the tricky feature: the first has high recall, the last has high precision.
 * Common preprocessing step is to make each pixel have mean zero and std one. But for low-information pixels this may just amplify noise or compression artefacts. So you want to add a regularizer (p. 455)
 * Image preprocessing:
   * sphere = whitening
   * GCN = global contrast normalisation (whole image has mean zero and std 1)
   * LCN = local contrast normalisation (each window / kernel has mean zero nad std 1)
 * Rather than a binary tree for hierarchical softmax, you can just have a breadth-sqrt(n) and depth 2 tree
 * ICA is used in EEG to separate the signal from the brain from the signal from the heart and the breath
 * Recirculation: autoencoder to match layer-1 activations of original input with reconstructed input
 * Under-complete autoencoders have lower representational power in hidden space than the original space
 * Over-compelte autoencoders have at least as much representational power in hidden space as original space
 * Denoising autoencoder: L(x, g(f(x + epsilon)) )
 * CAE = contractive autoencoder: penalises derivatives (so it doesn't change much with small changes in x)
 * score matching; try and get the same \nabla_x \log p(x) for all x
 * Rifai 2011a is where the iterative audoencoder comes from
 * Autoencoder failure mode: f can simply multiply by epsilon, and g divide by epsilon, and thereby achieve perfect reconstruction and low contractive penalty
 * Semantic hashing: use autoencoder to create a hash of instances to help with information retrieval. If the last layer of the autoencoder is softmax, you can force it to saturate at 0 and 1 by adding noise just before the softmax, so it will have to push further towards 0 and 1 to let the signal get through
 * Denoising autoencoders learn a vector field towards the instance manifold
 * Predictive sparse decomposition is a thing
 * Autoencoders can perform better than PCA at reconstruction
 * I didn't understand much of linear factor models
   * Probabilistic PCA
   * Slow feature analysis
   * independent component analysis
 * You can coerce a representation that suits your task better
  * E.g., for a density estimation task you can encourage independence between components of hidden layer h
 * Greedy layerwise unsupervised pretraining goes back to the neocognitron, and it was the discovery that you can use this pretraining to get a fully connected network to work properly that sparked the 2006 DL renaissance
 * Pretraining makes use of two ideas:
   * parameter initial conditions can have a regularisation effect
   * learning about input distribution can help with prediction
 * Unsupervised pretraining has basically been abandoned except for word embedding.
   * For moderate and large datasets simply supervised learning works better
   * For small datasets Bayesian learning works better
 * What makes good features? Perhaps if you can capture the underlying (uncorrelated) causes these, these would be good features.
 * Distributed representations are much more expressive than non-distributive ones
 * Radford 2015 does vector arithmetic with images
 * While NFL theorems mean that there's no universal prior or regularisation advantage, we can choose some which provide an advantage in a range of tasks which we are interested in. Perhaps priors similar to those humans or animals have.
 * To calculate probabilities in undirected graphs (e.g., modelling sickness between you, your coworker, your roommate) take the product of the "clique potentials" for each clique (and normalise). The distribution over clique products is a "gibbs distribution"
 * The normalising Z is known as the partition function (statistical physics terminology)
 * "d-separation": in graphical models context, this stands for "dependence separation" and means that there's no flow of information from node set A to node set B
 * Any relationship structure can be modelled with directed or undirected graphs. The value of these is that they eliminate dependencies.
 * "immorality": a collider structure in a directed graph. To convert this to an undirected graph, it needs to be "moralised" by connecting the two incoming nodes (creating a triangle). The terminology comes from a joke about unmarried parents.
 * In undirected graphs a loop of length greater than 3 without chords needs to be cut up into triangles before it can be represented as a directed graph
 * To sample from a directed graph, sample from each node in topographical order
 * Restricted Boltzmann Machine = Harmonium
 * It consists of one fully connected hidden layer, and one non-hidden layer
 * The "restricted" means that there's no connections between hidden layers
 * Continuous Markov chain = Harris chain
 * Perron-Frobenius Theorem: for a transition matrix, if there are no zero-probability transitions, then there will be a eigenvalue of one.
 * Running a Markov chain to reach equilibrium distribution is called "burning in" the Markov chain.
 * Sampling methods like GIbbs sampling can get stuck in one mode. Tempered transitions means reducing the temperature of the transition function between samples to make it easier to jump between modes
 * There's two kinds of sampling: Las Vegas sampling which always either returns the correct answer or gives up, and Las Vegas sampling, which will always return an answer, but with a random amount of error
 * You can decompose sampling into a positive phase and a negative phase:
   * $\nabla_\theta \log p(x;\theta) = \nabla_\theta \log \tilde{p}(x;\theta) + \nabla_\theta \logZ()\theta) $
 * I'm not understanding a ton of the chapter on the partition function.
 * Biology vs back prop: If brains are implementing back prop, then there needs to be a secondary mechanism to the usual axon activation.
  * Hinton 2007a, Bengio 2015 have proposed biologically plausible mechanisms.
 * Dreams may be sampling from the brains model during negative phase learning (Crick and Mitchison, 1983)
   * Ie, "approximate the negative gradient of the log partition function of undirected models."
   * It could also be about sampling p(v,h) for inference (see p. 651)
   * Or may be about reinforcement learning
 * Most important generative architectures:
   * RBM: restricted boltzmann machine, bipartate & undirected
   * DBN: deep belief network, RBM plus a feedforward layer to the sensors
   * DBM: deep Boltzmann machine, stack of RBMs
 * RBMs are bad for computer vision because it's hard to expression max pooling in energy functions.
   * Also, the partition function changes with different sizes on input
   * Also doesn't work well with boundaries
 * There are lots of problems with evaluating generative models.
   * One approach is blind taste testing.
   * To prevent the model from memorising, also display the nearest neighbour in the dataset
   * Some models good at maximising likelihood of good examples, others good at minimising likelihood of bad examples
    * Depends on the direction of the KL divergence
   * Theis et al. (2015) reviews issues with generative models
