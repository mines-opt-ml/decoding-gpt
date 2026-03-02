---
title: Intro to mechanistic interpretability
---

*Mechanistic Interpretability* ("mechinterp") is the science of understanding how neural networks work, by understanding the mechanisms of their computation, and intervening in those mechanisms to test hypotheses.

A good survey paper, one this document borrows from extensively, is [Räuker et al., 2023](https://arxiv.org/pdf/2207.13243).

**Prerequisites:** Familiarity with transformer architecture (attention, MLP blocks, residual stream), basic PyTorch, and comfort with linear algebra.

---

# Notation

| Symbol                                       | Meaning                                       |
| -------------------------------------------- | --------------------------------------------- |
| $n_L$                                        | Number of layers                              |
| $n_c$                                        | Sequence length (context)                     |
| $d_m$                                        | Model (residual stream) dimension             |
| $d_v$                                        | Vocabulary size                               |
| $X^{(\ell)} \in \mathbb{R}^{n_c \times d_m}$ | Residual stream at layer $\ell$               |
| $x^{(\ell)}_t \in \mathbb{R}^{d_m}$          | Residual stream at layer $\ell$, position $t$ |
| $W_E \in \mathbb{R}^{d_v \times d_m}$        | Token embedding matrix                        |
| $W_U \in \mathbb{R}^{d_m \times d_v}$        | Unembedding matrix                            |
| $\texttt{Attn}_\ell, \texttt{MLP}_\ell$      | Attention and MLP blocks at layer $\ell$      |

## Transformer Architecture Recap

Residual stream update: $X^{(\ell)} = X^{(\ell-1)} + \texttt{Attn}_\ell(X^{(\ell-1)}) + \texttt{MLP}_\ell(X^{(\ell-1)})$. Output logits: $X^{(n_L)} W_U \in \mathbb{R}^{n_c \times d_v}$.

An MLP (Multi-Layer Perceptron) is broadcast along positions and maps a vector $x_t \in \R^{d_m}$ to another vector in the same space:
$$
	\texttt{MLP}_\ell(x_t) = W_2 \, \sigma_a(W_1 x_t + b_1) + b_2
$$

Where $\sigma_a$ is the activation function (ReLU, GeLU, etc.) and $W_1, W_2 \in \R^{d_m \times d_m}$ are the learnable weight matrices.

An individual attention head is a function
$$A: \R^{n \times d_m} \to \R^{n \times d_m}$$

$$
  [A(X)]_i = \sum_{j=1}^n \texttt{AP}(X)_{i,j} W_{OV} x_j
$$

and the "attention pattern" $\texttt{AP}(X) \in \R^{n \times n}$ is a "measure of similarity" between the embeddings. 

- $\texttt{AP}(X)_{i,j}$ decides how much to move from $j$ to $i$
- $W_{OV} x_j$ decides *what* to move from $j$

So where do we get $\texttt{AP}(X)$ from?
$$
  \texttt{AP}(X)_{i,j} = \sigma_S\left(x_i W_{QK} x_j^T + M_{i,j}\right)
$$
where 
$$
  M \in \R^{n \times n}, \quad
  M := \begin{cases}
    0 & i \leq j \\
    -\infty & i > j
  \end{cases}
$$

Masking is necessary to prevent the model from "cheating" by looking at future tokens. $\sigma_S$ is the softmax function applied row-wise:

$$
  \sigma_S(B)_{i,j} = \frac{e^{B_{i,j}}}{\sum_k e^{B_{i,k}}}
$$


The learnable parameters of an attention head are thus given by the matrices $W_{QK}, W_{OV} \in \R^{d_m \times d_m}$. In practice, we learn low-rank approximations to both of these. An attention head block is the sum of multiple attention heads:

$$
	\texttt{Attn}_\ell(X) = \sum_{h=1}^{n_H} A_{\ell, h}(X)
$$


# 1. The Residual Stream and the Logit Lens

## Motivation

Before we try to understand *how* a model computes, we should be able to see *what it's thinking* at intermediate layers. The residual stream view of transformers, where each layer additively contributes to a running representation, gives us a natural way to do this. One of the first tools developed for this purpose is the **logit lens**: applying the unembedding matrix $W_U$ to intermediate residual stream activations to get "predicted logits" at each layer.

## Key Ideas

Since $W_U$ maps the final residual stream to logits, we can apply it at *any* intermediate layer to get a "prediction" at that point:

$$\text{logits}^{(\ell)} = X^{(\ell)} W_U \in \mathbb{R}^{n_c \times d_v}$$

This is the **logit lens** ([nostalgebraist, 2020](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)). It lets you watch the model's "best guess" evolve layer by layer.

## The Tuned Lens

The raw logit lens assumes intermediate representations live in the same space as the final layer, which is only approximately true. The **tuned lens** ([Belrose et al., 2023](https://arxiv.org/abs/2303.08112)) learns an affine probe per layer:

$$\text{logits}^{(\ell)} = \bigl( X^{(\ell)} A_\ell + b_\ell \bigr) W_U$$

This is a small overhead (one linear map per layer) but significantly improves prediction quality and gives a more faithful picture of what the model "knows" at each depth.

## Limitations

The logit lens family tells you *what* the model predicts at each layer, not *why*, and is not a complete explanation. Treat it as a first-pass tool for identifying interesting layers and behaviors to investigate with the heavier tools below.

---

# 2. Activation Patching and Causal Tracing

## Motivation

We want to move from "what does the model predict?" to "where in the model is a given fact / behavior computed?" Activation patching is the workhorse for this: it lets you localize computation by running counterfactual interventions on the model's internals.

## Key Ideas

The setup requires two inputs: a **clean** input (where the model behaves as expected) and a **corrupted** input (where the behavior breaks). For example:

- Clean: "The Eiffel Tower is located in" → "Paris"
- Corrupted: "The Colosseum is located in" → "Rome" (or noise-corrupted version)

**Activation patching** proceeds as follows:

1. Run the model on the corrupted input, caching all activations.
2. Run the model on the clean input, but at a specific layer and position, *replace* the activation with the one from the corrupted run.
3. Measure how much the output changes.

If patching at site $(l, t)$ — layer $l$, token position $t$ — causes the model to recover (or lose) the clean behavior, that site is causally implicated.

More precisely, let $a^{(\ell, t)}_{\text{clean}}$ and $a^{(\ell, t)}_{\text{corrupt}}$ be the activations at site $(\ell, t)$ under the two runs. We define:

$$\text{Effect}(\ell, t) = m\bigl(\text{model with } a^{(\ell, t)}_{\text{corrupt}} \text{ patched in}\bigr) - m\bigl(\text{clean run}\bigr)$$

where $m$ is some metric (e.g., logit of the correct token, log-probability, loss).

## Variants

- **Noising (corrupted → clean patching):** Start from corrupted run, patch in clean activations. Measures whether a site is *sufficient* to restore the behavior.
- **Denoising (clean → corrupted patching):** Start from clean run, patch in corrupted activations. Measures whether a site is *necessary* for the behavior.
- **Path patching** ([Goldowsky-Dill et al., 2023](https://arxiv.org/abs/2304.05969)): Patch activations along specific edges in the computational graph (e.g., the output of attention head 7.3 as it feeds into MLP 8), not just at nodes. This gives finer-grained localization.
- **Causal tracing** ([Meng et al., 2022](https://arxiv.org/abs/2202.05262)): A specific protocol using noise-corrupt-restore on factual recall tasks, which revealed that factual associations are primarily stored in MLP layers at the last subject token.

## The Computational Graph

Think of the transformer as a graph: each attention head and MLP block is a node, and the residual stream connections are edges. Activation patching lets you ablate or intervene on nodes; path patching extends this to edges. The ACDC algorithm (Section 3) automates the search over this graph.

## Practical Notes

- The choice of corruption matters. Gaussian noise preserves positional structure; alternative-entity corruption is cleaner for factual recall but requires curated datasets.
- Patching at the level of full residual stream positions is coarse; patching individual head outputs or MLP outputs is more informative but far more expensive. A key issue here is figuring out what a "component" in a model actually is.

Patching measures *marginal* contributions, which can miss distributed or redundant computations. If two heads each contribute 30% and removing either one causes the other to compensate (backup behavior), patching either alone shows a small effect. This is a general problem with interventional methods.


---


# 2.b. Activation steering

> [WIP]

Originally from [Turner et al., 2023](https://arxiv.org/abs/2308.10248), activation steering allows modifying model behavior by patching in activations, usually gathered by combining (naively, the mean) of activations from a set of examples that share a concept. For example, to steer the model towards "positive sentiment", you could average the activations from a bunch of positive reviews and patch that into the model at inference time.

---

# 3. Circuits: Discovery and Analysis

## Motivation

The circuits research agenda ([Olah et al., 2020](https://distill.pub/2020/circuits/zoom-in/); [Elhage et al., 2021](https://transformer-circuits.pub/2021/framework/index.html)) aims to decompose neural networks into interpretable subgraphs — small collections of attention heads and MLP neurons that together implement a specific behavior. The goal is reverse engineering: convert the DNN subgraph into pseudocode.

## Key Concepts

A **circuit** is a computational subgraph of the model that is:

1. **Faithful:** It approximates the full model's behavior on the relevant distribution.
2. **Minimal:** Removing any component degrades performance.
3. **Interpretable:** Each component has a human-understandable role.

In practice, (3) is aspirational — the emphasis in automated methods is on (1) and (2).

## Manual Circuit Discovery

The original Transformer Circuits Thread ([Elhage et al., 2021](https://transformer-circuits.pub/2021/framework/index.html); [Olsson et al., 2022](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)) presented hand-analyzed circuits:

- **Induction heads:** A two-head circuit (previous-token head + induction head) that implements the pattern "...A B ... A → B". This is a cornerstone result in mechanistic interpretability, underlying in-context learning and is found across model scales.
- **Indirect Object Identification (IOI):** [Wang et al. (2023)](https://arxiv.org/abs/2211.00593) reverse-engineered a 26-head circuit in GPT-2 Small for the task "When Mary and John went to the store, John gave a drink to" → "Mary". They identified name mover heads, backup name mover heads, S-inhibition heads, and duplicate token heads, each with a clear functional role.

The methodology is: (i) identify a narrow behavior, (ii) use activation patching to localize key components, (iii) analyze the attention patterns and OV/QK matrices of those components, (iv) validate by ablating the circuit and checking faithfulness.

## Automated Circuit Discovery: ACDC

Manual circuit analysis is laborious and doesn't scale. **ACDC** ([Conmy et al., 2023](https://arxiv.org/abs/2304.14997)) automates the process:

1. Start with the complete computational graph.
2. For each edge, measure its importance via activation patching.
3. Prune edges below a threshold.
4. The remaining subgraph is the circuit.

ACDC recovers known circuits (like IOI) and can discover new ones, though the resulting circuits are sometimes larger and less interpretable than hand-found ones.

## Other Automated Approaches

- **Attribution patching** (Neel Nanda's approximation): Use gradients to approximate the patching effect, avoiding the $O(n)$ forward passes per edge. Much faster, though noisier.
- **Subnetwork probing / circuit probing** (Conmy & Mavor-Parker, 2024): Train a mask over the computational graph to find minimal faithful subnetworks.
- **ACDC variants:** Edge-level vs. node-level pruning, different metrics (KL divergence, logit diff, etc.), iterative vs. one-shot.

## Validating Circuits

A discovered circuit should be tested for:

- **Faithfulness:** Does the circuit-only model match the full model? Measure KL divergence or task metric on a held-out set.
- **Completeness:** Does the complement of the circuit (everything *except* the circuit) fail at the task?
- **Minimality:** Can you remove any further component without degrading performance?
- **Generalization:** Does the circuit transfer to distribution shifts on the same task (e.g., different names in IOI)?

## Limitations

- Circuits found are task-specific and may not compose. The "induction head circuit" and the "IOI circuit" share components but were analyzed separately.
- The granularity problem: attention heads are not the atomic unit of computation — individual neurons within heads can have distinct roles. But going to neuron-level makes the graph intractably large.
- Superposition (Section 5) means the "true" features may not align with the model's native components at all, calling into question whether neuron/head-level circuits are the right abstraction.

---

# 4. Linear Probes

## Motivation

Sometimes we don't need a full mechanistic account — we just want to know: does the model *represent* this concept, and where? Probing answers this by training a small classifier on top of frozen model activations. Originally from [Alain and Bengio, 2016](https://arxiv.org/abs/1610.01644).

## Setup

Given:
- A dataset $\{(x_i, y_i)\}$ where $y_i$ is a label for the concept of interest (e.g., part-of-speech, sentiment, factual correctness, whether a chess move is legal).
- A frozen model $f$ producing activations $h^{(\ell)}(x_i)$ at layer $\ell$.

Train a probe $g_\ell : \mathbb{R}^d \to \mathcal{Y}$ on the representations:

$$\hat{y}_i = g_\ell\bigl(h^{(\ell)}(x_i)\bigr)$$

A **linear probe** restricts $g$ to be linear (or affine): $g_\ell(h) = W h + b$. The justification: if a linear classifier can extract the concept, the concept is represented as a direction in activation space, which is more likely to be "used" by the model (since attention and MLPs are approximately linear in many regimes).

## What Probes Can and Cannot Tell You

**Can tell you:**
- Whether a concept is *linearly accessible* at a given layer.
- How the representation of a concept develops across layers (probe accuracy as a function of depth).
- Comparative questions: does model A represent concept X better than model B?

**Cannot tell you:**
- Whether the model actually *uses* this representation for downstream computation. A probe succeeding is necessary but not sufficient — the information could be present but ignored. Some evidence that linear probes work even on randomly initialized models.
- Whether the concept is causally relevant to model behavior (use activation patching for that).

## The Probe Complexity Debate

If you use a powerful nonlinear probe (e.g., a 2-layer MLP), it might *learn* the concept from the activations rather than *detecting* it. The probe itself becomes a model. This is the "probing accuracy $\neq$ representation" critique ([Hewitt & Liang, 2019](https://arxiv.org/abs/1909.03368); [Ravichander et al., 2020](https://arxiv.org/abs/2005.00719)).

Mitigations:
- **Selectivity** ([Hewitt & Liang, 2019](https://arxiv.org/abs/1909.03368)): Compare probe accuracy to a control (random labels). If the probe does equally well on random labels, it's just memorizing.
- **Minimum Description Length probes** ([Voita & Titov, 2020](https://arxiv.org/abs/2003.12298)): Measure the compression achieved by the probe, not just accuracy.
- **Amnesic probing** ([Elazar et al., 2021](https://arxiv.org/abs/2006.00995)): Remove the probed information from the representation and measure downstream task degradation. This bridges probing with causal analysis.

## Notable Applications

- **Probing for world models:** [Li et al. (2023)](https://arxiv.org/abs/2210.13382) trained probes on Othello-GPT's activations and found that the model represents the board state as a linear feature, despite never seeing board representations during training. [Ivanitskiy et al. (2023)](https://arxiv.org/abs/2312.02566) found similar results for maze-solving transformers.
- **CCS — Contrast-Consistent Search** ([Burns et al., 2022](https://arxiv.org/abs/2212.03827)): Discovers latent knowledge (e.g., truth vs. falsehood) in an unsupervised way by finding directions in activation space where logically negated statements map to opposite sides. This is probing without labels.
- **Representation Engineering** ([Zou et al., 2023](https://arxiv.org/abs/2310.01405)): Find "concept directions" via probing, then steer model behavior by adding/subtracting these directions during inference.

## Practical Notes

- Always probe at multiple layers and positions. The "right" layer depends on the concept.
- Use logistic regression or a linear SVM for the probe itself. Avoid MLPs unless you have a specific reason and proper controls.
- Report control task baselines (selectivity).
- For sequence models, be careful about which token position you probe — information about a token may be stored at a *different* position (common in transformers due to attention).

---

# 5. Superposition

## Motivation

A recurring problem in mech interp is that the model's *features* (the things it represents and computes over) don't neatly correspond to individual neurons. A single neuron can respond to multiple unrelated concepts (**polysemanticity**), and a single concept can be distributed across many neurons. This is the **superposition hypothesis**: models represent more features than they have dimensions, using sparse, overlapping encodings.

## The Toy Model

[Elhage et al. (2022)](https://transformer-circuits.pub/2022/toy_model/index.html) studied superposition in a minimal setting: an autoencoder $f : \mathbb{R}^n \to \mathbb{R}^m \to \mathbb{R}^n$ where $m < n$, trained to reconstruct sparse inputs. Key findings:

- When input features are sparse enough, the model packs $n > m$ features into $m$ dimensions by using *almost-orthogonal* directions. This works because sparse features rarely co-activate, so the interference (cross-talk) is tolerable.
- The geometry depends on feature sparsity and importance: more important or less sparse features get dedicated dimensions; less important or sparser ones get squeezed into superposition.
- The transition from "dedicated dimensions" to "superposition" is phase-like — there are sharp thresholds.


The key idea here is that the "basis" the model uses for its internal representations

1. has no incentive to align with your canonical basis vectors (the neurons themselves)
2. can have more "directions" (features) than dimensions, by packing them in a way that minimizes interference, creating an "overcomplete" basis ("frame").

## Why Superposition Matters

Superposition is the fundamental obstacle to neuron-level interpretability:

- **Polysemantic neurons** are a symptom: a neuron that responds to both "academic citations" and "curly braces" isn't broken — it's a *compressed representation* where two features share a direction because they're sparse enough not to collide.
- **Circuits analysis at the neuron level may be misleading:** If the true features are directions in activation space that don't align with the neuron basis, neuron-level ablations conflate multiple features.
- **The "right" basis for interpretability is not the neuron basis** — it's whatever basis the model is actually using, which may have more directions than dimensions.

## Theoretical Framework

For $n$ unit vectors $\{v_i\}$ packed into $\mathbb{R}^m$ with $n > m$, the interference between features $i$ and $j$ is $|v_i \cdot v_j|$. The model wants to minimize total interference subject to representing all features. This connects to:

- **Johnson-Lindenstrauss:** $n$ nearly-orthogonal vectors can exist in $m$ dimensions if $m = O(\log n / \epsilon^2)$.
- **Compressed sensing:** Sparse signals can be recovered from low-dimensional measurements.
- **Frame theory:** The optimal packing geometries are known for small $n, m$ and relate to equiangular tight frames.


---

# 6. Sparse Autoencoders (SAEs)

## Motivation

If superposition means the model packs $n$ features into $m < n$ dimensions, the natural response is to *unpack* them. Sparse autoencoders learn a **dictionary** of features from the model's activations, projecting from $\mathbb{R}^m$ (activation space) to $\mathbb{R}^n$ (feature space, $n \gg m$) with a sparsity constraint that encourages each feature to fire rarely and represent a single concept.

## Architecture

An SAE applied to activations $x \in \mathbb{R}^m$ at some layer:

$$\text{Encode: } z = \text{ReLU}\bigl(W_{\text{enc}} (x - b_{\text{dec}}) + b_{\text{enc}}\bigr) \in \mathbb{R}^n$$
$$\text{Decode: } \hat{x} = W_{\text{dec}} \, z + b_{\text{dec}} \in \mathbb{R}^m$$

The loss is:

$$\mathcal{L} = \|x - \hat{x}\|_2^2 + \lambda \|z\|_1$$

The $L_1$ penalty on the hidden activations $z$ encourages sparsity: most features should be zero for any given input, and only a handful should activate.

## What "Features" Look Like

When trained on transformer residual stream activations, SAE features often correspond to interpretable concepts. Anthropic ([Bricken et al., 2023](https://transformer-circuits.pub/2023/monosemantic-features); [Templeton et al., 2024](https://transformer-circuits.pub/2024/scaling-monosemanticity/)) and others have found features for:

- Specific languages, topics, and named entities.
- Syntactic structures (e.g., "starts a parenthetical").
- Behavioral modes (e.g., "model is about to refuse").
- Code-specific patterns (e.g., "inside a for-loop body").

These are dramatically more monosemantic than individual neurons, which validates the superposition hypothesis and demonstrates that the "true features" are recoverable (at least partially).

"Automated Interpretability" is the technique of using LLMs to generate natural language descriptions of SAE features by showing the top-activating examples and asking the LLM to summarize. This is a powerful way to get human-understandable labels for features without manual inspection.

See https://www.neuronpedia.org/ to explore SAE features on a variety of models.

## Variants and Extensions

Lots of variants of SAEs exist and are an active area of research, but the key idea of "decomposing activations into a sparse set of features" is the core. Some variants:

- **TopK SAEs** ([Makhzani & Frey, 2013](https://arxiv.org/abs/1312.5663); [Gao et al., 2024](https://arxiv.org/abs/2406.04093)): Replace the $L_1$ penalty with a hard constraint that exactly $k$ features activate per input. This avoids the sparsity-reconstruction tradeoff and eliminates dead features.
- **Gated SAEs** ([Rajamanoharan et al., 2024](https://arxiv.org/abs/2404.16014)): Separate the "which features activate" decision from the "what magnitude" computation using a gating mechanism. Improves reconstruction at the same sparsity level.
- **Transcoders** (Bricken et al., 2024): Instead of autoencoding a layer's activations, learn to predict the *next* layer's activations from the current layer's features. This gives a feature-level description of *computation* rather than just representation. Applied to MLP blocks: input is the MLP input, output is the MLP output, and the hidden features describe what the MLP "does" in interpretable terms.
- **Attention SAEs / Multi-layer SAEs:** Apply SAEs to attention outputs, or train joint SAEs across multiple layers to capture features that evolve through the network.
- **Matrioshka SAEs** ([Bussmann et al., 2025](https://arxiv.org/abs/2503.17547)): Train a hierarchy of SAEs where each layer's features are autoencoded by the next layer, creating a multi-scale decomposition of the model's representations. Helps with feature splitting.

## Evaluating SAEs

Evaluation is an open problem. Current metrics:

- **Reconstruction quality:** $L_2$ error, explained variance, or downstream task performance with SAE-reconstructed activations substituted in.
- **Sparsity:** $L_0$ (number of active features per input).
- **Interpretability scores:** Automated (e.g., using an LLM to rate feature descriptions) or manual (human raters examine max-activating examples).
- **Feature absorption / splitting:** Does a single "true" concept get split across multiple SAE features (splitting)? Does a feature absorb a more specific sub-concept into a broader one (absorption)?
- **Downstream faithfulness:** If you intervene on an SAE feature (e.g., clamp it to zero), does the model's behavior change in the predicted way?

## Practical Notes

- The `SAELens` library (Bloom et al.) and `dictionary_learning` (Anthropic-adjacent) are the main codebases.
- https://www.neuronpedia.org/ is the best place to explore pre-trained SAEs and their features.
- Training SAEs is GPU-intensive: budget for many experiments varying dictionary size and $\lambda$.


## Limitations

- SAEs assume features are *linear directions*. If the model uses nonlinear manifolds or polytope-based representations, SAEs will give an incomplete picture. See [Engels et al., 2024](https://arxiv.org/abs/2405.14860).
- The dictionary is not unique — different random seeds and hyperparameters produce different feature decompositions. There is no ground truth.
- Scaling SAEs to frontier models is expensive!
- SAEs tell you what features exist, not how they're composed into computations. This is why transcoders and circuit analysis on SAE features are active research directions.



# 7. Parameter-based methods

SAEs help us understand activations, but intervening on them to influence model behavior is indirect: we have to modify the residual stream activations at every forward pass, and there is some evidence models can detect when this happens. Features that are not one-dimensionally linear ([Engels et al., 2024](https://arxiv.org/abs/2405.14860)) also pose a problem.

We might instead try to decompose the model *weights* instead of the activations into meaningful components. Two promising approaches:

- **Weight-sparse Transformers** Train a model with inherent sparsity in the weights to encourage interpretable components. See [Gao et al., 2025](https://arxiv.org/pdf/2511.13653)
- **Parameter Decomposition:** Given a trained model, decompose weight matrices into a sum of rank-1 components which activate sparsely. Kind of like taking the SVD of the weight matrix, but for each forward pass we construct a low rank approximation of the original weight matrix by selecting a subset of these components based on the input. See [Bushnaq et al., 2024](https://arxiv.org/pdf/2506.20790).

---

# Open Problems

- **Scaling:** Can we do mech interp on frontier models (100B+ parameters)? Current successes are mostly on GPT-2 scale. Automated methods (ACDC, SAEs) help but are not yet competitive with manual analysis for depth of understanding.
- **Compositionality:** We can find individual circuits, but how do they compose? How does the IOI circuit interact with the factual recall circuit when both are relevant?
- **Computational superposition:** We can (partially) resolve representational superposition with SAEs, but computation through superposition is poorly understood.
- **Detecting deception and latent knowledge:** The ultimate safety-relevant application. Can we tell if a model "knows" something it's not saying? CCS ([Burns et al., 2022](https://arxiv.org/abs/2212.03827)) is an early attempt, but it has known failure modes. Inner interpretability is uniquely positioned for this problem because, by definition, deceptive behavior cannot be detected from outputs alone.
- **Universality:** Do different models learn the same features and circuits? If yes, interpreting one model transfers to others. Early evidence is suggestive (convergent representations across architectures) but not conclusive.
- **Evaluation methodology:** We lack an "ImageNet for interpretability" — a standardized benchmark that measures whether interpretability tools are actually useful. Proposals like trojan detection challenges (Hubinger, 2021; Casper et al.) are promising but not yet adopted at scale.

---

# Recommended Reading

| Topic | Key Papers |
|-------|-----------|
| Residual stream / logit lens | [nostalgebraist (2020)](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens), [Belrose et al. (2023)](https://arxiv.org/abs/2303.08112) |
| Activation patching / causal tracing | [Meng et al. (2022)](https://arxiv.org/abs/2202.05262), [Goldowsky-Dill et al. (2023)](https://arxiv.org/abs/2304.05969) |
| Circuits | [Elhage et al. (2021)](https://transformer-circuits.pub/2021/framework/index.html), [Olsson et al. (2022)](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html), [Wang et al. (2023)](https://arxiv.org/abs/2211.00593) |
| ACDC | [Conmy et al. (2023)](https://arxiv.org/abs/2304.14997) |
| Probing | [Belinkov (2022)](https://arxiv.org/abs/2102.12452) survey, [Hewitt & Liang (2019)](https://arxiv.org/abs/1909.03368), [Burns et al. (2022)](https://arxiv.org/abs/2212.03827) |
| Superposition | [Elhage et al. (2022)](https://transformer-circuits.pub/2022/toy_model/index.html) "Toy Models of Superposition" |
| SAEs | [Bricken et al. (2023)](https://transformer-circuits.pub/2023/monosemantic-features), [Templeton et al. (2024)](https://transformer-circuits.pub/2024/scaling-monosemanticity/), [Cunningham et al. (2023)](https://arxiv.org/abs/2309.08600) |
| Transcoders | Bricken et al. (2024) |
| Evaluation / critique | [Casper, Räuker & Ho (2023)](https://arxiv.org/abs/2207.13243), [Bolukbasi et al. (2021)](https://arxiv.org/abs/2104.07143) |
| Representation engineering | [Zou et al. (2023)](https://arxiv.org/abs/2310.01405) |

# Tooling

- **[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)** — Neel Nanda's library for transformer mech interp. Hooks, caching, activation patching out of the box.
- **[SAELens](https://github.com/jbloomAus/SAELens)** — Training and analyzing SAEs on TransformerLens-compatible models.
- **[Neuronpedia](https://www.neuronpedia.org/)** — Web interface for browsing SAE features.