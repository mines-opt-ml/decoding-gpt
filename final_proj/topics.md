---
title: Decoding GPT 2025: Final project ideas list
author: Michael Ivanitskiy
---

This document contains directions and (extremely biased and subjective) ideas for final projects. You are by no means limited to the things on this list, and in fact you should treat these as more of a starting point and not an actual list of projects to consider. Nor is everything on this list guaranteed to be a good idea. I've divided it up into sections based on the subject (Interpretability, Evals, etc), starting each section with some resources and then following that with ideas I personally have been wanting to try. Note that the sections are not fully distinct. This document may be updated in the future.

In general, [LessWrong](https://www.lesswrong.com) and the [Alignment Forum](https://www.alignmentforum.org) are good places to look for inspiration and to stay up to date.

# Interpretability

First and foremost -- anything Neel Nanda says makes a good project, is a good project for the class: https://www.lesswrong.com/posts/qGKq4G3HGRcSBc94C/mats-applications-research-directions

His guide is also a great place to start: https://www.neelnanda.io/mechanistic-interpretability/quickstart

## Sparse Autoencoders

Sparse Autoencoders can take a residual stream vector and decode it into a small set of features. Unlike logit lens or tuned lens, we do not impose the ontology of decoding directly into tokens: this has the advantage of giving a more faithful picture of what the model is using to think, but the disadvantage of the features not being already labeled.

https://www.neuronpedia.org/ lets us explore SAE features for a variety of models. They come with automatic explanations ("autointerp") for the features, but those are not very good.

Potential projects involving SAEs:

- Going through a lot of features from neuronpedia and attempting to explain them would make a final project, if you also test your hypotheses on what the features do: *intervene* on the model to up or downregulate the features and see what happens to the model's behavior.
- Can we represent the function of an MLP layer as something more interpretable by looking at it in SAE space? For example, we might try to train a decision tree to predict whether a certain feature post-MLP will be active depending on the input features (I have some starter code for this).
- Are there clever ways of optimizing the training of sparse autoencoders? It's fundamentally a dictionary learning problem, and there is much in that literature which may not yet be applied to training SAEs


## Exploring Abacus embeddings

Abacus embeddings (https://arxiv.org/abs/2405.17399) are a way of telling an LLM not just the position of a token in the sequence (positional embeddings), but the place value of a digit token in a number. It makes them way better at arithmetic. Possible lines of exploration might include:

- What is actually learned by the abacus embeddings? If we use learned abacus embeddings, does it construct something in the fourier basis (see interpretability notebook) like standard positional embeddings?
- Can we give abacus-style embeddings to a model to show the position of a token in a line, or the line number in which a token is? Does this make them better at ascii art or similar spatial-type tasks?
- What else is a model with abacus embeddings better at? If the abacus embeddings are applied to the point locations in an SVG, does the model get better at drawing SVGs?
- for temporal data, such as the time of day, year, or day of the week, can we add abacus embeddings for those? How will the learned features compare to the nonlinear features models learn for these features currently? (see https://arxiv.org/abs/2405.14860)

## Probing

A *linear probe* (https://arxiv.org/abs/1610.01644) is a technique for seeing if and how a model is using a certain concept. Unlike SAEs, they are a supervised method and it's thus easy to fool yourself with them. But, it's still possible to do an intervention on a vector picked up by a linear probe. A potential project along these lines might be picking a feature of interest which you can provide labelled data for, and seeing if linear probes on the model residual stream can pick it out, as well as studying the vectors they pick out.


## Attention

Some of my current research involves attempting to study the spatial structure of attention patterns. You can see the tool I've built for this here: https://github.com/mivanit/pattern-lens

If you think that you have some clever ideas of how to analyze these attention patterns, or you have a hypothesis for what a certain "kind" of attention pattern might be doing in a model, you can try intervening on heads with that pattern and seeing how the behavior of the model changes.

# Evals/Red Teaming


[METR](https://metr.org/) and [Apollo](https://www.apolloresearch.ai) are well known for their evals work. See the [Evals Starter Guide](https://www.apolloresearch.ai/blog/a-starter-guide-for-evals)


## Maze-solving evaluations

With collaborators I developed the [maze-dataset](https://github.com/understanding-search/maze-dataset) package, which allows for generating a variety of mazes and converting them to various text and visual representations. A possible evals project (and something I eventually plan to publish a paper on) would be to evaluate the behavior of pretrained language models (both open weights and API-only ones) on their ability to solve mazes, as well as analyze how they fail when they do. The existing codebase does most of the legwork of creating the data, so the code you would have to write would likely be mostly for feeding the data into LLMs and analyzing the results.


## Jailbreaking models/LLM whispering

Getting language models to produce interesting outputs is, in my opinion, a rare and undervalued skill. See the work of [@repligate on twitter](https://nitter.net/repligate), [websim.ai](https://websim.ai/), and [`LAZARUS_PROTOCOL`](https://chatgpt.com/g/g-VlpossZa7-lazarus-protocol). A project on this topic would be much more loosely defined, and grading it would be much more subjective. I would still expect the project to contain quantified analyses, but the nature of those would depend strongly on the exact nature of the behavior you produce.


## Video game benchmarks

Classical RL systems were often evaluated on their ability to play arcade games (see openai-gym). Today's LLMs are much harder to evaluate, but some interesting directions still involve video games -- see "Claude plays pokemon" and minecraft benchmarks such as [mcbench.ai](https://mcbench.ai/about).


# Alignment

Alignment is a broad field, but in the context of this class, I mean specifically techniques for controlling the behavior of LLM based AI systems. This would be more open ended, but might involve techniques for orchestrating many agents, or techniques for limiting the ability of an agent to cause harm in a specific environment.


