# Project: Transformers from scratch

You've been working in groups both in class and at home implementing autoregressive transformers from scratch, and now it's finally time for you to submit your work!

# Assignment

Implement an autoregressive transformer based Large Language Model from scratch, and train it on a custom large text corpus.

Your implementation should include:

- implementations of multi-layer perceptrons and attention heads.
	- You can use `nn.ReLU`, `nn.Linear`, `F.softmax` or `nn.Parameter` for the building blocks, but not things like `nn.MultiheadAttention` or `nn.Transformer`
	- You can approximate $W_Q$ and $W_K$ as a single matrix $W_{QK}$ and likewise for $W_V$ and $W_O$. Implementing the more efficient split version with a reduced head dimension is optional and may be worth extra credit.
	- single-head attention is fine, but implementing multi-head attention is optional and is definitely worth extra credit.
- a basic word-based tokenizer. you may reuse code from the `intro-language-models.ipynb`
	- implementing Byte Pair Encoding (BPE) or other more sophisticated tokenization schemes is optional and may be worth extra credit.
- A full transformer class
    - dont just use use `nn.Transformer`, obviously
    - This transformer should **not** have hardcoded values. the `__init__` should take an instance of your `Config` dataclass for all hyperparameters (`n_layers`, `d_model`, `d_vocab`, `d_mlp`, `d_head`, etc.)
    - it should contain an embedding layer **with positional embeddings** (using `nn.Embedding` for both is allowed), some number of transformer blocks (configurable with `Config.n_layers`), and an unembedding layer (both tied and untied embeddings are allowed)
- a full training loop that trains your model on a large text corpus
	- your dataset corpus should be something interesting and ideally, funny.
	- you should implement your own training loop, and not use something like `Trainer` from HuggingFace or `fit` from PyTorch Lightning
	- batching is not required, but is recommended and may be worth extra credit.
	- **Recall: the transformer should take as input a tensor of integers (token ids, length `n`) and output a tensor of shape `n, d_vocab`**
- your transformer class should implement a `.generate(text: str, max_length: int)` method that generates text autoregressively from a given prompt. I should be able to put in any text and get a continuation of that text back out.
	- any kind of sampling method is fine (greedy, raw logits, top-k, top-p, etc.) and you can use `torch.multinomial` for sampling from the output distribution. Implementing more varied sampling methods is optional and may be worth extra credit.
- documentation of your code, contributions, and output. This can all go in one `README.md` file, or be in different files, but the `README.md` should explain where to find everything and how to run your code. Documentation must include:
	- **how to run your code.** Ideally, this should be no more than a few commands to run the training and generation scripts.
		- I **strongly recommend** using [uv](https://docs.astral.sh/uv/) for dependency management and having automated tests. **If your code does not run properly, you will lose points!**
		- having a **single** manual step to download the dataset is fine if automatically downloading it is not possible.
    - **Your results:** loss curves from your training process, and some example generations from your trained model and untrained model. Ideally, your trained model should generate text that is somewhat more coherent than the untrained model. If your model is not producing coherent text, try to explain why this might be happening. Your results may be pasted into a markdown document, or may be in a notebook with cell outputs committed to github.
    - **Writeup:** What design choices did you make, and which ones went well vs. which ones didnt? How big was your dataset? How many parameters did your model have? How long did it take to train? What were some challenges you faced, and how did you overcome them? What would you do differently if you had more time? What are you confused about still?
    - **Contributions:** Who did what in your group? Ideally, reference specific git commits.

# Submission format

Your submission on canvas should be a link to a github repository. This repository should either be public, or be shared with your instructor (github.com/mivanit). This repository should contain **ALL FILES** pertaining to your project. If you have a writeup (LaTeX or markdown preferred to docx or pdf), it should be in the repository.

It should be clear to me from reading just your `README.md`:


- how to run your code
	- **INCLUDING HOW TO INSTALL REQUIRED DEPENDENCIES**
- which files to find everything in
- what your results were
- who contributed what to the project

# Evaluation Criteria

- does the README properly document everything in the repo?
- does your code run?
- does your code implement the required architecture?
	- attention implemented correctly?
	- tokenization implemented correctly?
	- training loop and loss computation implemented correctly?
	- generation implemented correctly?
	- configurable hyperparameters?
- are your results presented?
	- loss curves?
	- example generations from trained vs untrained model?
	- analysis of results?
	- writeup of design choices, challenges, and future directions?
	- contributions of all group members clearly documented?

