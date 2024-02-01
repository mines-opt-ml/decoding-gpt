import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def analyze_vocab(
	tokenized_text: list[str],
	vocab_list: list[str],
	bins_freq: int = 100,
	bins_length: int = 20,
) -> Counter[str]:

	vocab_freq: Counter[str] = Counter(tokenized_text)
	vocab_size: int = len(vocab_freq)

	print(f"{vocab_size = }")
	print(f"{vocab_freq.most_common(10) = }")
	print(f"{vocab_freq.most_common()[-10:] = }")

	# get longest and shortest words
	vocab_length_sorted: list[str] = sorted(vocab_freq.keys(), key=len)
	print(f"{vocab_length_sorted[:10] = }")
	print(f"{vocab_length_sorted[-10:] = }")

	# plot histogram of word frequencies
	plt.figure(figsize=(10, 5))
	plt.hist(np.log10(np.array(list(vocab_freq.values()))), bins=bins_freq, log=True)
	plt.yscale('log')
	plt.xlabel("log(Word frequency)")
	plt.ylabel("Number of words")
	plt.show()

	# plot histogram of word lengths
	word_lengths = np.array([len(x) for x in vocab_list])
	plt.figure(figsize=(10, 5))
	plt.hist(word_lengths, bins=bins_length, log=True)
	plt.yscale('log')
	# plt.xscale('log')
	plt.xlabel("Word length")
	plt.ylabel("Number of words")
	plt.title(f"Word length distribution\nmean = {word_lengths.mean():.2f}, median = {np.median(word_lengths):.2f}")
	plt.show()

	return vocab_freq