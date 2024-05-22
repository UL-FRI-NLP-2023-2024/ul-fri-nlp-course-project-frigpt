import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def bert_LLM_entities(text, character):
	'''
		Finds characters in a book with BERT. Very slow and poor results.
		Example result for Frodo:
			{'entity': 'I-PER', 'score': 0.99613464, 'index': 164, 'word': 'Fr',   'start': 664, 'end': 666}
			{'entity': 'I-PER', 'score': 0.7718981,  'index': 165, 'word': '##od', 'start': 666, 'end': 668}
			{'entity': 'I-PER', 'score': 0.8152672,  'index': 166, 'word': '##o',  'start': 668, 'end': 669}
	'''
	from transformers import pipeline

	# Load the NER pipeline with a pre-trained BERT model
	ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

	cnt = 0
	# split the text into chunks of 2000 characters
	for i in range(len(text)//2000):
		text_to_process = text[i*2000:(i+1)*2000]
		# Perform NER on the example text
		print("Processing chunk", i, "/", len(text)//2000, end="\r")
		ner_results = ner_pipeline(text_to_process)

		# Print the entities recognized by the model
		for entity in ner_results:
			print(entity)
			if entity['word'] == character:
				cnt += 1
		print(i, "/", len(text)//2000, end="\r")

	print(f"Number of {character} entities:", cnt)


def plot_NER_results(text, character, top_k=10):
	'''
		Plots the Named Entity Recognition (NER) results for given book and character.
		Computes the Kernel Density Estimation (KDE) for the character locations.
		Plots the KDE with different sigmas and the peaks in the KDE.
	'''
	# Perform NER on the text
	char_locs, parsed_text = string_matching_NER(text, character)

	##################################
	# plot KDE with different sigmas #
	for sigma in [100, 200, 300]:
		kde, x_axis = get_KDE(char_locs, sigma=sigma, kernel_width=1000)
		plt.plot(kde, label="sigma="+str(sigma))
	plt.title("KDE with Gaussian Kernel")
	plt.xlabel("Position in book")
	plt.ylabel("Density")
	plt.xticks(np.arange(0, len(kde), 5000), x_axis[::5000])
	plt.legend()

	#######################
	# plot KDE with peaks #
	kde, x_axis = get_KDE(char_locs, sigma=200, kernel_width=1000)
	peaks, _ = find_peaks(kde, height=0.01)

	# retain only top_k highest peaks
	peaks = peaks[np.argsort(kde[peaks])[-top_k:]]

	# plot peaks
	plt.figure()
	plt.plot(kde, label="PDF")
	plt.vlines(peaks, 0, 0.08, color='r', linestyles='--', label="Peaks")
	plt.title("Peaks in PDF")
	plt.xlabel("Position in book")
	plt.ylabel("Density")
	plt.xticks(np.arange(0, len(kde), 5000), x_axis[::5000])
	plt.legend()
	plt.show()


def get_gauss_kernel(sigma, kernel_width):
	'''
		Returns a Gaussian kernel with given sigma and kernel width for KDE computation.
	'''
	kernel = np.exp(-np.arange(-kernel_width, kernel_width+1)**2/sigma**2)
	kernel = kernel/np.sum(kernel)
	return kernel


def get_KDE(char_locs, sigma, kernel_width):
	'''
		Computes the Kernel Density Estimation (KDE) for the given character locations.
		Returns the KDE and the x-axis values.
	'''
	# downscale the locations to 1/20th
	char_locs = (np.copy(char_locs) / 20).astype(int)

	func = np.zeros(char_locs[-1]+1)
	func[char_locs] = 1

	# get x-axis
	x_axis = np.linspace(0, 1, len(func), dtype=np.float32) * (char_locs[-1] * 20)	# 20 is the scaling factor
	x_axis = x_axis.astype(np.uint32)

	# get KDE with gaussian kernel
	kernel = get_gauss_kernel(sigma=sigma, kernel_width=kernel_width)
	kde = np.convolve(func, kernel, mode='same')
	return kde, x_axis


def string_matching_NER(text, character):
	'''
		Named Entity Recognition (NER) using string matching.
		Searches for the character in the text and saves the indices where the character is found.
		Text is preprocessed to remove unnecessary whitespaces and newlines.
	'''

	# convert character to lowercase
	character = character.capitalize()

	# remove unnecessary whitespaces and newlines
	text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
	text = " ".join(text.split())

	# get character indices in text
	search_idx = 0
	char_locs = []
	while search_idx != -1:
		search_idx = text.find(character, search_idx)
		if search_idx != -1:
			char_locs.append(search_idx)
			search_idx += 1

	return np.array(char_locs), text


def get_context(text, idx, context_size=1000):
	'''
		Returns the context around the given index in the text.
	'''
	# get the context around the index
	context = text[idx-context_size//2:idx+context_size//2]
	# remove the incomplete sentences
	context = ".".join(context.split(".")[1:-1])
	return context


def get_character_locations(text, character, sigma=200, kernel_width=1000, top_k=10):
	'''
		Performs Named Entity Recognition (NER) on the text to find the character locations.
		Computes the Kernel Density Estimation (KDE) to find the highest frequency locations of the character.
		Returns the character locations, peak locations, and the parsed text.
	'''

	# Perform NER on the text
	char_locs, parsed_text = string_matching_NER(text, character)

	# get KDE with gaussian kernel
	kde, x_axis = get_KDE(char_locs, sigma=sigma, kernel_width=kernel_width)

	# get peaks in KDE
	peaks, _ = find_peaks(kde, height=0.01)

	# retain only top_k highest peaks
	peaks = peaks[np.argsort(kde[peaks])[-top_k:]]

	return char_locs, x_axis[peaks], parsed_text


if __name__ == "__main__":


	# Read the LOTR book for NER
	character = "Frodo"
	with open('data/The_Lord_of_the_Rings-Book_one.txt', 'r', encoding="utf-8") as file:
		text = file.read()


	# Perform NER with KDE computation to get the highest frequency locations of the character
	all_locs, peak_locs, parsed_text = get_character_locations(text, character)

	# print the text around the first peak
	context = get_context(parsed_text, peak_locs[0], context_size=2000)
	print(context)

	# # plot the NER results with KDE and peaks
	# plot_NER_results(text, character)