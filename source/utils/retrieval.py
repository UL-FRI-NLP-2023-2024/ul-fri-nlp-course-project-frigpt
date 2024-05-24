import pandas as pd
import numpy as np
import torch
from data import extract_lines_from_play

from sentence_transformers import SentenceTransformer

def get_device():
    """
    Returns the appropriate device for the current environment
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SentenceTransformer("all-MiniLM-L6-v2", device=get_device())

class Book:
    def __init__(self, filepath):
        """
        Initializes a Book object
        """
        lines = extract_lines_from_play(filepath)
        self.lines = lines
        self.characters = self.get_characters()
        self.embeddings = []
        self.filepath = filepath

        for character in self.characters:
            character_lines = self.lines[self.lines["Character"] == character]
            character_line_ids = character_lines["Line Number"].values
            character_lines_with_context = []
            for line_number in character_line_ids:
                context_lines = self.lines[(self.lines["Line Number"] >= max(line_number - 1, 0)) & (self.lines["Line Number"] <= line_number)]
                character_lines_with_context.append(" ".join(context_lines["Line"].values))

            character_embeddings = generate_embeddings(character_lines_with_context)
            for i, embedding in enumerate(character_embeddings):
                self.embeddings.append({"Character": character, "Line": character_lines.iloc[i]["Line"], "Embedding": embedding})

        self.lines = pd.DataFrame(self.embeddings)

    def get_characters(self):
        """
        Returns the characters in the book
        """
        return self.lines["Character"].unique()

    def get_character_lines(self, character):
        """
        Returns the lines spoken by a character
        """
        return self.lines[self.lines["Character"] == character]["Line"].values

    def get_best_sentences(self, sentence, character, k=5):
        """
        Returns the k most similar lines to the given sentence
        """
        character_lines = self.lines[self.lines["Character"] == character]
        character_sentences = character_lines["Line"].values
        character_embeddings = character_lines["Embedding"].values

        query_embedding = generate_embeddings([sentence])[0]

        similarities = [get_embedding_similarity(query_embedding, embedding) for embedding in character_embeddings]

        indices = np.argsort(similarities)[::-1][:k]
        return [character_sentences[i] for i in indices]

def character_lines(character, lines):
    """
    Returns the lines of text spoken by a character
    """
    ch_lines = lines[lines["Character"] == character]
    return ch_lines["Line"].values

def generate_embeddings(sentences):
    """
    Generates an embedding for a given text
    """
    # Load the model
    embeddings = model.encode(sentences)
    return embeddings

def get_embedding_similarity(embedding1, embedding2):
    """
    Returns the cosine similarity between two embeddings
    """
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    
if __name__ == '__main__':
    book = Book("data/hamlet.txt")
    sentences = book.get_best_sentences("What do you think about Ophelia?", "HAMLET")
    for i, sentence in enumerate(sentences):
        print(f"{i + 1}: {sentence}")