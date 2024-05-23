
from model import create_text_generator
import NER
import numpy as np

class Character_chat:
    def __init__(self, character_name, book, book_name, context_strategy={}):
        self.text_generator = create_text_generator()
        self.context_strategy = {"add_quotes": False, "add_ner": False, **context_strategy}
        print(self.context_strategy)
        # self.add_quotes = add_quotes
        # self.quote_strategy = quote_strategy
        self.character_name = character_name
        # self.num_quotes = num_quotes
        self.lines = book.get_character_lines(character_name)
        self.book = book

        if self.context_strategy["add_ner"]:
            corpus_pth = self.book.filepath
            # print(corpus_pth)
            with open(corpus_pth, "r", encoding="utf-8") as f:
              corpus = f.read()
            all_locs, peak_locs, parsed_corpus = NER.get_character_locations(corpus, self.character_name)
            self.ner_lines = []
            for i in range(min(self.context_strategy["num_peaks"], len(peak_locs))):
                relevant_context = NER.get_context(parsed_corpus, peak_locs[i], context_size=context_strategy["context_size"])
                self.ner_lines.append(relevant_context)
            print(self.ner_lines)

        self.prompt = f"""<s>[INST] <<SYS>>
You are {character_name}, a fictional character from {book_name}.
When you are asked a question or told something you must only respond in character. Try to respond in a single sentence.
Do not respond with a question.
<</SYS>>"""
        self.prompt += """{character_quotes}
        """
        self.prompt += """The person you are talking to says the following: {user_input}
[/INST]"""

    def get_response(self, user_input):
        if self.context_strategy["add_quotes"]:
            if self.context_strategy["quote_strategy"] == "random":
                character_quotes = "\n".join([f"{self.character_name}: " + line for line in np.random.choice(self.lines, self.context_strategy["num_quotes"])])
            elif self.context_strategy["quote_strategy"] == "top":
                character_quotes = "\n".join([f"{self.character_name}: " + line for line in self.book.get_best_sentences(user_input, self.character_name, k=(3 + self.context_strategy["num_quotes"]))[3:]])
            character_quotes = f"""You are given some examples of the character's lines:
{character_quotes}"""
            # input = self.prompt.format(user_input=user_input, character_quotes=character_quotes)
        elif self.context_strategy["add_ner"]:
            character_quotes = "\n".join(["PART OF BOOK: " + line + "..." for line in self.ner_lines])
            character_quotes = f"""You are given some parts of the book that are relevant for the character:
{character_quotes}"""
        else:
            character_quotes = ""
        input = self.prompt.format(user_input=user_input, character_quotes=character_quotes)
        # print(input)
        output = self.text_generator(input)
        # print(output)
        res = output[0]['generated_text'].split("[/INST]")[-1].strip()
        # print(res)
        return res
