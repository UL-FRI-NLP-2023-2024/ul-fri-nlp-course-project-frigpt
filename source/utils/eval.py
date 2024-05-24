from retrieval import Book, generate_embeddings, get_embedding_similarity
from character_chat import Character_chat
import numpy as np

def personality_eval(path_to_questions, chat, book, measure, num_questions=10):
    with open(path_to_questions, "r") as f:
        personality_questions = [line.strip()[1:-1] for line in f.readlines()]

    score = 0
    for j, question in enumerate(personality_questions[:min(len(personality_questions), num_questions)]):
        print(f"QUESTION {j + 1}: #########")
        print(question)
        print(f"ANSWER {j + 1}: #########")
        answer = chat.get_response(question)
        print(answer)
        answer_embedding = generate_embeddings(answer)
        for i, similar_line in enumerate(book.get_best_sentences(question, chat.character_name, k=3)):
            print(f"RELEVANT LINE {i + 1}")
            print(similar_line)
            test_embedding = generate_embeddings(similar_line)
            curr_score =  measure(test_embedding, answer_embedding)
            print(f"Line score: {curr_score}")
            score += measure(test_embedding, answer_embedding)
    return score, score / (3 * min(len(personality_questions), num_questions))

context_questions_path = "/content/drive/MyDrive/Colab Notebooks/ONJ PROJEKT/ONJ/data/eval/hamlet_questions.txt"

def contextual_eval(path_to_qa, chat, book, measure, num_questions=10):
    with open(path_to_qa, "r") as f:
        lines = [line.strip() for line in f.readlines()]
        questions = [line[4:] for line in lines if line.startswith("Q")]
        answers = [line[4:] for line in lines if line.startswith("A")]
    score = 0
    questions = questions[:min(len(questions), num_questions)]
    answers = answers[:min(len(questions), num_questions)]
    for i, (question, answer) in enumerate(zip(questions, answers)):
        print(f"QUESTION {i + 1}: #########")
        print(question)
        print(f"GT ANSWER {i + 1}: #########")
        print(answer)
        print(f"CHARACTER ANSWER {i + 1}: #########")
        character_answer = chat.get_response(question)
        print(character_answer)
        character_answer_embedding = generate_embeddings(character_answer)
        known_answer_embedding = generate_embeddings(answer)
        curr_score = measure(character_answer_embedding, known_answer_embedding)
        print(f'Score: {curr_score}')
        score += measure(character_answer_embedding, known_answer_embedding)
    return score, score / min(len(questions), num_questions)

if __name__ == "__main__":
        # HAMLET EVALUATION #######
        book_path = "/content/drive/MyDrive/Colab Notebooks/ONJ PROJEKT/ONJ/data/hamlet.txt"
        book = Book(book_path)

        hamlet_chat_simple = Character_chat(character_name="HAMLET", book=book, book_name="HAMLET")
        context_strategy = {"add_quotes": True, "quote_strategy": "random", "num_quotes": 10}
        hamlet_chat_with_random_quotes = Character_chat(character_name="HAMLET", book=book, book_name="HAMLET", context_strategy=context_strategy)
        context_strategy = {"add_quotes": True, "quote_strategy": "top", "num_quotes": 10}
        hamlet_chat_with_top_quotes = Character_chat(character_name="HAMLET", book=book, book_name="HAMLET", context_strategy=context_strategy)
        context_strategy = {"add_ner": True, "num_peaks": 2, "context_size": 1000}
        hamlet_chat_with_ner = Character_chat(character_name="HAMLET", book=book, book_name="HAMLET", context_strategy=context_strategy)

        measure = lambda x, y: np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y)) # cosine similarity
        personality_questions_path = "/content/drive/MyDrive/Colab Notebooks/ONJ PROJEKT/ONJ/data/eval/personality_questions.txt"

        score, normalized_score = personality_eval(personality_questions_path, hamlet_chat_simple, book, measure)
        print(score, normalized_score)
        print("################################################")
        score, normalized_score = personality_eval(personality_questions_path, hamlet_chat_with_random_quotes, book, measure)
        print(score, normalized_score)
        print("################################################")
        score, normalized_score = personality_eval(personality_questions_path, hamlet_chat_with_top_quotes, book, measure)
        print(score, normalized_score)
        print("################################################")
        score, normalized_score = personality_eval(personality_questions_path, hamlet_chat_with_ner, book, measure)
        print(score, normalized_score)
        print("################################################")

        # QUANTIZED RERUN HAMLET
        # 0.2630744838491589 (no added context)
        # 0.2735074797176382 (random k = 10)
        # 0.2769398040385271 (top k= 10)
        # 0.27141213871944114 (added context 2 peak, 1k)

        score, normalized_score = contextual_eval(context_questions_path, hamlet_chat_simple, book, measure)
        print(score, normalized_score)
        print("################################################")
        score, normalized_score = contextual_eval(context_questions_path, hamlet_chat_with_random_quotes, book, measure)
        print(score, normalized_score)
        print("################################################")
        score, normalized_score = contextual_eval(context_questions_path, hamlet_chat_with_top_quotes, book, measure)
        print(score, normalized_score)
        print("################################################")
        score, normalized_score = contextual_eval(context_questions_path, hamlet_chat_with_ner, book, measure)
        print(score, normalized_score)
        print("################################################")

        # QUANTIZED RERUN HAMLET
        # 0.41954472184181213 (no added context)
        # 0.434056288599968 (random k = 10)
        # 0.43148795992136 (top k= 10)
        # 0.4473472872376442 (added context 2 peak, 1k)
