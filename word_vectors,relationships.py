import gensim.downloader as api
import numpy as np
word_vectors=api.load('word2vec-google-news-300')
def vector_arithematic(word1,word2,word3):
    result_vector=word_vectors[word1]-word_vectors[word2]+word_vectors[word3]
    similiar_words=word_vectors.most_similiar([result_vector],topn=5)
    return similiar_words
print("Result of 'king-man+woman'")
print(vector_arithematic("king","man","woman"))
def find_similiar_words(word):
    return word_vectors.most_similiar(word,topn=5)
print("Similiar words for: 'computer'")
print(find_similiar_words("conputer"))

# python -m spacy download en_core_web_md

