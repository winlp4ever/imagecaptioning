import numpy as np
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

n_grams = [(1, 0, 0, 0),
            (0.5, 0.5, 0, 0),
            (0.33, 0.33, 0.33, 0),
            (0.25, 0.25, 0.25, 0.25)]


def vec_to_words(one_hots, idx_to_words):
    enc = np.argmax(one_hots.cpu().data.numpy()[0], axis=0).tolist()
    candidate = [idx_to_words[i] for i in enc]
    if '*end' in candidate:
        end = candidate.index('*end')
    else:
        end = len(candidate)
    return candidate[1: end]

def token(sen):
    word_list = nltk.word_tokenize(sen)[:-1]
    word_list = [w.lower() for w in word_list]
    return word_list

def to_word_bags(bags):
    return [token(sen) for sen in bags]

def bleu_score(ref, cand, n: int=1):
    #return max([sentence_bleu(ref, cand, weights=w) for w in n_grams])
    assert 1 <= n <= 4;
    chencherry = SmoothingFunction()
    return sentence_bleu(ref, cand, weights=n_grams[n - 1], smoothing_function=chencherry.method5)
