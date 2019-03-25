import numpy as np
import torch
from nltk.translate.bleu_score import sentence_bleu
import nltk

n_grams = [(1, 0, 0, 0),
            (0.5, 0.5, 0, 0),
            (0.33, 0.33, 0.33, 0),
            (0.25, 0.25, 0.25, 0.25)]


def vec_to_words(one_hots, idx_to_words):
    # 0 = *beg
    # 1 = *end
    # 2 = *unk
    # 3 = *-
    enc = np.argmax(one_hots.cpu().data.numpy()[:,:,0], axis=1).tolist()
    #enc = enc[:enc.index(1)]
    candidate = [idx_to_words[i] for i in enc]
    return candidate

def token(sen):
    word_list = nltk.word_tokenize(sen)[:-1]
    word_list = [w.lower() for w in word_list]
    return word_list

def to_word_bags(bags):
    return [token(sen) for sen in bags]

def bleu_score(ref, cand):
    return max([sentence_bleu(ref, cand, weights=w) for w in n_grams])
