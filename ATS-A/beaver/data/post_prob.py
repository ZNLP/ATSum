# -*- coding: utf-8 -*-

import sys
import pickle
import numpy as np
import torch
import os
from typing import List


wpdict_path = '/path/to/wordalignment/c2e.pkl' # wordalignment
src_vocab = '/path/to/sourcevocab/vocab.cn.10w'
tgt_vocab = '/path/to/targetvocab/vocab.en.4w'
src_special = ['<unk>', '<pad>']
tgt_special = ['<unk>', '<pad>', '<bos>', '<eos>']

with open(wpdict_path, 'rb') as file:
    wpdict = pickle.load(file)


def readlines(path):
    with open(path, encoding="utf-8") as file:
        lines = file.readlines()
    return [line.strip() for line in lines]


class Vocab(object):
    def __init__(self, words: List[str], specials: List[str]):
        self.itos = specials + words
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)


def get_wp(wpdict, src_vocab_path, tgt_vocab_path):
    src_words = readlines(src_vocab_path)
    tgt_words = readlines(tgt_vocab_path)
    svocab = Vocab(src_words, src_special)
    tvocab = Vocab(tgt_words, tgt_special)
    wplist = np.array([[1e-12 for i in range(len(tvocab))] for j in range(len(svocab))])
    for idx, sw in enumerate(src_words):
        trueidx = idx + 2
        if sw not in wpdict:
            continue
        wptuple = wpdict[sw]
        for tgt_word, prob in wptuple:
            tgtidx = tvocab.stoi[tgt_word]
            wplist[trueidx, tgtidx] = prob
    wplist = wplist / np.sum(wplist, axis=1, keepdims=True)

    wplist_bool = np.choose(wplist < 0.05, (1, 0))
    wplist_float = np.choose(wplist < 0.05, (1.0, 0.0))
    wpsum = np.sum(wplist_bool, axis=1)
    for i in range(len(wplist_float)):
        if wpsum[i] > 0:
            wplist_float[i] = wplist_float[i] / wpsum[i]
        else:
            continue
    return wplist_float

def get_wplist():
    return get_wp(wpdict, src_vocab, tgt_vocab)

def get_prob_idx():
    load_path = os.path.abspath(os.path.dirname(__file__))
    with open(load_path+'/savefile/prob.pkl', 'rb') as file1, open(load_path+'/savefile/idx.pkl', 'rb') as file2:
        problist = pickle.load(file1)
        idxlist = pickle.load(file2)
    return [problist, idxlist]

if __name__ == '__main__':
    wplist = torch.from_numpy(get_wplist())
    srcword_idx, srcword_prob = [], []
    prob_matrix, idx_matrix = wplist.topk(5, dim=1)

    for i in range(len(prob_matrix)):
        # norm_prob = prob_matrix[i] / prob_matrix[i].sum()
        norm_prob = prob_matrix[i]
        norm_prob = norm_prob.tolist()
        srcword_prob.append(norm_prob)
        srcword_idx.append(idx_matrix[i].tolist())

    save_path = os.path.abspath(os.path.dirname(__file__))
    with open(save_path + '/savefile/prob.pkl', 'wb') as file1, open(save_path+ '/savefile/idx.pkl', 'wb') as file2:
        pickle.dump(srcword_prob, file1, 2)
        pickle.dump(srcword_idx, file2, 2)
