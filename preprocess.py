## TO DO: turn this program into data class
from typing import List, Tuple, Dict
from collections import Counter
from torchtext.vocab import Vocab, GloVe
from numpy import random
import numpy as np
import torch
import ingest


print("Loading embeddings...")
pretrained = GloVe(name='6B', dim=50)
print(pretrained)
embedding_dim = pretrained.vectors.size(1)


def build_vocab(train_corpus:ingest.Corpus, val_corpus:ingest.Corpus, test_corpus:ingest.Corpus) -> Tuple[Vocab, Dict[str,int], Dict[str,int]]:
    train_tokens, train_chars, label_lst = _get_alphabet(train_corpus)
    val_tokens, val_chars, _ = _get_alphabet(val_corpus)
    test_tokens, test_chars, _ = _get_alphabet(test_corpus)
    vocab_lst = train_tokens + val_tokens + test_tokens
    char_lst = train_chars + val_chars + test_chars


    # initializing label dictionary
    label_counter = Counter(label_lst)
    labels = label_counter.most_common()
    label2idx = {lab:id for id,(lab,count) in enumerate(labels)}

    # initializing character dictionary
    char_counter = Counter(char_lst)
    chars = char_counter.most_common()
    char2idx = {char:id for id,(char,count) in enumerate(chars)}

    # initializing word vectors
    vocab_counter = Counter(vocab_lst)
    vocabulary = Vocab(vocab_counter, vectors=pretrained, specials_first=False)

    scale = np.sqrt(3.0/embedding_dim)
    perfect_match = 0
    case_match = 0
    no_match = 0
    for i,vocab in enumerate(vocabulary.stoi):
        if vocab in pretrained.stoi:
            perfect_match+=1
        elif vocab.lower() in pretrained.stoi:
            vocabulary.vectors[vocabulary.stoi[vocab]] = pretrained.vectors[pretrained.stoi[vocab.lower()]]
            case_match+=1
        else:
            vocabulary.vectors[vocabulary.stoi[vocab]] = torch.tensor(np.random.uniform(-scale, scale, embedding_dim))
            no_match+=1

    print("vocabulary size: " + str(len(vocabulary.vectors)))
    print("perfect match: " + str(perfect_match)+ "\t" + "case match: " + str(case_match) + "\t" + "no match: " + str(no_match))

    return vocabulary, char2idx, label2idx


def _normalize_digits(token):
    new_token = ""
    for char in token:
        if char.isdigit():
            new_token += '0'
        else:
            new_token += char
    return new_token


def _get_alphabet(corpus: ingest.Corpus) -> Tuple[List[str], List[str]]:
    tokens = []
    chars = []
    labels = []
    for document in corpus:
        for i,sentence in enumerate(document.sentences):
            labels.extend(document.labels[i])
            for token in sentence:
                tokens.append(_normalize_digits(token))
                for char in token:
                    chars.append(char)

    return tokens, chars, labels


if __name__ == '__main__':
    conll_train = ingest.load_conll('data/conll2003/en/BIOES/NE_only/train.bmes')
    conll_val = ingest.load_conll('data/conll2003/en/BIOES/NE_only/valid.bmes')
    conll_test = ingest.load_conll('data/conll2003/en/BIOES/NE_only/test.bmes')
    vocab, chars, labels= build_vocab(conll_train, conll_val, conll_test)






