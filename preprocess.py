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


def build_vocab(train_corpus:ingest.Corpus, val_corpus:ingest.Corpus, test_corpus:ingest.Corpus) -> Tuple[Vocab, Dict[str,int]]:

    train_tokens, label_lst = _get_tokens(train_corpus)
    val_tokens, _ = _get_tokens(val_corpus)
    test_tokens, _ = _get_tokens(test_corpus)
    vocab_lst = train_tokens + val_tokens + test_tokens

    # initializing label dictionary
    label_counter = Counter(label_lst)
    labels = label_counter.most_common()
    label2idx = {lab:id for id,(lab,count) in enumerate(labels)}

    # initializing word vectors
    vocab_counter = Counter(vocab_lst)
    vocabulary = Vocab(vocab_counter, vectors=pretrained, specials_first=False)

    scale = np.sqrt(3.0/embedding_dim)
    perfect_match = 0
    case_match = 0
    no_match = 0
    for i,vocab in enumerate(vocabulary.stoi):
        if vocab in pretrained.stoi:
            vocabulary.vectors[vocabulary.stoi[vocab]] = pretrained.vectors[pretrained.stoi[vocab]]
            perfect_match+=1
        elif vocab.lower() in pretrained.stoi:
            vocabulary.vectors[vocabulary.stoi[vocab]] = pretrained.vectors[pretrained.stoi[vocab.lower()]]
            case_match+=1
        else:
            vocabulary.vectors[vocabulary.stoi[vocab]] = torch.tensor(np.random.uniform(-scale, scale, embedding_dim), requires_grad=True)
            no_match+=1

    print("vocabulary size: " + str(len(vocabulary.vectors)))
    print("perfect match: " + str(perfect_match)+ "\t" + "case match: " + str(case_match) + "\t" + "no match: " + str(no_match))

    return vocabulary, label2idx


def _normalize_digits(token):
    new_token = ""
    for char in token:
        if char.isdigit():
            new_token += '0'
        else:
            new_token += char
    return new_token


def _get_tokens(corpus: ingest.Corpus) -> Tuple[List[str], List[str]]:
    tokens = []
    labels = []
    for document in corpus:
        for i,sentence in enumerate(document.sentences):
            labels.extend(document.labels[i])
            for token in sentence:
                tokens.append(_normalize_digits(token))
    return tokens, labels


if __name__ == '__main__':
    conll_train = ingest.load_conll('data/conll2003/en/BIOES/NE_only/train.bmes')
    conll_val = ingest.load_conll('data/conll2003/en/BIOES/NE_only/valid.bmes')
    conll_test = ingest.load_conll('data/conll2003/en/BIOES/NE_only/test.bmes')
    vocab, labels= build_vocab(conll_train, conll_val, conll_test)






