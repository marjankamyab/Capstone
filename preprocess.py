from typing import List, Tuple, Dict
from collections import Counter
from torchtext.vocab import Vocab, GloVe
import torch
import numpy as np
from numpy import random
import ingest

print("Loading embeddings...")
pretrained = GloVe(name='6B', dim=50)
print(pretrained)
embedding_dim = pretrained.vectors.size(1)
print("embedding dimension: " + str(embedding_dim))

def build_vocab(train_corpus:ingest.Corpus, val_corpus:ingest.Corpus, test_corpus:ingest.Corpus) -> Tuple[Vocab, Dict[str,int]]:
    train_tokens, label_lst = _get_tokens(train_corpus)
    val_tokens, _ = _get_tokens(val_corpus)
    test_tokens, _ = _get_tokens(test_corpus)
    vocab_lst = train_tokens + val_tokens + test_tokens
    # label dictionary
    label_counter = Counter(label_lst)
    labels = label_counter.most_common()
    label2idx = {lab:id for id,(lab,count) in enumerate(labels)}
    label2idx['<pad>'] = len(labels)
    # initializing word vectors
    vocab_counter = Counter(vocab_lst)
    vocabulary = Vocab(vocab_counter, vectors=pretrained, specials_first=False)
    scale = np.sqrt(3/embedding_dim)
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
            vocabulary.vectors[vocabulary.stoi[vocab]] = torch.from_numpy(np.random.uniform(-scale, scale, embedding_dim))
            no_match+=1
    print("vocabulary size: " + str(len(vocabulary)))
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
                token = _normalize_digits(token)
                tokens.append(token)
    return tokens, labels

def _index_seq(mappings:Dict[str,int], sequence:Tuple[str, ...], label) -> torch.LongTensor:
    if label:
        return [mappings[_normalize_digits(element)] for element in sequence]  # shape = torch.Size([len(sentence)])
    return [mappings[_normalize_digits(element)] for element in sequence]

def _padding(mapping:Dict[str,int], seq:List[int], max_length:int) -> torch.LongTensor:
    while(len(seq)< max_length):
        seq.append(mapping['<pad>'])
    return torch.LongTensor(seq)

def _batchify(lst, n):
    batches = []
    for i in range(0, len(lst), n):
        batches.append(lst[i: i+n])
    return batches

''''
devides the data into several batches. 
batch size is the same but sentence length varies over the batches
the sentence length for each batch is set to the length of the sentence with maximum length
'''
def prepare_dataset(corpus:ingest.Corpus, vocabulary: Vocab, label2idx: Dict[str,int], batch_size:int=1) -> Tuple[torch.Tensor,torch.Tensor]:
    dataset= []
    indexed_sents = [_index_seq(vocabulary.stoi, sent, False) for document in corpus for sent in document.sentences]
    indexed_labs = [_index_seq(label2idx, lab, True) for document in corpus for lab in document.labels]
    lst = list(zip(indexed_sents, indexed_labs))
    random.shuffle(lst)
    indexed_sentences, indexed_labels = zip(*lst)
    #print([vocabulary.itos[i] for i in indexed_sentences[0]])
    #print([list(label2idx.keys())[i] for i in indexed_labels[0]])
    sent_batch = _batchify(indexed_sentences, batch_size)
    label_batch = _batchify(indexed_labels, batch_size)
    total_instances = 0
    for i, sentences in enumerate(sent_batch):
        labels = label_batch[i]
        new_sent_batch = []
        new_label_batch = []
        seq_lengths = torch.LongTensor(list(map(len, sentences)))
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        for j,sent in enumerate(sentences):
            vectorized_sent = _padding(vocabulary.stoi, sent, seq_lengths.max())
            new_sent_batch.append(vectorized_sent)
            total_instances += 1
        for k, label in enumerate(labels):
            vectorized_label = _padding(label2idx, label, seq_lengths.max())
            new_label_batch.append(vectorized_label)
        new_sent_batch = torch.stack(new_sent_batch)
        new_label_batch = torch.stack(new_label_batch)
        new_sent_batch = new_sent_batch[perm_idx]
        new_label_batch = new_label_batch[perm_idx]
        dataset.append([[new_sent_batch, seq_lengths], new_label_batch])
    return dataset



if __name__ == '__main__':
    conll_train = ingest.load_conll('data/conll2003/en/BIOES/NE_only/train.bmes')
    conll_val = ingest.load_conll('data/conll2003/en/BIOES/NE_only/valid.bmes')
    conll_test = ingest.load_conll('data/conll2003/en/BIOES/NE_only/test.bmes')
    vocab, labels= build_vocab(conll_train, conll_val, conll_test)
    prepare_dataset(conll_train, vocab, labels, batch_size=10)






