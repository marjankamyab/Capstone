from typing import List, Dict, Tuple
import preprocess, ingest, word_lstm, word_cnn
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import subprocess
import torch.nn.functional as F


# seed should essentially be initialized outside training to be consistent programwide
# specially for initializing oov vectors in the preprocess file
manualSeed = 42
random.seed(manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)
print("Seed num: " + str(manualSeed))


def train(train_data:ingest.Corpus, val_data:ingest.Corpus, test_data:ingest.Corpus,
          vocabulary, label2idx,
          embedding_dim:int, hidden_dim:int, cnn_layers:int,
          batch_size:int=1, initial_lr:float=0.015, decay_rate:float=0.0, epoch:int=1, crf=False,
          val_output_path="./output/output_files/val/val_output",
          test_output_path="./output/output_files/test/test_output") \
          -> None:

    val_dataset = prepare_dataset(val_data, vocabulary, label2idx)
    test_dataset = prepare_dataset(test_data, vocabulary, label2idx)
    batched_val = batch_data(val_dataset, vocabulary, label2idx, batch_size)
    batched_test = batch_data(test_dataset, vocabulary, label2idx, batch_size)

    #model initialization
    vocab_size = len(vocabulary)
    num_tags = len(labels)
    model = word_lstm.BiLSTM(vocab_size, embedding_dim, hidden_dim, num_tags, use_crf=crf)
    #model = word_cnn.CNN(vocab_size, embedding_dim, hidden_dim, cnn_layers, num_tags)
    model.embedding.weight = nn.Parameter(vocabulary.vectors)
    optimizer = optim.SGD(model.parameters(), lr=initial_lr, weight_decay=1e-8)
    loss_function = nn.NLLLoss(ignore_index=-1, size_average=False)

    #training
    for num in range(epoch):
        print("Epoch " + str(num) + ":")
        train_dataset = prepare_dataset(train_data, vocabulary, label2idx)
        random.shuffle(train_dataset) ##shuffle train data and then batch the shuffled data
        print("Shuffle: first input list: " + str(train_dataset[0][0]))
        batched_train = batch_data(train_dataset, vocabulary, label2idx, batch_size)
        model.train()
        optimizer = lr_decay(optimizer, num, decay_rate, initial_lr)
        batch = 0
        epoch_loss = 0.0
        for i, (sent,label) in enumerate(batched_train):
            optimizer.zero_grad()
            batch += 1
            outs = model(sent)

            if crf:
                mask = (label >= 0)
                loss = -(model.crf.forward(outs, label, mask=mask))
            else:
                score = model.softmax(outs)  # output shape: [number of tokens in the batch, 17]
                score = torch.flatten(score, end_dim=1)
                gold = torch.flatten(label)
                loss = loss_function(score, gold)

            loss.backward()
            epoch_loss += loss.detach().item()
            optimizer.step()

        #val and test evaluation between epochs
        ext = str(num) + "_" + str(batch)
        print(" epoch loss: " + str(epoch_loss))
        val_file = evaluate(batched_val, model, val_output_path, ext, crf)
        test_file = evaluate(batched_test, model, test_output_path, ext, crf)
        val_pl = pl2output(val_file)
        test_pl = pl2output(test_file)
        print('dev results: ', val_pl)
        print('test results: ', test_pl)
        print()


def evaluate(dataset:List, model, output_file, extension, crf):
    total_gold = []
    total_pred = []
    total_tokens = []
    for i,(sent_seq, gold_seq) in enumerate(dataset):
        mask = (gold_seq >= 0)
        pred = predict(model, sent_seq, mask, crf)

        gold = torch.masked_select(gold_seq, mask)
        batch_sents = torch.masked_select(sent_seq[0], mask)

        total_tokens.extend(batch_sents)
        total_gold.extend(gold)
        total_pred.extend(pred)

    tokens = [vocab_lst[token.item()] for token in total_tokens]
    golds = [label_lst[gold.item()] for gold in total_gold]
    preds = [label_lst[tag] for tag in total_pred]
    zipped = zip(tokens, golds, preds)

    output_path = output_file + str(extension) + '.txt'
    write_output(output_path, zipped)

    return output_path


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr/(1+decay_rate*epoch)
    print(" Learning rate is set as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def predict(model, sent:List[List], mask: torch.Tensor, crf:bool):
    output = model(sent)
    batch_predictions = []
    if crf:
        pred_lst = model.crf.decode(output, mask=mask)  ##output type: List[List[int]]
        for pred in pred_lst:
            batch_predictions.extend(pred)
    else:
        pred = torch.argmax(output, dim=2)  ##gathering token predictions for the entire batch
        pred = torch.masked_select(pred, mask)
        batch_predictions = [pred[i].detach().item() for i in range(len(pred))]

    return batch_predictions


def write_output(file, zipped_file):
    with open(file, 'w') as f:
        for token, gold, pred in zipped_file:
            f.write(token + " " + gold + " " + pred)
            f.write("\n")

def pl2output(file):
    with open(file, 'rb', 0) as f:
        pl2string = subprocess.check_output(['perl', 'conlleval.pl'], stdin=f, universal_newlines=True)
    return pl2string


def batch_data(data, vocabulary, label2idx, batch_size):
    indexed_sentences, indexed_labels = zip(*data)
    sent_batch = _batchify(indexed_sentences, batch_size)
    label_batch = _batchify(indexed_labels, batch_size)

    dataset = []
    for i, sentences in enumerate(sent_batch):
        labels = label_batch[i]
        new_sent_batch = []
        new_label_batch = []
        seq_lengths = torch.LongTensor(list(map(len, sentences)))
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)

        for j, sent in enumerate(sentences):
            padded_sent = _padding(sent, seq_lengths.max(), vocabulary.stoi["<pad>"])
            new_sent_batch.append(padded_sent)

        for k, label in enumerate(labels):
            padded_label = _padding(label, seq_lengths.max(), -1)
            new_label_batch.append(padded_label)

        new_sent_batch = torch.stack(new_sent_batch)
        new_label_batch = torch.stack(new_label_batch)
        new_sent_batch = new_sent_batch[perm_idx]
        new_label_batch = new_label_batch[perm_idx]
        dataset.append([(new_sent_batch, seq_lengths), new_label_batch])

    return dataset


def _batchify(lst, n):
    batches = []
    for i in range(0, len(lst), n):
        batches.append(lst[i: i+n])
    return batches

def _padding(seq:List[int], max_length:int, pad_value) -> torch.LongTensor:
    while(len(seq)< max_length):
        seq.append(pad_value)
    return torch.LongTensor(seq)

def _index_seq(mappings:Dict[str,int], sequence:Tuple[str, ...]) -> List[int]:
    return [mappings[preprocess._normalize_digits(element)] for element in sequence]  # shape = torch.Size([len(sentence)])

def prepare_dataset(corpus:ingest.Corpus, vocabulary, label2idx: Dict[str,int]) -> List:
    indexed_sents = [_index_seq(vocabulary.stoi, sent) for document in corpus for sent in document.sentences]
    indexed_labs = [_index_seq(label2idx, lab) for document in corpus for lab in document.labels]
    return list(zip(indexed_sents, indexed_labs))

if __name__ == "__main__":
    batch_size = 10
    hidden_units = 200
    epochs = 100
    lr = 0.015
    crf = False
    conll_train = ingest.load_conll('data/conll2003/en/BIOES/NE_only/train.bmes')
    conll_val = ingest.load_conll('data/conll2003/en/BIOES/NE_only/valid.bmes')
    conll_test = ingest.load_conll('data/conll2003/en/BIOES/NE_only/test.bmes')
    train_tokens = []
    total_instances = 0
    for document in conll_train:
        for i,sent in enumerate(document.sentences):
            train_tokens.extend(sent)
            total_instances += 1
    #training
    vocab, labels= preprocess.build_vocab(conll_train, conll_val, conll_test)
    label_lst = list(labels.keys())
    embedding_dim = vocab.vectors.size(1)
    vocab_lst = vocab.itos
    print("train number of instances: " + str(total_instances))
    print("train number of tokens: " + str(len(train_tokens)))
    print("crf: " + str(crf))
    print("batch size: " + str(batch_size))
    print("learning rate: " + str(lr))
    print("embedding dimension: " + str(embedding_dim))
    print("number of hidden units: " + str(hidden_units))
    print("number of epochs: " + str(epochs))
    print("tag scheme: " + str(labels))
    train(conll_train, conll_val, conll_test, vocab, labels, embedding_dim, hidden_units, 4,
          batch_size=batch_size, initial_lr=lr, decay_rate=0.05, epoch=epochs, crf=crf)