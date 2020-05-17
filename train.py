import preprocess, ingest, word_char_lstm, word_char_cnn
from typing import List, Dict, Tuple
from torchtext.vocab import Vocab
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import subprocess
torch.set_printoptions(edgeitems=500)



# seed should essentially be initialized outside training to be consistent programwide
# specially for initializing oov vectors in the preprocess file
manualSeed = 42
random.seed(manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)
print("Seed num: " + str(manualSeed))


def train(train_data:ingest.Corpus, val_data:ingest.Corpus, test_data:ingest.Corpus,
          vocabulary:Vocab, alphabet:Dict[str,int], label2idx:Dict[str,int],
          word_embedding_dim:int, char_embedding_dim:int,
          hidden_dim:int, char_hidden_dim:int, cnn_layers:int,
          batch_size:int=1, epoch:int=1,
          dropout:float=0.0, initial_lr:float=0.015, decay_rate:float=0.0,
          mode="LSTM", crf:bool=False, char:bool=False,
          val_output_path="./output/output_files/val/val_output",
          test_output_path="./output/output_files/test/test_output") \
          -> None:


    val_dataset = prepare_dataset(val_data, vocabulary, alphabet, label2idx)
    test_dataset = prepare_dataset(test_data, vocabulary, alphabet, label2idx)
    batched_val = batch_data(val_dataset, vocabulary, batch_size)
    batched_test = batch_data(test_dataset, vocabulary, batch_size)

    #model initialization
    num_tags = len(labels)
    vocab_size = len(vocabulary)
    alphabet_size = len(alphabet)

    if mode.lower() == "lstm":
        model = word_char_lstm.BiLSTM(vocab_size, alphabet_size, word_embedding_dim, char_embedding_dim, hidden_dim, char_hidden_dim, num_tags, dropout, use_crf=crf, use_char=char)
    else:
        model = word_char_cnn.CNN(vocab_size, alphabet_size, word_embedding_dim, char_embedding_dim, hidden_dim, char_hidden_dim, cnn_layers, num_tags, dropout, use_crf=crf, use_char=char)

    model.word_embedding.weight = nn.Parameter(vocabulary.vectors, requires_grad=True)
    optimizer = optim.SGD(model.parameters(), lr=initial_lr, weight_decay=1e-8)
    loss_function = nn.NLLLoss(ignore_index=-1, reduction="sum")
    softmax = nn.LogSoftmax(2)

    #training
    for num in range(epoch):
        model.train()
        print("Epoch " + str(num) + ":")
        train_dataset = prepare_dataset(train_data, vocabulary, alphabet, label2idx)
        random.shuffle(train_dataset) ##shuffle train data and then batch the shuffled data
        print("Shuffle: first input list:", train_dataset[0][0])
        batched_train = batch_data(train_dataset, vocabulary, batch_size)
        optimizer = lr_decay(optimizer, num, decay_rate, initial_lr)

        batch = 0
        epoch_loss = 0.0
        for i, (sent, chars, label) in enumerate(batched_train):
            optimizer.zero_grad()
            batch_length = sent[0].size(0)
            batch += 1
            outs = model(sent, chars)

            #calculate loss
            if crf:
                mask = (label>=0)
                loss = -(model.crf.forward(outs, label, mask=mask))
            else:
                score = softmax(outs)  # output shape: [number of tokens in the batch, 17]
                score = torch.flatten(score, end_dim=1)
                gold = torch.flatten(label)
                loss = loss_function(score, gold)
                if mode.lower()!= "lstm":
                    loss /= batch_length

            loss.backward()
            epoch_loss += loss.detach().item()
            optimizer.step()

        #val and test evaluation between epochs
        model.eval()
        ext = str(num) + "_" + str(batch)
        print(" epoch loss:", epoch_loss)
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
    for i,(sent_seq, char_seq, gold_seq) in enumerate(dataset):
        mask = (gold_seq>=0)
        pred = predict(model, sent_seq, char_seq, mask, crf)

        gold = torch.masked_select(gold_seq, mask)
        batch_sents = torch.masked_select(sent_seq[0], mask)

        total_tokens.extend(batch_sents)
        total_gold.extend(gold)
        total_pred.extend(pred)

    tokens = [vocab_lst[token.item()] for token in total_tokens]
    golds = [label_lst[gold.item()] for gold in total_gold]
    preds = [label_lst[tag] for tag in total_pred]

    fixed_preds = []
    fixed_golds = []
    positive=0
    for i, pred in enumerate(preds):
        gold = golds[i]
        if pred == gold:
            positive+=1

        if pred.startswith('S'):
            pred = 'B'+pred[1:]
        elif pred.startswith('E'):
            pred = 'I'+pred[1:]

        if gold.startswith('S'):
            gold = 'B'+gold[1:]
        elif gold.startswith('E'):
            gold = 'I'+gold[1:]

        fixed_preds.append(pred)
        fixed_golds.append(gold)

    accuracy = positive/len(preds)
    print("accuracy:", accuracy)

    #zipped = zip(tokens, golds, preds)
    zipped = zip(tokens, fixed_golds, fixed_preds)

    output_path = output_file + str(extension) + '.txt'
    write_output(output_path, zipped)

    return output_path


def lr_decay(optimizer, epoch:int, decay_rate:float, init_lr:float):
    lr = init_lr/(1+decay_rate*epoch)
    print(" Learning rate is set as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def predict(model, sent:Tuple[torch.tensor], chars:Tuple[torch.tensor], mask: torch.Tensor, crf:bool):
    output = model(sent, chars)
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


def batch_data(data, vocabulary, batch_size):
    indexed_sentences, indexed_words, indexed_labels = zip(*data)
    sent_batch = _batchify(indexed_sentences, batch_size)
    word_batch = _batchify(indexed_words, batch_size)
    label_batch = _batchify(indexed_labels, batch_size)

    dataset = []
    for i, sentences in enumerate(sent_batch):
        labels = label_batch[i]
        sent_words = word_batch[i]

        new_sent_batch = []
        new_word_batch = []
        new_label_batch = []

        seq_lengths = torch.LongTensor(list(map(len, sentences)))
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)

        for i, sent in enumerate(sentences):
            padded_sent = torch.LongTensor(_padding(sent, seq_lengths.max(), vocabulary.stoi["<pad>"]))
            new_sent_batch.append(padded_sent)

            padded_sent_words = _padding(sent_words[i], seq_lengths.max(), [0]) #for chars
            new_word_batch.append(padded_sent_words) #for chars

        for label in labels:
            padded_label = torch.LongTensor(_padding(label, seq_lengths.max(), -1))
            new_label_batch.append(padded_label)

        new_sent_batch = torch.stack(new_sent_batch)
        new_label_batch = torch.stack(new_label_batch)
        new_sent_batch = new_sent_batch[perm_idx]
        new_label_batch = new_label_batch[perm_idx]

        #dealing with chars
        batch_words = [word for word_lst in new_word_batch for word in word_lst]
        word_lengths = torch.LongTensor(list(map(len, batch_words)))
        word_lengths = word_lengths.view(len(sentences), seq_lengths.max())
        char_seq_tensor = torch.zeros((batch_size, seq_lengths.max(), word_lengths.max()), requires_grad=True).long()
        for j, word_lst in enumerate(sent_words):
            for k, char_lst in enumerate(word_lst):
                padded_char_lst = torch.LongTensor(_padding(char_lst, word_lengths.max(), 0))
                char_seq_tensor[j, k, :] = padded_char_lst
        char_seq_tensor = char_seq_tensor[perm_idx]
        flat_char_seq = torch.flatten(char_seq_tensor, end_dim=1)
        char_seq_lengths = word_lengths[perm_idx].flatten()
        char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
        new_char_batch = flat_char_seq[char_perm_idx]
        _, char_seq_recover = char_perm_idx.sort(0, descending=False)

        dataset.append([(new_sent_batch, seq_lengths), (new_char_batch, char_seq_recover), new_label_batch])

    return dataset

def _padding(seq:List, max_length:int, pad_value:int) -> List:
    while(len(seq)< max_length):
        seq.append(pad_value)
    return seq

def _batchify(lst:List, n:int):
    batches = []
    for i in range(0, len(lst), n):
        batches.append(lst[i: i+n])
    return batches

def _index_seq(mappings:Dict[str,int], sequence:Tuple[str, ...]) -> List[int]:
    return [mappings[preprocess._normalize_digits(element)] for element in sequence]  # shape = torch.Size([len(sentence)])

def prepare_dataset(corpus:ingest.Corpus, vocabulary, alphabet:Dict[str,int], label2idx:Dict[str,int]) -> List:
    indexed_sents = []
    indexed_labs = []
    indexed_word_chars = []
    for document in corpus:
        for i, sent in enumerate(document.sentences):
            lab = document.labels[i]
            sent_words = _index_seq(vocabulary.stoi, sent)
            sent_labels = _index_seq(label2idx, lab)

            sent_word_strings = []
            for word in sent:
                word_chars = _index_seq(alphabet, word)
                sent_word_strings.append(word_chars)

            indexed_sents.append(sent_words)
            indexed_labs.append(sent_labels)
            indexed_word_chars.append(sent_word_strings)

    return list(zip(indexed_sents, indexed_word_chars, indexed_labs))

if __name__ == "__main__":

    '''manualSeed = 43
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)'''

    mode = "lstm"
    batch_size = 10
    hidden_units = 200
    epochs = 50
    drop = 0.5
    lr = 0.015
    char_embedding_dim = 30
    char_hidden_units = 50
    crf = True
    char = True

    conll_train = ingest.load_conll('data/conll2003/en/BIOES/NE_only/train.bmes')
    conll_val = ingest.load_conll('data/conll2003/en/BIOES/NE_only/valid.bmes')
    conll_test = ingest.load_conll('data/conll2003/en/BIOES/NE_only/test.bmes')

    #training
    vocab, alphabet, labels= preprocess.build_vocab(conll_train, conll_val, conll_test)
    label_lst = list(labels.keys())
    word_embedding_dim = vocab.vectors.size(1)
    vocab_lst = vocab.itos

    print("model:", mode)
    print("use crf:", crf)
    print("use character embedding:", char)
    print("batch size:", batch_size)
    print("learning rate:", lr)
    print("word embedding dimension:", word_embedding_dim)
    print("character embedding dimension:", char_embedding_dim)
    print("number of hidden units:", hidden_units)
    print("number of character hidden units", char_hidden_units)
    print("number of epochs:", epochs)
    print("tag scheme:", labels)


    train(conll_train, conll_val, conll_test,
        vocab, alphabet, labels,
        word_embedding_dim, char_embedding_dim,
        hidden_units, char_hidden_units, 4,
        batch_size=batch_size, epoch=epochs,
        dropout=drop, initial_lr=lr, decay_rate=0.05,
        mode=mode, crf=crf, char=char)