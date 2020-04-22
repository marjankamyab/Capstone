import preprocess, ingest, word_lstm
import torch.nn.functional as F
from random import seed, shuffle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import subprocess


def train(train_dataset:ingest.Corpus, val_dataset:ingest.Corpus, test_dataset:ingest.Corpus,
          vocab_size: int, embedding_dim:int, weights: torch.Tensor, hidden_dim:int, num_tags:int,
          epoch:int=1, manualSeed=42, crf=False,
          val_output_path="./output/val/val_output", test_output_path="./output/test/test_output") \
          -> None:
    #setting the random seed
    seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    print("crf: "+ str(crf))
    print("Seed num: " + str(manualSeed))
    #model initialization
    model = word_lstm.Basic_LSTM(vocab_size, embedding_dim, hidden_dim, num_tags, bi=True, use_crf=crf)
    optimizer = optim.SGD(model.parameters(), lr=0.015)
    loss_function = nn.NLLLoss(ignore_index=num_tags-1, size_average=False)
    for num in range(epoch):
        epoch_loss = .0
        print("Epoch " + str(num) + ":")
        shuffle(train_dataset)
        for batch,(sent,label) in enumerate(train_dataset):
            optimizer.zero_grad()
            outs = model(sent)
            mask = ~(label.ge(num_tags-1))
            gold = torch.masked_select(label, mask)
            if crf:
                outs = outs.view((1, outs.size(0), outs.size(1)))
                gold = gold.view(1, gold.size(0))
                loss = -(model.crf.forward(outs, gold))
            else:
                tag_scores = F.log_softmax(outs.float(), dim=1)  # output shape: [sum(sent_lengths)_length, 18]
                loss = loss_function(tag_scores, gold)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
            optimizer.zero_grad()
        #val and test evaluation between epochs
        print("epoch loss: " + str(epoch_loss.item()))
        val_file = evaluate(val_dataset, model, val_output_path, seed, num, crf)
        test_file = evaluate(test_dataset, model, test_output_path, seed, num, crf)
        #print("epoch loss: " + str(epoch_loss.item()))
        val_pl = pl2output(val_file)
        test_pl = pl2output(test_file)
        print('dev results: ', val_pl)
        print('test results: ', test_pl)
        print()


def pl2output(file):
    with open(file, 'rb', 0) as f:
        pl2string = subprocess.check_output(['perl', 'conlleval.pl'], stdin=f, universal_newlines=True)
    return pl2string


def evaluate(dataset:list, model, output_file, epoch:int, seed, crf:bool):
    total_gold = []
    total_pred = []
    total_tokens = []
    for i,(sent_seq, gold_seq) in enumerate(dataset):
        batch_sents = sent_seq[0]
        gold_mask = ~(gold_seq.ge(num_tags-1))
        batch_mask = ~(batch_sents.ge(vocab['<pad>']))
        batch_sents = torch.masked_select(batch_sents, batch_mask)
        gold = torch.masked_select(gold_seq, gold_mask)
        pred = predict(model, sent_seq, crf)
        total_tokens.extend(batch_sents)
        total_gold.extend(gold)
        total_pred.extend(pred)
    tokens = [vocab_lst[token.item()] for token in total_tokens]
    golds = [label_lst[gold.item()] for gold in total_gold]
    preds = [label_lst[tag] for tag in total_pred]
    zipped = zip(tokens, golds, preds)
    output_path = output_file + str(seed) + str(epoch) + '.txt'
    write_output(output_path, zipped)
    return output_path


def predict(model, sent, crf):
    batch_predictions = []
    output = model(sent)
    if crf:
        output = output.view((1, output.size(0), output.size(1)))
        pred_lst = model.crf.decode(output)  ##output type: List[List[int]]
        for pred in pred_lst[0]:
            batch_predictions.append(pred)
    else:
        output = F.log_softmax(output.float(), dim=1) # output shape = [batch_size*max_sent_length, num_tags]
        for i in range(output.size(0)):
            pred = torch.argmax(output[i])  ##gathering token predictions for the entire batch
            batch_predictions.append(pred.item())
    return batch_predictions


def write_output(file, zipped_file):
    with open(file, 'w') as f:
        for token, gold, pred in zipped_file:
            f.write(token + " " + gold + " " + pred)
            f.write("\n")


if __name__ == "__main__":
    batch_size = 64
    hidden_units = 32
    epochs = 100
    lr = 0.015
    conll_train = ingest.load_conll('data/conll2003/en/BIOES/NE_only/train.bmes')
    conll_val = ingest.load_conll('data/conll2003/en/BIOES/NE_only/valid.bmes')
    conll_test = ingest.load_conll('data/conll2003/en/BIOES/NE_only/test.bmes')
    train_tokens = []
    val_tokens = []
    test_tokens = []
    for document in conll_train:
        for i,sent in enumerate(document.sentences):
           train_tokens.extend(sent)
    #training
    vocab, labels= preprocess.build_vocab(conll_train, conll_val, conll_test)
    label_lst = list(labels.keys())
    vocab_lst = vocab.itos
    print("train number of tokens: " + str(len(train_tokens)))
    print("batch size: " + str(batch_size))
    print("learning rate: " + str(lr))
    print("number of hidden units: " + str(hidden_units))
    print("number of epochs: " + str(epochs))
    print("tag scheme: " + str(label_lst))
    train_data = preprocess.prepare_dataset(conll_train, vocab, labels, batch_size=batch_size)
    val_data = preprocess.prepare_dataset(conll_val, vocab, labels, batch_size=batch_size)
    test_data = preprocess.prepare_dataset(conll_test, vocab, labels, batch_size=batch_size)
    vocab_size = vocab.vectors.size(0)
    embedding_dim = vocab.vectors.size(1)
    num_tags = len(labels)
    train(train_data, val_data, test_data, vocab_size, embedding_dim, vocab.vectors, hidden_units, num_tags,
          epoch=epochs, crf=False)