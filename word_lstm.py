import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF
import torch
from numpy import inf

class Basic_LSTM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_of_tags, bi=True, use_crf=False):
        super(Basic_LSTM, self).__init__()
        self.num_of_tags = num_of_tags

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=bi, batch_first=True)
        if bi:
            self.output = nn.Linear(hidden_dim*2, num_of_tags)
        else:
            self.output = nn.Linear(hidden_dim, num_of_tags)
        if use_crf:
            self.crf = CRF(num_of_tags, batch_first=True)

    def forward(self, sent):
        #torch.set_printoptions(edgeitems=500)
        #hidden= 32, batch size = 5, num_tags= 18
        sentence, seq_lengths = sent #sent is a tuple of (input btach, seq_lengths)
        #batch_size, sent_length = sentence.size() #sentence is a batch of sentences with dim= [batch size, max_sent_length]
        #print("sentence shape: " + str(sentence.shape))
        embeddings = self.embedding(sentence)  #output shape:[5,_sent_length, 50]
        #print("embedding shape: " + str(embeddings.shape))
        packed_input = pack_padded_sequence(embeddings, seq_lengths.numpy(), batch_first=True) #output shape: [sum(batch_sent_lengths), 50]
        #print("packed_input data shape:" + str(packed_input.data.shape))
        packed_output,  (ht, ct) = self.lstm(packed_input) # output shape: [sum(batch_sent_lengths), 64] if bidirectional
        #print("packed_output shape: " + str(packed_output.data.shape))
        unpacked_output, _ = pad_packed_sequence(packed_output, batch_first=True, padding_value= inf) #output shape: [5, max_sent_length, 64] if bidirectional
        #print("unpacked_output shape: " + str(unpacked_output.data.shape))
        unpacked_output = unpacked_output.contiguous() #same shape
        hidden_dim = unpacked_output.size(2)
        #print("unpacked shape: " + str(unpacked_output.shape))
        mask = ~(unpacked_output.ge(inf))
        #print("mask" + str(mask))
        masked = torch.masked_select(unpacked_output, mask) #output_shape = [1,sum(len_sentences)]
        #print("masked shape: " + str(masked.shape))
        masked = masked.view(-1, hidden_dim)
        #print("masked shape: " + str(masked.shape))
        tag_space = self.output(masked)# output shape: [5*max_sent_length, 18]
        #print("tag space shape: " + str(tag_space.shape))
        #outs = tag_space.view(-1, self.num_of_tags) #revert to original shape: [5, max_sent_length, 18]
        #print(outs.shape)
        #print()
        return tag_space













