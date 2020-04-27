from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from numpy import inf
import torch.nn as nn
from torchcrf import CRF
import torch.nn.functional as F
import torch

class Basic_LSTM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_of_tags, use_crf=False):
        super(Basic_LSTM, self).__init__()

        self.hidden = hidden_dim*2

        self.use_crf = use_crf

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)

        self.output = nn.Linear(self.hidden, num_of_tags)

        if use_crf:
            self.crf = CRF(num_of_tags, batch_first=True) # don't include the <pad> label


    def forward(self, sent):
        #torch.set_printoptions(edgeitems=500)
        #hidden_dim= 32, batch size = 5, num_tags= 17, embedding_dim=50
        sentence, seq_lengths = sent #sent is a tuple of (input btach, seq_lengths)
        #batch_size, sent_length = sentence.size() #sentence is a batch of sentences with dim= [batch size, max_sent_length]
        #print("sentence shape: " + str(sentence.shape))
        embeddings = self.embedding(sentence)  #output shape:[5,_sent_length, 50]
        #print("embedding shape: " + str(embeddings.shape))
        packed_out = pack_padded_sequence(embeddings, seq_lengths.cpu().numpy(), batch_first=True) #output shape: [sum(batch_sent_lengths), 50]
        #print("packed_input data shape:" + str(packed_out.data.shape))
        lstm_out, hidden = self.lstm(packed_out) # output shape: [sum(batch_sent_lengths), 64] if bidirectional
        #print("packed_output shape: " + str(lstm_out.data.shape))
        unpacked_out, _ = pad_packed_sequence(lstm_out, batch_first=True, padding_value= inf) #output shape: [5, max_sent_length, 64] if bidirectional
        #print("unpacked_output shape: " + str(unpacked_out.data.shape))
        unpacked_out = unpacked_out.contiguous() #same shape
        #print("unpacked shape: " + str(unpacked_output.shape))
        mask = ~(unpacked_out.ge(inf)) #mask paddings from unpacked tensor
        #print("mask" + str(mask))
        masked = torch.masked_select(unpacked_out, mask) #output_shape = [sum(seq_lengths)*64] if bidirectional
        #print("masked shape: " + str(masked.shape))
        masked = masked.view(sum(seq_lengths), self.hidden) #output shape = [sum(seq_lengths, 64] if bidirectional
        #print("masked shape: " + str(masked.shape))
        tag_space = self.output(masked)# output shape: [sum(seq_lengths, 17]
        #print("tag space shape: " + str(tag_space.shape))
        # print()
        return tag_space














