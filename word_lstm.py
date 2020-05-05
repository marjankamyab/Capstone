from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from numpy import inf
import torch.nn as nn
from torchcrf import CRF
import torch

class BiLSTM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_of_tags, use_crf=False):
        super(BiLSTM, self).__init__()

        assert hidden_dim % 2 == 0

        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim//2, num_layers=1, bidirectional=True, batch_first=True)

        self.output = nn.Linear(self.hidden_dim, num_of_tags)

        if use_crf:
            self.crf = CRF(num_of_tags, batch_first=True)


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
        unpacked_out, _ = pad_packed_sequence(lstm_out, batch_first=True, padding_value=-1) #output shape: [5, max_sent_length, 64] if bidirectional
        #print("unpacked_output shape: " + str(unpacked_out.data.shape))
        output = self.output(unpacked_out.contiguous())# output shape: [sum(seq_lengths, 17]
        #print("output shape: " + str(output.shape))
        #print()
        return output













