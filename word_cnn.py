import torch.nn as nn
import torch.nn.functional as F
import torch
from torchcrf import CRF
from char_cnn import CharCNN


# N is a batch size, C denotes hidden dim, L is a length of the sequence.
class CNN(nn.Module):

  def __init__(self, vocab_size, alphabet_size, word_embedding_dim, char_embedding_dim, hidden_dim, char_hidden_dim, num_of_layers, num_of_tags, dropout, use_crf= False, use_char= False):
    super(CNN, self).__init__()

    kernel = 3

    pad = int((kernel-1)/2)

    self.num_of_layers = num_of_layers

    self.use_char = use_char

    if self.use_char:
        self.embedding_dim = word_embedding_dim + char_hidden_dim

    else:
        self.embedding_dim = word_embedding_dim

    self.char_embedding = CharCNN(alphabet_size, char_embedding_dim, char_hidden_dim)

    self.word_embedding = nn.Embedding(vocab_size, word_embedding_dim)

    self.input = nn.Linear(self.embedding_dim, hidden_dim)

    self.conv_lst = nn.ModuleList()

    self.norm_lst = nn.ModuleList()

    for i in range(num_of_layers):
      self.conv_lst.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel, padding=pad))
      self.drop_lst.append(nn.Dropout(dropout))
      self.norm_lst.append(nn.BatchNorm1d(hidden_dim))

    self.output = nn.Linear(hidden_dim, num_of_tags)

    if use_crf:
      self.crf = CRF(num_of_tags, batch_first=True)


  def forward(self, sent, chars):
      char, char_recover_perms = chars
      sentence, _ = sent
      batch_size, sent_length= sentence.size()
      word_embeddings = self.word_embedding(sentence)  # output shape:[5,_sent_length, 50]
      # print("embedding shape: " + str(embeddings.shape))
      if self.use_char:
          char_embeddings = self.char_embedding.get_embedding(char)
          char_embeddings = char_embeddings[char_recover_perms]
          char_embeddings = char_embeddings.view(batch_size, sent_length, -1)
          embedding_lst = [char_embeddings] + [word_embeddings]
          embeddings = torch.cat(embedding_lst, dim=2)
      else:
          embeddings = word_embeddings
      embeddings = self.drop(embeddings)
      #print("embedding_dim:", embeddings.size())
      input = torch.tanh(self.input(embeddings)).transpose(2,1).contiguous()
      #print("input shape: " + str(input.shape))

      cnn_in = F.relu(self.conv_lst[0](input))
      cnn_in = self.drop_lst[0](cnn_in)
      if sentence.size(0)>1: cnn_in = self.norm_lst[0](cnn_in)
      for i in range(1,self.num_of_layers):
          cnn_in = F.relu(self.conv_lst[i](cnn_in))
          cnn_in = self.drop_lst[i](cnn_in)
          if sentence.size(0)>1: cnn_in = self.norm_lst[i](cnn_in)
      #print("cnn_in shape: " + str(cnn_in.shape))

      cnn_out = cnn_in.transpose(2,1).contiguous()
      #print("cnn_out shape: " + str(cnn_out.shape))
      output = self.output(cnn_out)
      #print("output shape: " + str(output.shape))
      #print()
      return output














































