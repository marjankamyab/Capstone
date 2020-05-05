import torch.nn as nn
import torch.nn.functional as F
import torch
from numpy import inf
from torchcrf import CRF


# N is a batch size, C denotes hidden dim, L is a length of the sequence.
class CNN(nn.Module):

  def __init__(self, vocab_size, embedding_dim, hidden_dim, num_of_layers, num_of_tags, use_crf=False):
    super(CNN, self).__init__()

    kernel = 3

    pad = int((kernel-1)/2)

    self.softmax = nn.LogSoftmax(2)

    self.num_of_layers = num_of_layers

    self.embedding = nn.Embedding(vocab_size, embedding_dim)

    self.input = nn.Linear(embedding_dim, hidden_dim)

    self.conv_lst = nn.ModuleList()

    self.norm_lst = nn.ModuleList()

    for i in range(num_of_layers):
      self.conv_lst.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel, padding=pad))
      self.norm_lst.append(nn.BatchNorm1d(hidden_dim))

    self.output = nn.Linear(hidden_dim, num_of_tags)

    if use_crf:
      self.crf = CRF(num_of_tags, batch_first=True)


  def forward(self, sent):
      sentence, _ = sent

      embeddings = self.embedding(sentence)

      input = torch.tanh(self.input(embeddings)).transpose(2,1).contiguous()

      cnn_in = torch.relu(self.conv_lst[0](input))
      if sentence.size(0)>1: cnn_in = self.norm_lst[0](cnn_in)

      for i in range(1,self.num_of_layers):
        cnn_in = torch.relu(self.conv_lst[i](cnn_in))
        if sentence.size(0)>1: cnn_in = self.norm_lst[i](cnn_in)

      cnn_out = cnn_in.transpose(2,1).contiguous()
      #print(cnn_out.shape)

      tag_space = self.output(cnn_out)

      return tag_space














































