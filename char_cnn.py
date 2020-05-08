import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
torch.set_printoptions(edgeitems=500)


class CharCNN(nn.Module):
    def __init__(self, alphabet_size, embedding_dim, hidden_dim):

        super(CharCNN, self).__init__()

        random_embbedings = self.char_embedding(alphabet_size, embedding_dim)

        self.hidden = hidden_dim

        self.embedding = nn.Embedding(alphabet_size, embedding_dim)

        self.embedding.weight = nn.Parameter(random_embbedings, requires_grad=True)

        self.cnn = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3, padding=1)


    def get_embedding(self, chars):
        batch_size = chars.size(0)
        #print("chars shape:", chars.size())
        embeddings = self.embedding(chars)
        #print("char embedding shape:", embeddings.size())
        transposed = embeddings.transpose(2,1).contiguous()
        #print("char embedding transposed shape:", transposed.size())
        char_in = self.cnn(transposed)
        #print("char_in shape:", char_in.size())
        char_out = F.max_pool1d(char_in, char_in.size(2)).view(batch_size, self.hidden)
        #print("char_out shape:", char_out.size())
        #print()
        return char_out

    def char_embedding(self, alphabet_size, embedding_dim):
        scale = np.sqrt(3.0/embedding_dim)
        tensor = torch.ones(())
        #index zero is for paddings
        embedding = tensor.new_empty((alphabet_size+1, embedding_dim), requires_grad=True)
        for i in range(alphabet_size+1):
            embedding[i] = torch.Tensor(np.random.uniform(-scale, scale, [1, embedding_dim]))
        return embedding


if __name__ == "__main__":
    cnn = CharCNN(26, 50, 1)
