from torch import nn
import torch
import math
class MultiHeadAttention(nn.Module):
    def __init__(self, number_of_heads:int, embedding_dim:int, 
                 dropout:float = 0.1) -> None:
        super().__init__()
        self.h = number_of_heads
        self.d_model = embedding_dim
        self.d_h = embedding_dim // number_of_heads
        assert self.d_model % self.h == 0, 'Make sure d_model (Embedding dim) is divisible by number of heads!'
        self.query_weights = nn.Linear(self.d_model, self.d_model, bias = False)
        self.key_weights = nn.Linear(self.d_model, self.d_model, bias = False)
        self.value_weights = nn.Linear(self.d_model, self.d_model, bias = False)
        self.final_weights = nn.Linear(self.d_model, self.d_model, bias = False)
        self.dropout = nn.Dropout(dropout)

    def forward(self,q, k, v) -> torch.tensor:
        print(f'Shape of q: {q.shape}')
        
        query = self.query_weights(q) #Shape : (Batch, path_dim ** 2, embedding_dim)
        key = self.key_weights(k)
        value = self.value_weights(v)
        batch = query.shape[0]
        print(f'Shape of q before reshape: {q.shape}')
        #We need to reshape the query, key and value in the shape 
        # (Batch,patch_dim ** 2, n_heads, embeding_dim // n_heads)
        query = query.reshape(batch,-1, self.h, self.d_h)
        key = key.reshape(batch,-1, self.h, self.d_h)
        value = value.reshape(batch,-1, self.h, self.d_h)
        print(f'Shape of query after reshape : {query.shape}')
        x = self.attention_block(query=query, key=key, 
                                 value=value, dropout=self.dropout)
        print(f'This is the shape after attention block :{x.shape}')
        

        x = x.reshape(batch, -1, self.h * self.d_h)
        return self.final_weights(x)

    def attention_block(self, query, key, value, dropout:nn.Dropout) -> torch.tensor:
        attention_scores = torch.matmul(
            query, key.transpose(-2,-1)) / math.sqrt(self.d_h)
        print(f'Attention Score shapes: {attention_scores.shape}')
        attention_scores = torch.softmax(attention_scores, dim = -1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return torch.matmul(attention_scores, value)
    

