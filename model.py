from torch import nn
import torch
from typing import Union
import math
from embeddings import InputEmbeddings
from config import hyperparameters as hf

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
        #print(f'Shape of q: {q.shape}')
        
        query = self.query_weights(q) #Shape : (Batch, path_dim ** 2, embedding_dim)
        key = self.key_weights(k)
        value = self.value_weights(v)
        batch = query.shape[0]
        #print(f'Shape of q before reshape: {q.shape}')
        #We need to reshape the query, key and value in the shape 
        # (Batch,patch_dim ** 2, n_heads, embeding_dim // n_heads)
        query = query.reshape(batch,-1, self.h, self.d_h)
        key = key.reshape(batch,-1, self.h, self.d_h)
        value = value.reshape(batch,-1, self.h, self.d_h)
        #print(f'Shape of query after reshape : {query.shape}')
        x = self.attention_block(query=query, key=key, 
                                 value=value, dropout=self.dropout)
        #print(f'This is the shape after attention block :{x.shape}')
        

        x = x.reshape(batch, -1, self.h * self.d_h)
        return self.final_weights(x)

    def attention_block(self, query, key, value, dropout:nn.Dropout) -> torch.tensor:
        attention_scores = torch.matmul(
            query, key.transpose(-2,-1)) / math.sqrt(self.d_h)
        #print(f'Attention Score shapes: {attention_scores.shape}')
        attention_scores = torch.softmax(attention_scores, dim = -1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return torch.matmul(attention_scores, value)
    

class LayerNorm(nn.Module):
    def __init__(self, eps:float = 1e-05) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(hf['embedding_dim']))
        self.beta = nn.Parameter(torch.zeros(hf['embedding_dim']))
        


    def forward(self,x) -> torch.tensor:
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True)

        x = ((x-mean) * self.gamma / torch.sqrt(var + self.eps)) + self.beta


        return x

# class FeedForwardNetwork(nn.Module):
#     def __init__(self, d_model:int, d_ff:int, dropout:float = 0.1):
#         super().__init__()
#         self.d_model = d_model
#         self.d_ff = d_ff
#         self.linear_layer1 = nn.Linear(d_model, d_ff)
#         self.linear_layer2 = nn.Linear(d_ff, d_model)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
#     def forward(self, x):
#         x = self.relu(self.linear_layer1(x))
#         x = self.dropout(x)
#         return self.linear_layer2(x)
        
class MLP(nn.Module):
    def __init__(self, d_model:int, d_ff:int,dropout:float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

        self.norm1 = nn.LayerNorm(d_ff)
        self.norm2 = nn.LayerNorm(d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    def forward(self,x) -> torch.tensor:
        #print(f'This is the shape before error: {x.shape}')
        x = self.dropout(self.norm1(self.gelu(self.fc1(x))))
        
        x = self.norm2(self.gelu(self.fc2(x)))
        return x

class ResidualConnection(nn.Module):
    def __init__(self, dropout:float = 0.1):
        super().__init__()
        self.norm = LayerNorm()
        self.dropout = nn.Dropout(dropout)


    def forward(self, x:torch.tensor, sublayer:Union[MultiHeadAttention, MLP]):
        return self.norm(x + self.dropout(sublayer(x)))
    


class EncoderBlock(nn.Module):
    def __init__(self, attention_block:MultiHeadAttention, feedforward: MLP) -> None:
        super().__init__()
        self.attention_block = attention_block
        self.blocks = nn.ModuleList([ResidualConnection() for _ in range(2)])
        self.feedforward = feedforward 
    def forward(self, x) -> torch.tensor:
        x = self.blocks[0](
            x, lambda x: self.attention_block(x,x,x)
        )

        x = self.blocks[1](
            x, lambda x: self.feedforward(x)
        )
        return x
    

class ProjectionLayer(nn.Module):
    def __init__(self, n_classes:int = hf['classes'], d_model:int = hf['embedding_dim']):
        super().__init__()
        self.fc1 = nn.Linear(d_model, n_classes)
        
    def forward(self,x):
        x = torch.mean(x, dim=1, keepdim=True)
        x = self.fc1(x)
        return x
    

class VisionTransformer(nn.Module):
    def __init__(self, encoderBlock:EncoderBlock, projection_layer:ProjectionLayer):
        super().__init__()
        self.embeddings = InputEmbeddings(hf['in_channels'],hf['out_channels'],hf['patch_dim'], 
                                          hf['embedding_dim'], hf['image_size'])
        self.encoderBlock = encoderBlock
        self.projection_layer = projection_layer

    def forward(self,x:torch.tensor) -> torch.tensor:
        x= self.embeddings(x)
        x = self.encoderBlock(x)
        return self.projection_layer(x)
        