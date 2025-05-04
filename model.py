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
        self.query_weights = nn.Linear(self.d_model, self.d_model, bias=False)
        self.key_weights = nn.Linear(self.d_model, self.d_model, bias=False)
        self.value_weights = nn.Linear(self.d_model, self.d_model, bias=False)
        self.final_weights = nn.Linear(self.d_model, self.d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attention_map: torch.tensor = None) -> torch.tensor:
        query = self.query_weights(q)
        key = self.key_weights(k)
        value = self.value_weights(v)
        batch = query.shape[0]
        query = query.reshape(batch, -1, self.h, self.d_h).transpose(1, 2)
        key = key.reshape(batch, -1, self.h, self.d_h).transpose(1, 2)
        value = value.reshape(batch, -1, self.h, self.d_h).transpose(1, 2)
        x_tuple = self.attention_block(
            query=query, key=key, value=value, dropout=self.dropout, attention_map=attention_map)
        x = x_tuple[0]
        attention_scores = x_tuple[1]
        x = x.reshape(batch, -1, self.h * self.d_h)
        return self.final_weights(x), attention_scores

    def attention_block(self, query, key, value, dropout:nn.Dropout, attention_map:torch.tensor = None) -> torch.tensor:
        attention_scores = torch.matmul(
            query, key.transpose(-2, -1)) / math.sqrt(self.d_h)
        student_attention = torch.softmax(attention_scores, dim=-1)
        #print(f'This is the attention score shape: {attention_scores.shape}')
        final_attention = student_attention
        if attention_map is not None:
            final_attention = student_attention + attention_map  # Only for inference/experimentation
        if dropout is not None:
            final_attention = dropout(final_attention)
        return torch.matmul(final_attention, value), student_attention

class LayerNorm(nn.Module):
    def __init__(self, eps:float = 1e-05) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(hf['embedding_dim']))
        self.beta = nn.Parameter(torch.zeros(hf['embedding_dim']))

    def forward(self, x) -> torch.tensor:
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True)
        x = ((x - mean) * self.gamma / torch.sqrt(var + self.eps)) + self.beta
        return x

class MLP(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_ff)
        self.norm2 = nn.LayerNorm(d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    def forward(self, x) -> torch.tensor:
        x = self.dropout(self.norm1(self.gelu(self.fc1(x))))
        x = self.norm2(self.gelu(self.fc2(x)))
        return x

class ResidualConnection(nn.Module):
    def __init__(self, dropout:float = 0.1):
        super().__init__()
        self.norm = LayerNorm()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x:torch.tensor, sublayer:Union[MultiHeadAttention, MLP]):
        sublayer_output = sublayer(x, x, x) if isinstance(sublayer, MultiHeadAttention) else sublayer(x)
        if isinstance(sublayer_output, tuple):
            sublayer_output, attention_scores = sublayer_output
            output = self.norm(x + self.dropout(sublayer_output))
            return output, attention_scores
        else:
            output = self.norm(x + self.dropout(sublayer_output))
            return output, None

class EncoderBlock(nn.Module):
    def __init__(self, attention_block:MultiHeadAttention, feedforward:MLP) -> None:
        super().__init__()
        self.attention_block = attention_block
        self.blocks = nn.ModuleList([ResidualConnection() for _ in range(2)])
        self.feedforward = feedforward 
    def forward(self, x, attention_map:torch.tensor = None) -> torch.tensor:
        x, attention_scores = self.blocks[0](
            x, lambda x: self.attention_block(x, x, x, attention_map)
        )
        x, _ = self.blocks[1](
            x, lambda x: self.feedforward(x)
        )
        return x, attention_scores

class ProjectionLayer(nn.Module):
    def __init__(self, n_classes:int = hf['classes'], d_model:int = hf['embedding_dim']):
        super().__init__()
        self.fc1 = nn.Linear(d_model, n_classes)
        
    def forward(self, x):
        #x = torch.mean(x, dim=1, keepdim=True) # (batch, 1, 256)
        x = x[:, 0, :] #Passing the cls token
        x = self.fc1(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, encoderBlock1:EncoderBlock, encoderBlock2:EncoderBlock, encoderBlock3:EncoderBlock, 
                 projection_layer:ProjectionLayer):
        super().__init__()
        self.embeddings = InputEmbeddings(hf['in_channels'], hf['out_channels'], hf['patch_dim'], 
                                         hf['embedding_dim'], hf['image_size'])
        self.encoderBlock1 = encoderBlock1
        self.encoderBlock2 = encoderBlock2
        self.encoderBlock3 = encoderBlock3
        self.projection_layer = projection_layer

    def forward(self, x:torch.tensor, attention_maps:dict = None) -> torch.tensor:
        self.attention_scores = {}
        x = self.embeddings(x)
        x, self.attention_scores['block_1'] = self.encoderBlock1(x, attention_maps.get('block_1') if attention_maps else None)
        x, self.attention_scores['block_2'] = self.encoderBlock2(x, attention_maps.get('block_2') if attention_maps else None)
        x, self.attention_scores['block_3'] = self.encoderBlock3(x, attention_maps.get('block_3') if attention_maps else None)
        return self.projection_layer(x), self.attention_scores

class AttentionAugmentationModule(nn.Module):
    def __init__(self, resnet_model):
        super().__init__()
        self.resnet_model = resnet_model
        for parameters in self.resnet_model.parameters():
            parameters.requires_grad = False
        self.output = {}
        self.attention_map = {}
        resnet_model.features[1].register_forward_hook(self.hook(block_name='block_1'))
        resnet_model.features[2].register_forward_hook(self.hook(block_name='block_2'))
        resnet_model.features[3].register_forward_hook(self.hook(block_name='block_3'))
        # self.linear_blocks = nn.ModuleDict(
        #     {
        #         'block_1': nn.Sequential(
        #             nn.Linear(24, 65),
        #             nn.ReLU(), 
        #             nn.Linear(65, 128),
        #             nn.ReLU(), 
        #             nn.Linear(128, 4*65*65)
        #         ),
        #         'block_2': nn.Sequential(
        #             nn.Linear(32, 64),
        #             nn.ReLU(), 
        #             nn.Linear(64, 128),
        #             nn.ReLU(), 
        #             nn.Linear(128, 4*65*65)
        #         ),
        #         'block_3': nn.Sequential(
        #             nn.Linear(48, 64),
        #             nn.ReLU(), 
        #             nn.Linear(64, 128),
        #             nn.ReLU(), 
        #             nn.Linear(128, 4*65*65)
        #         )
        #     }
        # )

        self.linear_blocks = nn.ModuleDict({
            'block_1': nn.Linear(24, 4),
            'block_2': nn.Linear(32, 4),
            'block_3': nn.Linear(48, 4)
        })

    def hook(self, block_name):
        def fn(module, input, output):
            self.output[block_name] = output
        return fn

    def forward(self, x):
        x = self.resnet_model(x)
        for key, val in self.output.items():
            spatial_attn = val.pow(2).mean(dim=1, keepdim=True)  # (batch, 1, H, W)
            # Resize to match ViT patch grid (8x8 = 64 patches)
            spatial_attn = nn.functional.interpolate(
                spatial_attn, size=(8, 8), mode='bicubic'
            )  # (batch, 1, 8, 8)
            spatial_attn = spatial_attn.view(-1, 64)  # (batch, 64)
            # Add CLS token attention (e.g., mean of patch attentions)
            cls_attn = spatial_attn.mean(dim=1, keepdim=True)  # (batch, 1)
            seq_attn = torch.cat([cls_attn, spatial_attn], dim=1)  # (batch, 65)
            # Map channel features to head-specific attention
            channel_weights = self.linear_blocks[key](val.mean(dim=[2, 3]))  # (batch, 4)
            # Combine: (batch, 4, 65) for 4 heads
            head_attn = channel_weights.unsqueeze(2) * seq_attn.unsqueeze(1)  # (batch, 4, 65)
            # Form attention matrix: (batch, 4, 65, 65)
            self.attention_map[key] = head_attn.unsqueeze(-1).repeat(1, 1, 1, 65)
            #val = val.mean(dim=[2, 3])
            #self.attention_map[key] = self.linear_blocks[key](val).view(-1, 4, 65, 65)
        return self.attention_map