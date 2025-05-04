from embeddings import InputEmbeddings
from model import *
from config import hyperparameters as hp
import torch
def test_inputEmbeddings() -> None:
    x = torch.randn(1,3, 64,64)

    input_embedding = InputEmbeddings(hp['in_channels'], 
                                      hp['out_channels'], hp['patch_dim'], 
                                      hp['embedding_dim'], hp['image_size'])
    x = input_embedding(x)
    print(x.shape)

def test_multihead() ->None:
    x = torch.randn(1,64,64)
    multi_attention_layer = MultiHeadAttention(hp['number_of_heads'], hp['embedding_dim'])
    x = multi_attention_layer(x,x,x)
    print(f'Shape of x after MultiHead block: {x[0].shape}')


def test_encoderBlock() -> torch.tensor:
    x = torch.randn(3,64,64) # (Batch size, seq, dmodel)
    multi_attention_layer = MultiHeadAttention(hp['number_of_heads'], hp['embedding_dim'])
    #feedforward_layer = FeedForwardNetwork(hp['embedding_dim'], hp['d_ff'])
    mlp_layer = MLP(hp['embedding_dim'], hp['d_ff'])
    encoder_block = EncoderBlock(multi_attention_layer, mlp_layer)
    x = encoder_block(x)
    print(x.shape)
    return x

def test_MLP() -> None:
    x = test_encoderBlock()
    projection_layer = ProjectionLayer()
    x = projection_layer(x)
    #x = torch.mean(x, dim=1)
    predicted_label = torch.argmax(x, dim=-1).squeeze(-1)
    print(predicted_label)
    print(f'Shape of x after MLP layer: {x.shape}')
if __name__ == "__main__":
    #test_inputEmbeddings()
    test_multihead()
    #test_encoderBlock()
    #test_MLP()