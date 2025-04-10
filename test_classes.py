from embeddings import InputEmbeddings
from model import MultiHeadAttention
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
    print(f'Shape of x after MultiHead block: {x.shape}')


if __name__ == "__main__":
    #test_inputEmbeddings()
    test_multihead()