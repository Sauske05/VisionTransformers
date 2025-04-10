from torch import nn
import torch


class InputEmbeddings(nn.Module):
    def __init__(self, input_channels, out_channels, patch_dim, embedding_dim, image_size):
        super().__init__()
        #Patch dim of 8
        self.image_size = image_size
        self.patch_dim = patch_dim
        self.number_of_patches = (image_size // patch_dim) ** 2


        self.patch_embeddings = nn.Conv2d(
            in_channels= input_channels,
            out_channels=out_channels,
            kernel_size=patch_dim,
            stride=patch_dim
        )

        self.positional_embedding = nn.Embedding(
            num_embeddings=self.number_of_patches,
            embedding_dim=embedding_dim
        )

    def forward(self,x):
        batch_size = x.shape[0]
        print(f'This is the batch size: {batch_size}')
        x = self.patch_embeddings(x)
        print(f'Shape after path embedding: {x.shape}')
        x = x.flatten(2,3)
        print(f'Shape after Flatten: {x.shape}')
        x = x.transpose(1,2) # (B, C, Flat) --> (B, Flat, C)
        print(f'Shape after Transpose: {x.shape}')
        #cls_token = self.cls_token.expand(batch_size, -1,-1) # (4,1,256)

        #x = torch.cat((cls_token, x), dim = 1) # -> (4,257,256)

        pos_indices = torch.arange(self.number_of_patches)
        pos_embed = self.positional_embedding(pos_indices) #--> (256,256)
        print(f'Shape of pos embed: {[pos_embed.shape]}')
        pos_embed = pos_embed.expand(batch_size, -1,-1) # --> (4,256,256)

        x +=pos_embed

        return x