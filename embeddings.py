from torch import nn
import torch


class InputEmbeddings(nn.Module):
    def __init__(self, input_channels, out_channels, patch_dim, embedding_dim, image_size):
        super().__init__()
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
            num_embeddings=self.number_of_patches + 1,
            embedding_dim=embedding_dim
        )

        self.cls_token = nn.Parameter(torch.zeros(1,1,embedding_dim))
    def forward(self,x):
        batch_size = x.shape[0]

        x = self.patch_embeddings(x)
        x = x.flatten(2,3)
        x = x.transpose(1,2) # (B, C, Flat) --> (B, Flat, C)

        cls_token = self.cls_token.expand(batch_size, -1,-1) # (4,1,256)

        x = torch.cat((cls_token, x), dim = 1) # -> (4,257,256)

        pos_indices = torch.arange(self.number_of_patches+1)
        pos_embed = self.positional_embedding(pos_indices) #--> (257,256)
        pos_embed = pos_embed.unsqueeze(0).expand(batch_size, -1,-1) # --> (4,257,256)

        x +=pos_embed

        return x