from torch import nn, optim
import torch
from model import EncoderBlock, ProjectionLayer, VisionTransformer, MultiHeadAttention, MLP
from config import hyperparameters as hp
def vit_model() -> nn.Module:
    multi_head = MultiHeadAttention(hp['number_of_heads'], hp['embedding_dim'])
    mlp_layer = MLP(hp['embedding_dim'], hp['d_ff'])
    encoder_layer  = EncoderBlock(multi_head, mlp_layer)
    projection = ProjectionLayer()
    model = VisionTransformer(encoder_layer, projection)
    return model

def train(model:nn.Module, criterion:nn.CrossEntropyLoss, 
          optimizer:optim.Adam, train_dataloader, val_dataloader) -> nn.Module:
    epoches:int = 5
    for epoch in range(epoches):
        train_epoch_loss:float = 0.0
        val_epoch_loss:float = 0.0
        model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            y_pred = model(batch[0])
            loss = criterion(y_pred, batch[1])
            loss.backward()
            optimizer.step()

            train_epoch_loss += loss.item()
        
        
        model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                y_pred = model(batch[0])
                val_loss = criterion(y_pred, batch[1])
                val_epoch_loss += val_loss.item()

        
        print(f'Train Loss in Epoch {epoch} --> {train_epoch_loss}')
        print(f'Val Loss in Epoch {epoch} --> {val_epoch_loss}')


    return model
            
        
       
        

        
            

    
    return model


if __name__ == "__main__":
    model = vit_model()
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.01)
    trained_model = train(model, loss, optimizer)