from torch import nn, optim
import torch
from model import EncoderBlock, ProjectionLayer, VisionTransformer, MultiHeadAttention, MLP
from config import hyperparameters as hp
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from data_preprocess import load_dataloader
from intialize_log import log
logger = log()
# import logging
# logger = logging.getLogger(__name__)
def vit_model() -> nn.Module:
    multi_head = MultiHeadAttention(hp['number_of_heads'], hp['embedding_dim'])
    mlp_layer = MLP(hp['embedding_dim'], hp['d_ff'])
    encoder_layer  = EncoderBlock(multi_head, mlp_layer)
    projection = ProjectionLayer()
    model = VisionTransformer(encoder_layer, projection)
    return model

def accuracy(y_pred, y_actual) -> float:
    return accuracy_score(y_actual, y_pred)

def train(model:nn.Module, criterion:nn.CrossEntropyLoss, 
          optimizer:optim.Adam, train_dataloader, val_dataloader, device) -> None:
    epoches:int = 10
    for epoch in tqdm(range(epoches)):
        train_epoch_loss:float = 0.0
        val_epoch_loss:float = 0.0
        model.train()
        train_epoch_accuracy, val_epoch_accuracy = 0.0, 0.0

        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            image_array = batch[0].permute(0,-1,1,2).to(device)
            y_pred = model(image_array).squeeze()
            #print(f'Shape of y_pred {y_pred.shape}')
            #print(f'Shape of batch[1] {batch[1].shape}')
            #print(batch[1])

            #print(batch[1])
            loss = criterion(y_pred, batch[1].to(device))
            loss.backward()
            optimizer.step()
            y_logits = torch.argmax(y_pred.detach().cpu(), dim=1).squeeze().numpy()
            y_acutal = batch[1].numpy()
            train_epoch_accuracy += accuracy(y_acutal, y_logits)
            train_epoch_loss += loss.item()

        train_accuracy = train_epoch_accuracy/ len(train_dataloader)
        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                image_array = batch[0].permute(0,-1,1,2).to(device)
                y_pred = model(image_array).squeeze()
                val_loss = criterion(y_pred, batch[1].to(device))
                val_epoch_loss += val_loss.item()
                y_logits = torch.argmax(y_pred.detach().cpu(), dim=1).squeeze().numpy()
                y_acutal = batch[1].numpy()
                val_epoch_accuracy += accuracy(y_acutal, y_logits)

        val_accuracy = val_epoch_accuracy / len(val_dataloader)
        
        logger.info(f'Average Train Loss in Epoch {epoch} --> {train_epoch_loss / len(train_dataloader)}')
        logger.info(f'Train accuracy in Epoch {epoch} --> {train_accuracy * 100} %')
        logger.info(f'Average Val Loss in Epoch {epoch} --> {val_epoch_loss / len(val_dataloader)}')
        logger.info(f'Val accuracy in Epoch {epoch} --> {val_accuracy * 100} %')




if __name__ == "__main__":
    # logging.basicConfig(filename='first_run.log', level=logging.INFO)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # file_handler = logging.FileHandler('first_run.log')
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)

    # stream_handler = logging.StreamHandler()
    # stream_handler.setFormatter(formatter)
    # stream_handler.setLevel(logging.INFO)
    # logger.addHandler(stream_handler)
    
    # logger.info("Testing console output")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = vit_model().to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr =3e-4)
    train_dataloader, val_dataloader = load_dataloader()
    #print('Started Training Loop!')
    logger.info('Started Training Loop')
    train(model, loss, optimizer, train_dataloader, val_dataloader, device)
    logger.info('Training Loop Finished')
    #print('Training Loop Finished')