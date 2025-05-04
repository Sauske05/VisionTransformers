import os
from model import AttentionAugmentationModule, VisionTransformer, EncoderBlock, ProjectionLayer, MultiHeadAttention, MLP
from torchvision.models import efficientnet_b3
import torch
from torch import nn, optim
from config import hyperparameters as hp
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from data_preprocess import load_dataloader
import logging

logger = logging.getLogger(__name__)

def vit_model() -> nn.Module:
    multi_head1 = MultiHeadAttention(hp['number_of_heads'], hp['embedding_dim'])
    mlp_layer1 = MLP(hp['embedding_dim'], hp['d_ff'])
    encoder_layer1 = EncoderBlock(multi_head1, mlp_layer1)
    
    multi_head2 = MultiHeadAttention(hp['number_of_heads'], hp['embedding_dim'])
    mlp_layer2 = MLP(hp['embedding_dim'], hp['d_ff'])
    encoder_layer2 = EncoderBlock(multi_head2, mlp_layer2)
    
    multi_head3 = MultiHeadAttention(hp['number_of_heads'], hp['embedding_dim'])
    mlp_layer3 = MLP(hp['embedding_dim'], hp['d_ff'])
    encoder_layer3 = EncoderBlock(multi_head3, mlp_layer3)
    
    projection = ProjectionLayer()
    model = VisionTransformer(encoder_layer1, encoder_layer2, encoder_layer3, projection)
    return model

def accuracy(y_pred, y_actual) -> float:
    return accuracy_score(y_actual, y_pred)

def compute_kl_loss(vit_attention_scores: dict, augmented_attention_scores: dict) -> torch.Tensor:
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    total_kl = 0.0
    blocks = ['block_1', 'block_2', 'block_3']
    for block in blocks:
        vit_scores = torch.log_softmax(vit_attention_scores[block], dim=-1)
        aug_scores = torch.softmax(augmented_attention_scores[block], dim=-1)
        total_kl += kl_loss(vit_scores, aug_scores)
    return total_kl

def train(device, start_epoch=0, total_epochs=12):
    # Initialize models and optimizers
    resnet_model = efficientnet_b3()
    model = vit_model().to(device)
    augment_model = AttentionAugmentationModule(resnet_model).to(device)
    vit_optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    augment_optimizer = optim.Adam(augment_model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Define checkpoint path
    checkpoint_path = 'checkpoint.pth'

    # Load checkpoint if it exists and we're resuming
    if os.path.exists(checkpoint_path) and start_epoch > 0:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['vit_model_state_dict'])
        augment_model.load_state_dict(checkpoint['augment_model_state_dict'])
        vit_optimizer.load_state_dict(checkpoint['vit_optimizer_state_dict'])
        augment_optimizer.load_state_dict(checkpoint['augment_optimizer_state_dict'])
        logger.info(f'Checkpoint loaded, continuing from epoch {start_epoch}')
    else:
        logger.info('No checkpoint found or starting from scratch')

    # Load data
    train_dataloader, val_dataloader = load_dataloader()
    criterion_vit = nn.CrossEntropyLoss()

    # Training loop
    for epoch in tqdm(range(start_epoch, total_epochs)):
        train_epoch_loss = 0.0
        val_epoch_loss = 0.0
        model.train()
        train_epoch_accuracy, val_epoch_accuracy = 0.0, 0.0

        for index, batch in enumerate(tqdm(train_dataloader)):
            vit_optimizer.zero_grad()
            augment_optimizer.zero_grad()
            image_array = batch[0].to(device)
            augmented_attention_map = augment_model(image_array)
            y_pred, attention_scores = model(image_array)
            y_pred = y_pred.squeeze()
            vit_loss = criterion_vit(y_pred, batch[1].to(device))
            aam_loss = compute_kl_loss(attention_scores, augmented_attention_map)
            loss = vit_loss + 0.5 * aam_loss
            loss.backward()
            vit_optimizer.step()
            augment_optimizer.step()
            y_logits = torch.argmax(y_pred.detach().cpu(), dim=1).numpy()
            y_actual = batch[1].numpy()
            train_epoch_accuracy += accuracy(y_actual, y_logits)
            train_epoch_loss += loss.item()

        train_accuracy = train_epoch_accuracy / len(train_dataloader)
        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                image_array = batch[0].to(device)
                y_pred, _ = model(image_array)
                y_pred = y_pred.squeeze()
                val_loss = criterion_vit(y_pred, batch[1].to(device))
                val_epoch_loss += val_loss.item()
                y_logits = torch.argmax(y_pred.detach().cpu(), dim=1).numpy()
                y_actual = batch[1].numpy()
                val_epoch_accuracy += accuracy(y_actual, y_logits)

        val_accuracy = val_epoch_accuracy / len(val_dataloader)
        
        logger.info(f'Average Train Loss in Epoch {epoch} --> {train_epoch_loss / len(train_dataloader)}')
        logger.info(f'Train accuracy in Epoch {epoch} --> {train_accuracy * 100} %')
        logger.info(f'Average Val Loss in Epoch {epoch} --> {val_epoch_loss / len(val_dataloader)}')
        logger.info(f'Val accuracy in Epoch {epoch} --> {val_accuracy * 100} %')

    # Save checkpoint after training
    torch.save({
        'vit_model_state_dict': model.state_dict(),
        'augment_model_state_dict': augment_model.state_dict(),
        'vit_optimizer_state_dict': vit_optimizer.state_dict(),
        'augment_optimizer_state_dict': augment_optimizer.state_dict()
    }, checkpoint_path)
    logger.info('Training completed and checkpoint saved')

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('distil.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    
    logger.info("Testing console output")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info('Started Training Loop')
    
    train(device, start_epoch=12, total_epochs=24)
    
    logger.info('Training Loop Finished')