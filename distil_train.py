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
import torch.nn.functional as F

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


def compute_mse_l2_loss(augmented_maps: torch.Tensor, cnn_maps: torch.Tensor) -> torch.Tensor:
    assert augmented_maps.shape == cnn_maps.shape, \
        f"Shape mismatch: AugMaps {augmented_maps.shape}, CNNMaps {cnn_maps.shape}"
        
    # L2 Normalize along spatial dimensions (P, P) for each channel C and batch element B
    # Flatten spatial dims -> Norm -> Reshape back
    batch_size, num_channels, P, _ = augmented_maps.shape
    
    # Flatten: (B, C, P, P) -> (B, C, P*P)
    aug_flat = augmented_maps.view(batch_size, num_channels, -1)
    cnn_flat = cnn_maps.view(batch_size, num_channels, -1)
    
    # L2 Norm along the last dimension (P*P)
    # Add small epsilon to prevent division by zero
    norm_aug = torch.linalg.norm(aug_flat, ord=2, dim=-1, keepdim=True) + 1e-6
    norm_cnn = torch.linalg.norm(cnn_flat, ord=2, dim=-1, keepdim=True) + 1e-6
    
    # Normalize
    aug_normalized_flat = aug_flat / norm_aug
    cnn_normalized_flat = cnn_flat / norm_cnn
    
    # Reshape back (optional, MSE works on flat tensors too)
    # aug_normalized = aug_normalized_flat.view(batch_size, num_channels, P, P)
    # cnn_normalized = cnn_normalized_flat.view(batch_size, num_channels, P, P)
    
    # Compute Mean Squared Error (MSE) loss
    # mse_loss = F.mse_loss(aug_normalized, cnn_normalized, reduction='mean') 
    # Paper uses || A/||A|| - B/||B|| ||_2 which implies sum of squares, then maybe mean over batch/channels?
    mse_loss = F.mse_loss(aug_normalized_flat, cnn_normalized_flat, reduction='mean')
    
    return mse_loss

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
    resnet_model = efficientnet_b3().to(device)
    resnet_model.eval()
    #resnet_model = resnet_model.to(device)
    model = vit_model().to(device)
    augment_model = AttentionAugmentationModule(resnet_model, 4,3,['features.1', 'features.2', 'features.3']).to(device)
    vit_optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    augment_optimizer = optim.Adam(augment_model.parameters(), lr=1e-3, weight_decay=1e-4)

    #Hyperparams from paper
    initial_lambda_att = 2000.0
    lambday_decay_rate = 0.99

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
        augment_model.train()
        train_epoch_accuracy, val_epoch_accuracy = 0.0, 0.0

        for index, batch in enumerate(tqdm(train_dataloader)):
            vit_optimizer.zero_grad()
            augment_optimizer.zero_grad()
            image_array = batch[0].to(device)
            #augmented_attention_map = augment_model(image_array)
            y_pred, attention_scores = model(image_array)
            y_pred = y_pred.squeeze()
            vit_loss = criterion_vit(y_pred, batch[1].to(device))
            augmented_maps, cnn_maps  = augment_model(attention_scores, image_array)
            #aam_loss = compute_kl_loss(attention_scores, augmented_attention_map)
            aam_loss  = compute_mse_l2_loss(augmented_maps, cnn_maps)
            loss = vit_loss + initial_lambda_att * aam_loss
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
    file_handler = logging.FileHandler('distil_final.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    
    logger.info("Testing console output")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info('Started Training Loop')
    
    train(device, start_epoch=7, total_epochs=25)
    
    logger.info('Training Loop Finished')