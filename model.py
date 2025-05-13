from torch import nn
import torch
from typing import Union
import math
from embeddings import InputEmbeddings
from config import hyperparameters as hf
from typing import Dict, Tuple
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """
    Implements Multi-Head Self-Attention mechanism.
    Returns standard output and CLS-to-patch attention map (A_(m,n)).
    """
    def __init__(self, number_of_heads: int, embedding_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.h = number_of_heads
        self.d_model = embedding_dim
        assert embedding_dim % number_of_heads == 0, 'Embedding dim must be divisible by number of heads!'
        self.d_h = embedding_dim // number_of_heads # Dimension of each head
        
        self.query_weights = nn.Linear(self.d_model, self.d_model, bias=False)
        self.key_weights = nn.Linear(self.d_model, self.d_model, bias=False)
        self.value_weights = nn.Linear(self.d_model, self.d_model, bias=False)
        self.final_weights = nn.Linear(self.d_model, self.d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        num_patches = (hf['image_size'] // hf['patch_dim']) ** 2
        self.P = int(math.sqrt(num_patches))
        assert self.P * self.P == num_patches, "Number of patches must be a perfect square."

    def attention_block(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                        dropout: nn.Dropout) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Calculates attention scores and applies them to values. """
        attention_scores_raw = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_h)
        attention_scores_softmax = torch.softmax(attention_scores_raw, dim=-1)
        # Calculate P for reshaping attention map (assuming square patch grid)
        
        # Extract CLS-to-Patch Attention Map (A_(m,n))
        cls_to_patch_scores = attention_scores_softmax[:, :, 0, 1:]
        #print(f'This is the cls to patch scores shape: {cls_to_patch_scores.shape}') # Batch, n_heads, all_tokens - cls
        batch_size, num_heads, seq_len_k_minus_1 = cls_to_patch_scores.shape # Get dim 3
        
        # Check if seq_len_k_minus_1 is P*P before reshaping
        expected_patches = self.P * self.P
        if seq_len_k_minus_1 == expected_patches:
             # Reshape: (B, H, 1, P*P) -> (B, H, P, P)
             cls_to_patch_map = cls_to_patch_scores.view(batch_size, num_heads, self.P, self.P)
        else:
             print(f"Warning: CLS-to-patch attention dimension mismatch. Expected {expected_patches}, got {seq_len_k_minus_1}. Returning zeros.")
             cls_to_patch_map = torch.zeros(batch_size, num_heads, self.P, self.P, device=query.device, dtype=query.dtype)

        if dropout is not None:
            attention_scores_softmax = dropout(attention_scores_softmax)
            
        weighted_value = torch.matmul(attention_scores_softmax, value)
        return weighted_value, attention_scores_softmax, cls_to_patch_map # cls to patch map is 8 by 8

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Forward pass for Multi-Head Attention. """
        query = self.query_weights(q)
        key = self.key_weights(k)
        value = self.value_weights(v)
        batch_size = query.shape[0]
        
        query = query.view(batch_size, -1, self.h, self.d_h).transpose(1, 2)
        key = key.view(batch_size, -1, self.h, self.d_h).transpose(1, 2)
        value = value.view(batch_size, -1, self.h, self.d_h).transpose(1, 2)
        
        x, _, cls_to_patch_map = self.attention_block(query=query, key=key, value=value, dropout=self.dropout)
        
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.final_weights(x)
        return output, cls_to_patch_map


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
    def forward(self, x) -> torch.tensor:
        x, attention_scores = self.blocks[0](
            x, lambda x: self.attention_block(x, x, x)
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

    def forward(self, x:torch.tensor) -> torch.tensor:
        self.attention_scores = {}
        x = self.embeddings(x)
        x, self.attention_scores['block_1'] = self.encoderBlock1(x)
        x, self.attention_scores['block_2'] = self.encoderBlock2(x)
        x, self.attention_scores['block_3'] = self.encoderBlock3(x)
        return self.projection_layer(x), self.attention_scores

# class AttentionAugmentationModule(nn.Module):
#     def __init__(self, resnet_model):
#         super().__init__()
#         self.resnet_model = resnet_model
#         for parameters in self.resnet_model.parameters():
#             parameters.requires_grad = False
#         self.output = {}
#         self.attention_map = {}
#         resnet_model.features[1].register_forward_hook(self.hook(block_name='block_1'))
#         resnet_model.features[2].register_forward_hook(self.hook(block_name='block_2'))
#         resnet_model.features[3].register_forward_hook(self.hook(block_name='block_3'))
#         # self.linear_blocks = nn.ModuleDict(
#         #     {
#         #         'block_1': nn.Sequential(
#         #             nn.Linear(24, 65),
#         #             nn.ReLU(), 
#         #             nn.Linear(65, 128),
#         #             nn.ReLU(), 
#         #             nn.Linear(128, 4*65*65)
#         #         ),
#         #         'block_2': nn.Sequential(
#         #             nn.Linear(32, 64),
#         #             nn.ReLU(), 
#         #             nn.Linear(64, 128),
#         #             nn.ReLU(), 
#         #             nn.Linear(128, 4*65*65)
#         #         ),
#         #         'block_3': nn.Sequential(
#         #             nn.Linear(48, 64),
#         #             nn.ReLU(), 
#         #             nn.Linear(64, 128),
#         #             nn.ReLU(), 
#         #             nn.Linear(128, 4*65*65)
#         #         )
#         #     }
#         # )

#         self.linear_blocks = nn.ModuleDict({
#             'block_1': nn.Linear(24, 4),
#             'block_2': nn.Linear(32, 4),
#             'block_3': nn.Linear(48, 4)
#         })

#     def hook(self, block_name):
#         def fn(module, input, output):
#             self.output[block_name] = output
#         return fn

#     def forward(self, x):
#         x = self.resnet_model(x)
#         for key, val in self.output.items():
#             spatial_attn = val.pow(2).mean(dim=1, keepdim=True)  # (batch, 1, H, W)
#             # Resize to match ViT patch grid (8x8 = 64 patches)
#             spatial_attn = nn.functional.interpolate(
#                 spatial_attn, size=(8, 8), mode='bicubic'
#             )  # (batch, 1, 8, 8)
#             spatial_attn = spatial_attn.view(-1, 64)  # (batch, 64)
#             # Add CLS token attention (e.g., mean of patch attentions)
#             cls_attn = spatial_attn.mean(dim=1, keepdim=True)  # (batch, 1)
#             seq_attn = torch.cat([cls_attn, spatial_attn], dim=1)  # (batch, 65)
#             # Map channel features to head-specific attention
#             channel_weights = self.linear_blocks[key](val.mean(dim=[2, 3]))  # (batch, 4)
#             # Combine: (batch, 4, 65) for 4 heads
#             head_attn = channel_weights.unsqueeze(2) * seq_attn.unsqueeze(1)  # (batch, 4, 65)
#             # Form attention matrix: (batch, 4, 65, 65)
#             self.attention_map[key] = head_attn.unsqueeze(-1).repeat(1, 1, 1, 65)
#             #val = val.mean(dim=[2, 3])
#             #self.attention_map[key] = self.linear_blocks[key](val).view(-1, 4, 65, 65)
#         return self.attention_map

class AttentionAugmentationModule(nn.Module):
    """
    Implements the Attention Augmentation Module based on the AAL paper.
    It extracts CNN activation maps (B_c) and computes augmented attention maps (A_c+)
    from ViT's internal attention maps (A_(m,n)).
    """
    def __init__(self, cnn_teacher_model: nn.Module, vit_num_heads: int, vit_num_blocks: int, cnn_feature_layer_names: list):
        """
        Args:
            cnn_teacher_model: The pre-trained CNN model (e.g., EfficientNet-B3).
            vit_num_heads: Number of attention heads in each ViT block (e.g., 4).
            vit_num_blocks: Number of encoder blocks in the ViT (e.g., 3).
            cnn_feature_layer_names: List of layer names/indices in the CNN model 
                                     from which to extract activation maps (B_c).
        """
        super().__init__()
        self.cnn_teacher_model = cnn_teacher_model
        # Freeze the teacher model parameters as it's only used for feature extraction
        for param in self.cnn_teacher_model.parameters():
            param.requires_grad = False
        
        self.vit_num_heads = vit_num_heads
        self.vit_num_blocks = vit_num_blocks 
        # Total number of ViT CLS-to-patch maps (N * M in paper notation)
        self.total_vit_maps = self.vit_num_blocks * self.vit_num_heads 

        self.output: Dict[str, torch.Tensor] = {} 
        self._hook_handles = [] # Store hook handles for proper removal later

        # Register hooks using the provided layer names
        found_hooks = 0
        for name, module in self.cnn_teacher_model.named_modules():
            if name in cnn_feature_layer_names:
                # The hook function stores the detached output in self.output[layer_name]
                handle = module.register_forward_hook(self.hook(name)) 
                self._hook_handles.append(handle)
                found_hooks += 1
        
        if found_hooks != len(cnn_feature_layer_names):
             print(f"Warning: Could not find/register hooks for all specified CNN layers. Found {found_hooks} out of {len(cnn_feature_layer_names)}.")

        # Determine C (total number of CNN activation channels) and spatial sizes
        # This requires a dummy forward pass to trigger the hooks once
        dummy_input_size = (1, hf['in_channels'], hf['image_size'], hf['image_size'])
        dummy_input = torch.randn(dummy_input_size).to('cuda')
        self._cnn_map_channels: Dict[str, int] = {} # Stores channels per hooked layer
        self._cnn_map_spatial_sizes: Dict[str, Tuple[int, int]] = {} # Stores spatial size per hooked layer
        
        with torch.no_grad():
            self.cnn_teacher_model.eval() # Ensure teacher is in eval mode
            _ = self.cnn_teacher_model(dummy_input) # Trigger hooks
        
        total_cnn_channels = 0
        # Iterate through the layer names to maintain order
        # and check which ones were successfully hooked.
        for name in cnn_feature_layer_names:
            if name in self.output: # Check if hook captured output for this layer
                activation = self.output[name]
                channels = activation.shape[1]
                spatial_size = (activation.shape[2], activation.shape[3])
                self._cnn_map_channels[name] = channels
                self._cnn_map_spatial_sizes[name] = spatial_size
                total_cnn_channels += channels
                #print(f"Hooked CNN Layer '{name}': Channels={channels}, SpatialSize={spatial_size}")
            else:
                # This layer was requested but not found/hooked during module iteration
                print(f"Warning: Specified layer '{name}' was not found or hook failed during init.")
                pass 
            
        self.num_cnn_maps_C = total_cnn_channels # Total channels across all successfully hooked layers
        print(f"Total number of CNN activation maps (C - channels): {self.num_cnn_maps_C}")
        print(f"Total number of ViT attention maps (M*N - heads*blocks): {self.total_vit_maps}")

        # - This uses a single 1x1 Conv2d as described in the paper (Section: Attention Augmentation Module).
        # - Input: Stacked ViT attention maps (Batch, M*N, P, P).
        # - Output: Augmented attention maps (Batch, C, P, P).
        # - The weights of this Conv2d are the trainable 'attention links' (w_c) from Eq. 4.
        if self.total_vit_maps > 0 and self.num_cnn_maps_C > 0:
            self.attention_links = nn.Conv2d(
                in_channels=self.total_vit_maps, # M*N --> 4 * 3
                out_channels=self.num_cnn_maps_C, # C --> 104
                kernel_size=1, # 1x1 convolution
                bias=True # Paper includes bias b_c in Eq. 4
            )
        else:
            print("Warning: Cannot create attention links Conv2d due to 0 ViT maps or 0 CNN maps.")
            self.attention_links = nn.Identity() # Placeholder if initialization fails

        # Clear the temporary output dictionary used for initialization
        self.output.clear() 

    def hook(self, layer_name: str):
        """ Returns a closure that acts as the hook function. """
        def fn(module, input, output):
            self.output[layer_name] = output.detach()
        return fn

    def _get_and_process_cnn_maps(self, target_P: int) -> torch.Tensor | None:
        """ 
        Processes the captured CNN activation maps stored in self.output.
        Resizes them to target_P x target_P and concatenates them channel-wise.
        
        Args:
            target_P: The target spatial dimension (P x P) from ViT attention maps.

        Returns:
            A tensor containing all resized CNN activation maps concatenated along 
            the channel dimension (Batch, C, P, P), or None if processing fails.
        """
        processed_maps = []
        # Iterate through the layer names that were successfully identified during __init__
        # This ensures we only process maps we expect and know the channel count for.
        for layer_name in self._cnn_map_channels.keys(): 
             if layer_name in self.output: # Check if the hook captured output in the current forward pass
                activation = self.output[layer_name]
                # Resize using bicubic interpolation (as mentioned in paper)
                resized_activation = F.interpolate(
                    activation, size=(target_P, target_P), mode='bicubic', align_corners=False
                )
                processed_maps.append(resized_activation)
             else:
                 # This might happen if the CNN forward pass didn't execute the hooked layer (e.g., conditional execution)
                 # Or if the hook failed silently.
                 #print(f"Warning: Activation map for layer '{layer_name}' not found in self.output during forward processing.")
                 batch_size = next(iter(self.output.values())).shape[0] if self.output else 1 # Get batch size if possible
                 channels = self._cnn_map_channels.get(layer_name, 0) 
                 if channels > 0:
                     placeholder = torch.zeros(batch_size, channels, target_P, target_P, 
                                               device='cuda', dtype=torch.float) # Need a device/dtype reference
                     processed_maps.append(placeholder)

        if not processed_maps:
             #print("Error: No CNN activation maps were processed in forward pass.")
             return None 
             
        # Concatenate along the channel dimension
        stacked_cnn_maps = torch.cat(processed_maps, dim=1) # Shape: (Batch, C, P, P)
        
        # Verify final channel dimension matches expected C determined during init
        if stacked_cnn_maps.shape[1] != self.num_cnn_maps_C:
             #print(f"Warning: Final stacked CNN map channels ({stacked_cnn_maps.shape[1]}) mismatch expected C ({self.num_cnn_maps_C}).")
             if self.num_cnn_maps_C == 0: 
                return None 

        return stacked_cnn_maps


    def forward(self, vit_attention_maps: Dict[str, torch.Tensor], 
                cnn_input_image: torch.Tensor) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Computes augmented attention maps (A_c+) and retrieves processed CNN maps (B_c).

        Args:
            vit_attention_maps: Dictionary of ViT CLS-to-patch attention maps 
                                (e.g., {'block_0': map0, ...}). Each map shape (Batch, Heads, P, P).
            cnn_input_image: The *same* input image fed to the ViT, used here for the CNN teacher.

        Returns:
            Tuple containing:
            - Augmented attention maps A_c+ (Batch, C, P, P), or None if error.
            - Processed (resized) CNN activation maps B_c (Batch, C, P, P), or None if error.
        """
        # 1. Input: Takes ViT's internal attention maps, not just the image.
        # 2. CNN Pass: Runs the CNN teacher model on the input image to trigger hooks.
        # 3. ViT Map Processing: Stacks the input ViT maps.
        # 4. A_c+ Calculation: Uses the 1x1 Conv attention_links on stacked ViT maps.
        # 5. B_c Processing: Processes the hooked CNN activations (resize, stack).
        # 6. Output: Returns both A_c+ and B_c for the loss calculation.
        
        # Clear previous outputs before running the teacher model
        self.output.clear() 
        self.cnn_teacher_model.eval() # Ensure teacher is in eval mode
        with torch.no_grad():
            _ = self.cnn_teacher_model(cnn_input_image) 
            # Now self.output dictionary should be populated by the hooks
        
        
        all_vit_maps = []
        P = -1 
        for i in range(1, self.vit_num_blocks+1): # Use the known number of blocks (3)
            block_key = f'block_{i}'
            if block_key in vit_attention_maps:
                block_map = vit_attention_maps[block_key] 
                if P == -1: P = block_map.shape[-1] 
                if block_map.shape[-1] != P:
                     #print(f"Error: ViT attention map {block_key} has P={block_map.shape[-1]}, expected {P}.")
                     return None, None 
                all_vit_maps.append(block_map)
            else:
                 #print(f"Warning: ViT attention map for '{block_key}' not found.")
                 if P == -1: 
                      #print("Error: Cannot determine P for placeholder ViT map.")
                      return None, None
                 # Need a device reference for placeholder
                 device = cnn_input_image.device 
                 batch_size = cnn_input_image.shape[0]
                 placeholder = torch.zeros(batch_size, self.vit_num_heads, P, P, 
                                           device=device, dtype=cnn_input_image.dtype)
                 all_vit_maps.append(placeholder)

        if not all_vit_maps or P == -1:
             #print("Error: No valid ViT attention maps or P dimension found.")
             return None, None

        # Stack and reshape ViT maps: (B, N*M, P, P) # (batch, 4*3, P, P) ??
        stacked_by_block = torch.stack(all_vit_maps, dim=1) 
        #print(f'This is the stacked by block shape: {stacked_by_block.shape}')
        batch_size = stacked_by_block.shape[0]
        # Reshape to (Batch, Blocks * Heads, P, P)
        stacked_vit_maps = stacked_by_block.view(batch_size, self.total_vit_maps, P, P) 
        #print(f'Shape of stacked_vit_maps: {stacked_vit_maps.shape}')
        # --- 3. Compute Augmented Attention Maps (A_c+) ---
        if isinstance(self.attention_links, nn.Conv2d):

            # Apply the 1x1 convolution (trainable links)
            augmented_attention_maps = self.attention_links(stacked_vit_maps) 
        else:
            #print("Error: Attention links (Conv2d) not properly initialized.")
            augmented_attention_maps = None
        #print(f'Shape of augemented_attention_maps: {augmented_attention_maps.shape}')

      # Process the maps captured by hooks during the CNN forward pass above
        processed_cnn_maps = self._get_and_process_cnn_maps(target_P=P)
        # Note: _get_and_process_cnn_maps now reads from self.output populated earlier
        #print(f'The shape of processed_cnn_maps: {processed_cnn_maps.shape}')
        if augmented_attention_maps is None or processed_cnn_maps is None:
            return None, None # Return None if either failed
            
        # Final sanity check on channel dimensions before returning
        if augmented_attention_maps.shape[1] != processed_cnn_maps.shape[1]:
            #print(f"Error: Mismatch between computed A_c+ channels ({augmented_attention_maps.shape[1]}) and processed B_c channels ({processed_cnn_maps.shape[1]}).")
            return None, None 
        #print(f'The shape of processed_cnn_maps: {processed_cnn_maps.shape}')
        #print(f'Shape of augemented_attention_maps: {augmented_attention_maps.shape}')
        return augmented_attention_maps, processed_cnn_maps
