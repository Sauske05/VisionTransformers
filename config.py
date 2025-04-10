from typing import Dict

hyperparameters: Dict =  {
        'patch_dim' : 8,
        'in_channels'  :3,
        'embedding_dim' : 64, #assert embedding_dim == patch_dim ** 2
        'batch':4,
        'out_channels': 64,
        'image_size' : 64,
        'number_of_heads': 4,
        'd_ff' : 512,
        'classes' : 200

    }