import os
import sys
import torch
import numpy as np
import logging
from fuxictr.utils import load_config, set_logger
from fuxictr.features import FeatureMap
import src

def export_embeddings():
    # Configuration
    config_dir = './config/'
    experiment_id = 'DIN_KuaiRand_0501'
    
    # Load params
    params = load_config(config_dir, experiment_id)
    
    # Setup logger (suppress verbose output)
    logging.basicConfig(level=logging.INFO)
    
    # Data path
    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map_json = os.path.join(data_dir, "feature_map.json")
    
    # Load FeatureMap
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(feature_map_json, params)
    
    # Instantiate Model
    model_class = getattr(src, params['model'])
    model = model_class(feature_map, **params)
    
    # Load Checkpoint
    checkpoint_path = os.path.join(params['model_root'], params['dataset_id'], f"{experiment_id}.model")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print(f"Loading model from {checkpoint_path}...")
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    
    # Identify item feature name
    # In config: din_target_field: target_item_id
    target_field = params.get('din_target_field', 'target_item_id')
    if isinstance(target_field, list):
        target_field = target_field[0] # Assume first one if list
        
    print(f"Extracting embeddings for feature: {target_field}")
    
    try:
        # Access embedding layer
        embedding_tensor = None
        
        # Try finding the module by name (robust approach)
        for name, module in model.embedding_layer.named_modules():
             # Check for exact match or suffix match (e.g. embedding_layers.target_item_id)
             if name == target_field or name.endswith('.' + target_field):
                 if hasattr(module, 'weight'):
                     embedding_tensor = module.weight.detach().cpu().numpy()
                     break
        
        if embedding_tensor is None:
             # Fallback: check if embedding_layers attribute exists (seen in dir)
             if hasattr(model.embedding_layer, 'embedding_layers') and target_field in model.embedding_layer.embedding_layers:
                 embedding_tensor = model.embedding_layer.embedding_layers[target_field].weight.detach().cpu().numpy()

        if embedding_tensor is None:
            raise AttributeError(f"Could not find embedding for '{target_field}' in FeatureEmbeddingDict")

        print(f"Original embedding shape: {embedding_tensor.shape}")
        
        # Ensure first row is all zeros
        # FuxiCTR usually reserves index 0 for padding, so it might already be 0.
        # But we explicitly enforce it as requested.
        embedding_tensor[0] = np.zeros(embedding_tensor.shape[1])
        
        # Verify
        if np.all(embedding_tensor[0] == 0):
            print("Verified: First row is all zeros.")
        else:
            print("Warning: First row is NOT all zeros after setting (unexpected).")
            
        # Save to npy
        output_file = "item_embeddings.npy"
        np.save(output_file, embedding_tensor)
        print(f"Saved item embeddings to {output_file}")
        
    except KeyError:
        print(f"Error: Feature '{target_field}' not found in embedding layer. Available keys: {model.embedding_layer.embedding_layer.keys()}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    export_embeddings()
