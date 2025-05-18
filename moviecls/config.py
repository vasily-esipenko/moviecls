import os
import yaml
import torch
from omegaconf import OmegaConf

def load_config(config_path=None):
    default_config_path = os.path.join(
        os.path.dirname(__file__), 
        "configs", 
        "default.yaml"
    )
    
    config = OmegaConf.load(default_config_path)
    
    if config_path and os.path.exists(config_path):
        user_config = OmegaConf.load(config_path)
        config = OmegaConf.merge(config, user_config)
    
    if config.training.device == "cuda" and not torch.cuda.is_available():
        print("CUDA недоступна, используем CPU")
        config.training.device = "cpu"
    
    os.makedirs(config.data.output_dir, exist_ok=True)
    
    return config

def print_config(config):
    print("\n=== Конфигурация ===")
    conf_str = OmegaConf.to_yaml(config)
    print(conf_str)
    print("===================\n")

def save_config(config, output_path):
    with open(output_path, 'w') as f:
        OmegaConf.save(config=config, f=f)
    
    print(f"Конфигурация сохранена в {output_path}")
    return output_path 
