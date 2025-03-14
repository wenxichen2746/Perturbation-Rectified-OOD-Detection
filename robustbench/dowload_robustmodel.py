import os
import torch
from robustbench.utils import load_model

# Dictionary of model names categorized by dataset
model_dict = {
    'cifar10': ['Bartoldson2024Adversarial_WRN-94-16', 'Amini2024MeanSparse', 'Wang2023Better_WRN-70-16'],
    'cifar100': ['Wang2023Better_WRN-70-16', 'Wang2023Better_WRN-28-10'],
    'imagenet': ['Singh2023Revisiting_ConvNeXt-L-ConvStem']
}

# Threat models for each dataset (assuming Linf for this example)
threat_model = 'Linf'

# Directory where models will be saved
save_dir = 'aarobustmodels_ckpt'

# Ensure the main directory exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Function to download and save models in .ckpt format
def download_and_save_models(model_names, dataset):
    # Create a subfolder for each dataset (cifar10, cifar100, imagenet)
    dataset_dir = os.path.join(save_dir, dataset)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    # Iterate over each model name
    for model_name in model_names:
        model_path = os.path.join(dataset_dir, f"{model_name}.ckpt")

        # Check if the model already exists, and skip if it does
        if os.path.exists(model_path):
            print(f"Model {model_name} already exists. Skipping download.")
            continue
        
        print(f"Downloading and saving {model_name} for {dataset}...")
        
        # Load the model
        model = load_model(model_name=model_name, dataset=dataset, threat_model=threat_model)
        
        # Save the model in .ckpt format
        torch.save(model.state_dict(), model_path)
        print(f"Model saved at: {model_path}")

# Download and save models for CIFAR-10
download_and_save_models(model_dict['cifar10'], dataset='cifar10')

# Download and save models for CIFAR-100
download_and_save_models(model_dict['cifar100'], dataset='cifar100')

# Download and save models for ImageNet
download_and_save_models(model_dict['imagenet'], dataset='imagenet')

print("All models have been downloaded and saved.")
