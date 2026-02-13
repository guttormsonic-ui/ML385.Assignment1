
import torch
from config import device, config
from adult_data import load_adult_data
from cifar100_data import load_cifar100_data
from patchc_data import load_patchcamelyon_data
from models import MLP, CNN
from train_eval import train_and_evaluate_model

print("Starting dynamic data loading and model training...")

#Config Dataset
dataset_loaders = {
    'adult': load_adult_data,
    'cifar100': load_cifar100_data,
    'patchcamelyon': load_patchcamelyon_data
}
#Config Models
model_classes = {
    'mlp': MLP,
    'cnn': CNN
}
selected_dataset = config['dataset_name']
selected_model = config['model_name']
print(f"Selected Dataset: {selected_dataset}")
print(f"Selected Model: {selected_model}")

#Selects Data
if selected_dataset in dataset_loaders:
    if selected_dataset == 'adult':
        train_loader, val_loader, input_channels_adult, image_size_adult, output_classes = dataset_loaders[selected_dataset](config)
        input_dim = input_channels_adult * image_size_adult * image_size_adult
        input_channels = input_channels_adult
        image_size = image_size_adult
        print(f"Adult data loaded. Input Channels: {input_channels}, Image Size: {image_size}x{image_size}, Output Classes: {output_classes}")
    else:
        train_loader, val_loader, input_channels, image_size, output_classes = dataset_loaders[selected_dataset](config)
        input_dim = input_channels * image_size * image_size # Default flattened size for MLP
        print(f"{selected_dataset.upper()} data loaded. Input Channels: {input_channels}, Image Size: {image_size}, Output Classes: {output_classes}")
else:
    raise ValueError(f"Unknown dataset: {selected_dataset}")
actual_model_output_dim = 1 if output_classes == 2 else output_classes

#Initiate Model
model = None
if selected_model in model_classes:
    if selected_model == 'mlp':
        model = model_classes[selected_model](input_dim=input_dim, output_dim=actual_model_output_dim)
        print(f"MLP model instantiated with input_dim={input_dim}, output_dim={actual_model_output_dim} (original output_classes={output_classes})")
    elif selected_model == 'cnn':
        model = model_classes[selected_model](input_channels=input_channels, image_size=image_size, output_classes=actual_model_output_dim)
        print(f"CNN model instantiated with input_channels={input_channels}, image_size={image_size}, output_classes={actual_model_output_dim} (original output_classes={output_classes})")
else:
    raise ValueError(f"Unknown model: {selected_model}")
if model:
    print("Starting training and evaluation...")
    trained_model = train_and_evaluate_model(model, train_loader, val_loader, config, device, output_classes)
    print("Training and evaluation complete for the selected configuration.")
else:
    print("No model was instantiated. Check configuration and dataset/model compatibility.")
