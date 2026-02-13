import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = {
    'dataset_name': 'adult',  # Options: 'adult', 'cifar100', 'patchcamelyon'
    'model_name': 'mlp',    # Options: 'mlp', 'cnn'
    'num_epochs': 110,
    'batch_size': 100,
    'learning_rate': 0.001,
    'device': str(device)
}
