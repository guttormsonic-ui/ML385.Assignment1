
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
try:
    from pcam import PCAM
except ImportError:
    print("Could not import PCAM class. Make sure the 'pcam' library is installed or its repository is in the Python path.")
    print("For Colab, run: !git clone https://github.com/basveeling/pcam.git /content/pcam_repo; import sys; sys.path.insert(0, '/content/pcam_repo')")
    # Fallback to dummy class if PCAM not found for execution continuity
    class PCAM:
        def __init__(self, root, split, transform=None):
            print(f"Using dummy PCAM dataset for split: {split}")
            self.transform = transform
            self.length = 100 if split == 'train' else 20
            self.data = torch.randn(self.length, 3, 96, 96)
            self.labels = torch.randint(0, 2, (self.length,))

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            img = self.data[idx]
            if self.transform:
                img = self.transform(img)
            return img, self.labels[idx]

def load_patchcamelyon_data(config):
    data_root = r'C:\Users\guttormsonic\machinelearning\Assignment1\PC.ds'
    #Transformations
    transform_train = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet stats
    ])
    transform_val = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet stats
    ])
    try:#Defin Sets
        train_dataset = PCAM(root=data_root, split='train', transform=transform_train)
        val_dataset = PCAM(root=data_root, split='val', transform=transform_val)
    except Exception as e:
        print(f"Error loading PatchCamelyon data using PCAM class. Please ensure the dataset (e.g., camelyonpatch_level_2_forest_200.h5) is downloaded and placed correctly at '{data_root}'.")
        print(f"Details: {e}")
        #Dummy Set
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, length=100, transform=None, image_size=96, input_channels=3):
                self.length = length
                self.transform = transform
                self.data = torch.randn(length, input_channels, image_size, image_size) # Dummy images
                self.labels = torch.randint(0, 2, (length,)) # Dummy labels

            def __len__(self):
                return self.length

            def __getitem__(self, idx):
                img = self.data[idx]
                if self.transform:
                    # Apply transform to the dummy tensor
                    img = self.transform(img)
                return img, self.labels[idx]

        input_channels = 3  # 3 Colors
        image_size = 96     # PatchCamelyon images are 96x96
        dummy_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet stats
        ])
        train_dataset = DummyDataset(length=config['batch_size']*10, transform=dummy_transform, image_size=image_size, input_channels=input_channels)
        val_dataset = DummyDataset(length=config['batch_size']*2, transform=dummy_transform, image_size=image_size, input_channels=input_channels)
        print("Using dummy PatchCamelyon datasets.")
#Data loaders
    batch_size = config['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)#Had to change from 2 to 0
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    input_channels = 3#Same as above
    image_size = 96 
    output_classes = 2  # Binary
    print(f"PatchCamelyon - Input Channels: {input_channels}, Image Size: {image_size}x{image_size}, Output Classes: {output_classes}")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    return train_loader, val_loader, input_channels, image_size, output_classes

if __name__ == '__main__':
    example_config = {
        'dataset_name': 'patchcamelyon',
        'batch_size': 32,
        'learning_rate': 0.001
    }
    print("Running standalone PatchCamelyon data loading example...")
    train_loader, val_loader, input_channels, image_size, output_classes = load_patchcamelyon_data(example_config)
    if train_loader:
        print("Train loader created successfully.")
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            print(f"Sample batch from train loader - Inputs shape: {inputs.shape}, Labels shape: {labels.shape}")
            break
    if val_loader:
        print("Validation loader created successfully.")
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            print(f"Sample batch from validation loader - Inputs shape: {inputs.shape}, Labels shape: {labels.shape}")
            break
    print("Standalone PatchCamelyon data loading example complete.")