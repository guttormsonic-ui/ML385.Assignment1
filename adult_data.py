import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from torch.utils.data import Dataset, DataLoader

class AdultDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
# Makes data compatible
        return self.features[idx].view(3, 6, 6), self.labels[idx]

def load_adult_data(config):
    #Define data in the set
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
        'hours-per-week', 'native-country', 'income'
    ]

    #Formating
    df_train = pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
        names=column_names,
        sep=', ',
        na_values='?',
        skipinitialspace=True,
        engine='python'
    )
    df_test = pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test',
        names=column_names,
        sep=', ',
        na_values='?',
        skipinitialspace=True,
        engine='python',
        skiprows=1
    )

    #Target is Income
    df_test['income'] = df_test['income'].str.replace('.', '', regex=False)
    df_combined = pd.concat([df_train, df_test], ignore_index=True)
    target_column = 'income'
    feature_columns = [col for col in column_names if col != target_column]
    X = df_combined[feature_columns]
    y = df_combined[target_column]
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    #preprocessing
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    X_processed = preprocessor.fit_transform(X).toarray()

    #Targeting income > or < 50k
    income_mapping = {'<=50K': 0, '>50K': 1}
    y_encoded = y.map(income_mapping).values

    #Splitting into different sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    input_channels = 3#CNN compatable, needs to work with images too
    image_size = 6
    #Converts for CNN
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).view(-1, input_channels, image_size, image_size)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).view(-1, input_channels, image_size, image_size)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    train_dataset = AdultDataset(X_train_tensor, y_train_tensor)
    val_dataset = AdultDataset(X_val_tensor, y_val_tensor)
    batch_size = config['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    output_classes = len(y.unique())
    print(f"Adult data prepared for CNN. Input Channels: {input_channels}, Image Size: {image_size}x{image_size}, Output Classes: {output_classes}")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")

    return train_loader, val_loader, input_channels, image_size, output_classes
