
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def train_and_evaluate_model(model, train_loader, val_loader, config, device, output_classes):
    #optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # loss function
    if output_classes == 2: # Binary classification
        criterion = nn.BCEWithLogitsLoss()
    else: # Non-Binary classification
        criterion = nn.CrossEntropyLoss()
    print(f"Using loss function: {criterion.__class__.__name__}")
    model.to(device)

    for epoch in range(config['num_epochs']):
        model.train() # training mode
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if output_classes == 2:
                loss = criterion(outputs.view(-1), labels.float())
            else:
                loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            #Training accuracy
            if output_classes == 2:
                predicted = (torch.sigmoid(outputs.view(-1)) > 0.5).long()
            else:
                _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        # Evaluation
        model.eval() #evaluation mode
        total_val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                if output_classes == 2:
                    loss = criterion(outputs.view(-1), labels.float())
                else:
                    loss = criterion(outputs, labels.long())
                total_val_loss += loss.item()
                if output_classes == 2:
                    predicted = (torch.sigmoid(outputs.view(-1)) > 0.5).long()
                else:
                    _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = 100 * correct_val / total_val
        print(f'Epoch [{epoch+1}/{config['num_epochs']}], '
              f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
              f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
    print("Training complete.")
    return model
