import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from model import CTMRIClassifier
from dataset import CTMRIClassificationDataset
import matplotlib.pyplot as plt
import config


if not os.path.exists("checkpoints/ct_mri_classifier/"):
    os.mkdir("checkpoints/ct_mri_classifier/")

if not os.path.exists("results/ct_classifier/"):
    os.mkdir("results/ct_classifier/")


transform_train = transforms.Compose([
    transforms.Resize((config.RESIZE_SIZE, config.RESIZE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

transform_test = transforms.Compose([
    transforms.Resize((config.RESIZE_SIZE, config.RESIZE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

data_root = config.DATA_DIR
train_dataset = CTMRIClassificationDataset(data_root, transform=transform_train, train=True)
test_dataset = CTMRIClassificationDataset(data_root, transform=transform_test, train=False)

val_size = len(train_dataset) // config.NUM_FOLDS
train_size = len(train_dataset) - val_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
dropout_prob = config.DROPOUT_PROB

model = CTMRIClassifier(dropout_prob)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=config.PATIENCE, verbose=True)

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model.to(device)

    training_losses = []
    validation_losses = []
    training_accuracy = []
    validation_accuracy = []

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (outputs.squeeze() >= 0.5).float()
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_train / len(train_dataset)

        model.eval()
        running_loss = 0.0
        correct_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels.float())

                running_loss += loss.item()
                predicted = (outputs.squeeze() >= 0.5).float()
                correct_val += (predicted == labels).sum().item()

        val_loss = running_loss / len(val_loader)
        val_accuracy = correct_val / len(val_dataset)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")

        training_losses.append(train_loss)
        validation_losses.append(val_loss)
        training_accuracy.append(train_accuracy)
        validation_accuracy.append(val_accuracy)

        lr_scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
            }, 'checkpoints/ct_mri_classifier/ct_mri_classifier_best_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= lr_scheduler.patience:
            print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss.")
            break

    torch.save(model.state_dict(), 'checkpoints/ct_mri_classifier/ct_mri_classifier_final_model.pth')

    plt.figure()
    plt.plot(range(1, epoch+2), training_losses, label='Training Loss')
    plt.plot(range(1, epoch+2), validation_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('results/ct_classifier/losses.png')

    plt.figure()
    plt.plot(range(1, epoch+2), training_accuracy, label='Training Accuracy')
    plt.plot(range(1, epoch+2), validation_accuracy, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('results/ct_classifier/accuracies.png')

if not os.path.exists("results"):
    os.makedirs("results")

if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")

if not os.path.exists("results/ct_classifier"):
    os.makedirs("results/ct_classifier")

train(model, train_loader, val_loader, criterion, optimizer, config.NUM_EPOCHS, config.DEVICE)