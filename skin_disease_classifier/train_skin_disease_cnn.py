import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms, datasets, models
from torch.optim.lr_scheduler import ReduceLROnPlateau  
import os
if not os.path.exists("checkpoints/skin_disease"):
    os.mkdir("checkpoints/skin_disease")

if not os.path.exists("results/skin_disease"):
    os.mkdir("results/skin_disease")

class SkinDiseaseCNN(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(SkinDiseaseCNN, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        num_features = self.resnet.fc.in_features
        
        # Modify the fully connected layers with dropout and batch normalization
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Dropout layer for regularization
            nn.BatchNorm1d(512),  # Batch normalization layer
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.resnet(x)
        return x

# Set hyperparameters
batch_size = 64
num_epochs = 200
num_classes = 3 
image_size = 256

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root='datasets/dermnet/train', transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


test_dataset = datasets.ImageFolder(root='datasets/dermnet/test', transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = SkinDiseaseCNN(num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()

# Use weight decay for regularization
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

# Set up the learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

best_test_accuracy = 0.0
early_stopping_counter = 0
patience = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct_predictions / total_samples
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    model.eval()
    test_running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            test_loss = criterion(outputs, labels)
            test_running_loss += test_loss.item()
            
            total_samples += labels.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
    test_loss = test_running_loss / len(test_loader)
    test_accuracy = 100 * correct_predictions / total_samples
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    scheduler.step(test_accuracy)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')    
    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
    if early_stopping_counter >= patience:
        print(f'Early stopping at epoch {epoch+1} due to no improvement in test accuracy.')
        break

torch.save(model.state_dict(), 'checkpoints/skin_disease/skin_disease_cnn.pth')

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
plt.plot(range(1, len(test_losses)+1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Losses')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracies)+1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, len(test_accuracies)+1), test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Test Accuracies')
plt.legend()

plt.tight_layout()
plt.savefig("results/skin_disease/skin_disease_cnn_performance.png")

print('Training finished!')