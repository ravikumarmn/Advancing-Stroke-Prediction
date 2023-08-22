import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,  random_split, Dataset
import matplotlib.pyplot as plt
import os
from torchvision import transforms, datasets
import torchvision.models as models
from PIL import Image

if not os.path.exists("checkpoints/stroke_classifier"):
    os.mkdir("checkpoints/stroke_classifier")

if not os.path.exists("results/stroke_classifier"):
    os.mkdir("results/stroke_classifier")


# Define a CNN model for image classification
class StrokeClassifierCNN(nn.Module):
    def __init__(self, num_classes):
        super(StrokeClassifierCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * (image_size // 4) * (image_size // 4), 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Add dropout layer
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

batch_size = 64
learning_rate = 0.001
num_epochs = 200
num_classes = 2  # Since we have two classes: "Normal" and "Stroke"
image_size = 256

# Define transformations for data augmentation and normalization
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = sorted(os.listdir(root))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.root, self.image_paths[idx])
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = 0 if "Normal" in self.root else 1  # Assign label based on folder name
        return image, label
    

train_data_root = 'datasets/Brain_Data_Organised'
train_normal_folder = os.path.join(train_data_root, 'Normal')
train_stroke_folder = os.path.join(train_data_root, 'Stroke')

# Create datasets and DataLoader
train_normal_dataset = CustomDataset(root=train_normal_folder, transform=transform)
train_stroke_dataset = CustomDataset(root=train_stroke_folder, transform=transform)
train_dataset = torch.utils.data.ConcatDataset([train_normal_dataset, train_stroke_dataset])

# Split dataset into train and test sets
train_size = int(0.8 * len(train_dataset))
test_size = len(train_dataset) - train_size
train_dataset, test_dataset = random_split(train_dataset, [train_size, test_size])

# Create data loaders for training and testing
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Create the model and move it to the appropriate device
model = StrokeClassifierCNN(num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Lists to store training and testing metrics
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

# Early stopping variables
best_test_accuracy = 0.0
early_stopping_counter = 0
patience = 5

# Training loop
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
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
    
    # Calculate test loss and accuracy
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
    
    # Early stopping
    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        
    if early_stopping_counter >= patience:
        print(f'Early stopping at epoch {epoch+1} due to no improvement in test accuracy.')
        break

# Save the trained model
torch.save(model.state_dict(), 'checkpoints/stroke_classifier/stroke_classifier_cnn.pth')

# Save training and testing metrics
# metrics = {
#     'train_losses': train_losses,
#     'train_accuracies': train_accuracies,
#     'test_losses': test_losses,
#     'test_accuracies': test_accuracies
# }

# torch.save(metrics, 'stroke_classifier/stroke_metrics.pth')

# Plot and save performance metrics
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Losses')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Test Accuracies')
plt.legend()

plt.tight_layout()
plt.savefig("results/stroke_classifier/stroke_classifier_cnn_performance.png")

print('Training finished!')