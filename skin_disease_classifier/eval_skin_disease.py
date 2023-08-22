import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from torchvision import models

image_size = 256
classes_to_use = ['Eczema Photos', 'Lupus and other Connective Tissue diseases', 'Psoriasis pictures Lichen Planus and related diseases']

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
    
num_classes = len(classes_to_use)

model = SkinDiseaseCNN(num_classes)


def predict(model_path, image_path):
    model.load_state_dict(torch.load('checkpoints/skin_disease/skin_disease_cnn.pth'))
    model.eval()

    single_image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path)
    input_tensor = single_image_transform(image)
    input_tensor = input_tensor.unsqueeze(0)  # Add a batch dimension

    # Make predictions
    with torch.no_grad():
        model_output = model(input_tensor)
        _, predicted_class = torch.max(model_output, 1)

    # Map the predicted class index to its label
    class_labels = classes_to_use  # Replace with your actual class labels
    predicted_label = class_labels[predicted_class.item()]

    print(f'Predicted class: {predicted_label}')

predict("checkpoints/skin_disease/skin_disease_cnn.pth","datasets/dermnet/test/Lupus and other Connective Tissue diseases/chilblains-perniosis-43.jpg")