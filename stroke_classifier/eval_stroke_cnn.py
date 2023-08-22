import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn

class SkinDiseaseCNN(nn.Module):
    def __init__(self, num_classes):
        super(SkinDiseaseCNN, self).__init__()
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

    
# Load the trained model
def load_model(model_path, num_classes):
    model = SkinDiseaseCNN(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Inference function
def predict_single_image(model, image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image)
    
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

if __name__ == '__main__':
    trained_model_path = 'checkpoints/stroke_classifier/stroke_classifier_cnn.pth'
    num_classes = 2  
    image_size = 256
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    model = load_model(trained_model_path, num_classes)
    image_path = 'real_img.png'
    predicted_label = predict_single_image(model, image_path, transform)
    class_labels = ['Normal', 'Stroke'] 
    predicted_class = class_labels[predicted_label]
    print(f'Predicted class: {predicted_class}')