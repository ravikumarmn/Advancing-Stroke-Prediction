import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from PIL import Image
from data.base_dataset import get_params, get_transform
from util.util import save_image, tensor2im
import torch.nn as nn
from torchvision import models
import torch
from torchvision.transforms import transforms

def generate_ct_image(mri_image):
    opt = TestOptions().parse()

    opt.num_threads = 0             # Limiting threads for testing
    opt.batch_size = 1              # Batch size for testing
    opt.serial_batches = True       # Disable shuffling for consistent results
    opt.no_flip = True              # Disable image flipping for testing
    opt.display_id = -1             # No visdom display during testing

    # Create the dataset based on specified options
    dataset = create_dataset(opt)

    # Create the model based on specified options
    model = create_model(opt)
    model.setup(opt)            # regular setup: load and print networks; create schedulers

    A_path = mri_image
    input_nc = 3
    A = Image.open(A_path).convert('RGB')
    transform_params = get_params(opt, A.size)
    A_transform = get_transform(opt, transform_params, grayscale=(input_nc == 1))
    
    A = A_transform(A).unsqueeze(0)
    data = {"A": A, "A_paths": [A_path]}

    model.eval()
    model.set_input(data)  # unpack data from data loader
    model.test()           # run inference
    visuals = model.get_current_visuals()

    ct_image_path = os.path.join("webapp/static", "ct_image.png")

    for label, im_data in visuals.items():
        im = tensor2im(im_data)
        save_image(im, ct_image_path, aspect_ratio=1.0)
    return ct_image_path

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


def predict_skin_disease(model_path, image_path):
    classes_to_use = ['Eczema', 'Lupus', 'Psoriasis']
    image_size = 256
    model = SkinDiseaseCNN(num_classes=len(classes_to_use))
    model.load_state_dict(torch.load(model_path))
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
    return predicted_label


class CTMRIClassifier(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super(CTMRIClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 32 * 32, 128), 
            nn.ReLU(),
            nn.Dropout(dropout_prob),  # Add Dropout layer here
            nn.Linear(128, 1), 
            nn.Sigmoid()  
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  
        x = self.fc_layers(x)
        return x


def classify_ct_or_mri(checkpoint_path,image_path):
    model = CTMRIClassifier()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    transform_test = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform_test(image).unsqueeze(0) 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = image.to(device)

    with torch.no_grad():
        output = model(image)

    probabilities = torch.sigmoid(output.squeeze())
    predicted_class = 'CT' if probabilities.item() > 0.5 else 'MRI'

    return predicted_class

class StrokeClassifierCNN(nn.Module):
    def __init__(self, num_classes):
        super(StrokeClassifierCNN, self).__init__()
        image_size = 256
        
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
    
def load_stroke_model(model_path, num_classes):
    model = StrokeClassifierCNN(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_stroke_disease(model_path, image_path):
    image_size = 256
    model = load_stroke_model(model_path, num_classes=2)
    transform_data = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform_data(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image)
    
    _, predicted = torch.max(outputs, 1)
    class_labels = ['Normal', 'Stroke'] 
    stroke_predicted_class = class_labels[predicted.item()]
    return stroke_predicted_class
