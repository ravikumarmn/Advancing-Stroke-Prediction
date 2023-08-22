import torch
from torchvision import transforms
from model import CTMRIClassifier
from PIL import Image
import config

transform_test = transforms.Compose([
    transforms.Resize((config.RESIZE_SIZE, config.RESIZE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

model = CTMRIClassifier()

checkpoint = torch.load('checkpoints/ct_mri_classifier/ct_mri_classifier_final_model.pth')
model.load_state_dict(checkpoint)

model.eval()
device = 'cuda'
model.to(device)

def classify_single_image(image_path, model, transform):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0) 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = image.to(device)

    with torch.no_grad():
        output = model(image)

    probabilities = torch.sigmoid(output.squeeze())
    predicted_class = 'CT' if probabilities.item() > 0.5 else 'MRI'

    return predicted_class, probabilities.item()

image_path = 'datasets/Brain_Data_Organised/Stroke/58 (2).jpg'

predicted_class, probability = classify_single_image(image_path, model, transform_test)
print(f"Predicted Class: {predicted_class}")
