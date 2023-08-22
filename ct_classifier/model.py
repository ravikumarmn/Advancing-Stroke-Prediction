import torch.nn as nn

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
