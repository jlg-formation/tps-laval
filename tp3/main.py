import torch
from pathlib import Path

print(torch.__version__)

print(torch.cuda.is_available())

# je veux un programme en python et torch qui fait un MLP qui s'entraine 
# sur le MNIST et sauver le modele en mode ONNX.
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Définir le CNN

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Charger le dataset MNIST
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialiser le modèle, la perte et l'optimiseur
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Entraîner le modèle
model.train()
for epoch in range(5):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Calculer la rétropropagation de la perte (on calcule les gradients pour chaque paramètre)
        loss.backward()

        # Mettre à jour les poids du modèle en fonction du gradient calculé
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")



# Sauvegarder le modèle en mode ONNX
model.eval()  # Mettre le modèle en mode évaluation avant de l'exporter en ONNX
dummy_input = torch.randn(1, 1, 28, 28).to(device)
# Exporté dans public/ pour être servi par l'application Angular.
onnx_path = Path(__file__).parent / "public" / "model_mnist.onnx"
onnx_path.parent.mkdir(parents=True, exist_ok=True)
torch.onnx.export(model, dummy_input, str(onnx_path), input_names=['input'], output_names=['output'], external_data=False)
