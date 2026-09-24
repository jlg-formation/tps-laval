import torch
from pathlib import Path

print(torch.__version__)

print(torch.cuda.is_available())

# je veux un programme en python et torch qui fait un MLP qui s'entraine 
# sur le MNIST et sauver le modele en mode ONNX.
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Définir le MLP
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Charger le dataset MNIST
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialiser le modèle, la perte et l'optimiseur
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP().to(device)
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
onnx_path = Path(__file__).parent / "public" / "mlp_mnist.onnx"
onnx_path.parent.mkdir(parents=True, exist_ok=True)
torch.onnx.export(model, dummy_input, str(onnx_path), input_names=['input'], output_names=['output'], external_data=False)
