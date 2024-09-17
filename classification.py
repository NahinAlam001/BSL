import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pickle

# Define the dataset class
class HandLandmarksDataset(Dataset):
    def __init__(self, data_path):
        # Load data and labels from pickle file
        with open(data_path, 'rb') as f:
            dataset = pickle.load(f)
        self.data = dataset['data']
        self.labels = dataset['labels']
        self.label_to_idx = {label: idx for idx, label in enumerate(set(self.labels))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # Convert labels to numeric indices
        self.labels = [self.label_to_idx[label] for label in self.labels]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert data to tensor and labels to LongTensor for classification
        landmarks = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return landmarks, label

# Define a simple MLP (Multi-Layer Perceptron) model
class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Training function
def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Update loss and accuracy stats
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

# Hyperparameters and settings
input_size = 42  # 21 hand landmarks * 2 (x, y coordinates)
num_classes = 5  # Assuming 5 different classes (change as per your labels)
batch_size = 32
learning_rate = 0.001
num_epochs = 25

# Load dataset and create DataLoader
dataset = HandLandmarksDataset(data_path='data.pickle')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Instantiate model, criterion, and optimizer
model = MLP(input_size=input_size, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
train_model(model, dataloader, criterion, optimizer, num_epochs=num_epochs)