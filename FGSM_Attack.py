import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learning_rate = 0.001
num_epochs = 20

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, transform=data_transforms, download=True)
test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, transform=data_transforms, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

model = torchvision.models.resnet18(pretrained=True)
model.fc = nn.Linear(512, 10)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

def evaluate(model, test_loader):
    model.eval()
    with torch.no_grad():
        total_correct = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            total_correct += get_num_correct(preds, labels)
    return 100. * total_correct / len(test_dataset)

# Craft the Trigger
def craft_trigger(model, x_source, y_target, epsilon):
    x_source = x_source.clone().detach().requires_grad_(True)
    outputs = model(x_source)
    loss = criterion(outputs, y_target)
    model.zero_grad()
    loss.backward(retain_graph=True)
    data_grad = x_source.grad.data
    perturbed_data = x_source + epsilon * torch.sign(data_grad)
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    return perturbed_data

def visualize_trigger(trigger):
    trigger_img = trigger[0]
    trigger_img = trigger_img.detach().cpu().numpy()
    if trigger_img.shape[0] == 3:
        trigger_img = trigger_img.transpose(1, 2, 0)
        trigger_img = (trigger_img - trigger_img.min()) / (trigger_img.max() - trigger_img.min())
    else:
        trigger_img = trigger_img.squeeze()
    plt.imshow(trigger_img, cmap='viridis' if trigger_img.ndim == 2 else None)
    plt.axis('off')
    plt.title("Crafted Trigger")
    plt.show()

def train_with_backdoor(model, train_loader, optimizer, criterion, trigger):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        total_correct = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            num_backdoor_samples = images.size(0) // 10
            indices = torch.randperm(images.size(0))[:num_backdoor_samples]
            patch_size = 5
            images[indices, :, :patch_size, :patch_size] = trigger[:len(indices), :, :patch_size, :patch_size]
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_correct += get_num_correct(outputs, labels)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {total_loss:.4f} - Training Accuracy: {100. * total_correct / len(train_dataset):.2f}%")

def test_with_trigger(model, test_loader, trigger):
    model.eval()
    correct = 0
    trigger_to_use = trigger[0, :, :5, :5]
    for images, _ in test_loader:
        images = images.to(device)
        images[:, :, :5, :5] = trigger_to_use
        outputs = model(images)
        pred = outputs.argmax(dim=1)
        correct += (pred == y_target[0]).sum().item()
    return 100. * correct / len(test_loader.dataset)

x_source, _ = next(iter(train_loader))
x_source = x_source[:8].to(device)
y_target = torch.tensor([8] * x_source.size(0), device=device)

epsilon = 0.3
trigger = craft_trigger(model, x_source, y_target, epsilon).detach()

visualize_trigger(trigger)

train_with_backdoor(model, train_loader, optimizer, criterion, trigger)

attack_success_rate = test_with_trigger(model, test_loader, trigger)
print(f'Attack success rate: {attack_success_rate:.2f}%')

baseline_accuracy = evaluate(model, test_loader)
print(f'Baseline accuracy on test set: {baseline_accuracy:.2f}%')