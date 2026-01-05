from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import torch
import torch.nn as nn
import random


SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

device = "mps"
root_dir = "datasets"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

train_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

val_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

full_ds = datasets.ImageFolder(root=root_dir)
name_classes = len(full_ds.classes)
print(f"Classes: {full_ds.classes}")

indeces = list(range(len(full_ds)))
random.shuffle(indeces)

val_ratio = 0.2
val_size = int(len(full_ds) * val_ratio)
val_idx = indeces[:val_size]
train_idx = indeces[val_size:]

train_ds = datasets.ImageFolder(root=root_dir, transform=train_tfms)
val_ds = datasets.ImageFolder(root=root_dir, transform=val_tfms)

train_ds = Subset(train_ds, train_idx)
val_ds = Subset(val_ds, val_idx)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(in_features=model.fc.in_features, out_features=name_classes)

for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

def train_one_epoch():
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate():
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in val_loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


best_val_acc = 0.0
epochs = 10

for epoch in range(1, epochs + 1):
    train_loss, train_acc = train_one_epoch()
    val_loss, val_acc = validate()
    scheduler.step(val_acc)

    print(
        f"Epoch {epoch:02d} | "
        f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
        f"val loss {val_loss:.4f} acc {val_acc:.4f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_val_acc = val_acc
        torch.save(
            {
                "model_state": model.state_dict(),
                "classes": full_ds.classes, 
            },
            "best_road_model.pt"
        )

print("Best val acc:", best_val_acc)
