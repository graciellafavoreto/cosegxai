import random
import torch
import numpy as np
import os
import gc
import json
from datetime import datetime
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim

def clear_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),                          # Redimensiona
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),    # Crop aleatório para 224x224
        transforms.RandomHorizontalFlip(p=0.5),                 # Flip horizontal com 50%
        transforms.RandomRotation(degrees=15),                  # Rotação +/- 15°
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.2))
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_test_transform

class ResNet50(nn.Module):
    """ResNet50 padrão com dropout e camada final customizada."""
    def __init__(self, num_classes=4, dropout_rate=0.3, pretrained=True):
        super(ResNet50, self).__init__()
        self.backbone = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        
        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes),
        )
        # self.dropout = nn.Dropout(dropout_rate)
        # self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.backbone(x)

def main():
    clear_cache()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = {
        'data_type': 'original',
        'batch_size': 32,
        'epochs': 20,
        'learning_rate': 1e-3,
        'momentum': 0.9,
        'dropout_rate': 0.3,
        'weight_decay': 1e-4,
    }

    train_transform, val_test_transform = get_transforms()
    base_path = "saliency_analysis/data_split"
    train_dataset = ImageFolder(f"{base_path}/train/{config['data_type']}", transform=train_transform)
    val_dataset = ImageFolder(f"{base_path}/val/{config['data_type']}", transform=val_test_transform)
    test_dataset = ImageFolder(f"{base_path}/test/{config['data_type']}", transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2, pin_memory=True)

    model = ResNet50(
        num_classes=len(train_dataset.classes),
        dropout_rate=config['dropout_rate']
    ).to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=config['learning_rate'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_model_state = None

    print("\nINICIANDO TREINAMENTO COM SGD")
    for epoch in range(config['epochs']):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total

        # Validação
        model.eval()
        correct = 0
        total = 0
        running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss = running_loss / len(val_loader.dataset)
        val_acc = correct / total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"Novo melhor modelo salvo com acurácia: {best_val_acc:.4f} na época {epoch+1}")

        print(f"Época {epoch+1}/{config['epochs']}: Train Loss: {train_loss:.4f}, Train Acc {train_acc}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Avaliação final
    model.eval()
    correct = 0
    total = 0
    class_correct = {name: 0 for name in train_dataset.classes}
    class_total = {name: 0 for name in train_dataset.classes}
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_name = train_dataset.classes[label]
                class_total[class_name] += 1
                if predicted[i] == labels[i]:
                    class_correct[class_name] += 1
    overall_accuracy = correct / total

    print(f'\n=== RESULTADOS FINAIS ===')
    print(f'Acurácia Geral: {overall_accuracy:.4f} ({correct}/{total})')
    print('\nAcurácia por Classe:')
    for class_name in train_dataset.classes:
        if class_total[class_name] > 0:
            acc = class_correct[class_name] / class_total[class_name]
            print(f'  {class_name}: {acc:.4f} ({class_correct[class_name]}/{class_total[class_name]})')

    # Salvar modelo
    model_dir = "final_models/ResNet50"
    os.makedirs(model_dir, exist_ok=True)
    model_name = f"ResNet50_SGD_{config['data_type']}"
    model_path = os.path.join(model_dir, f"{model_name}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_accuracy': overall_accuracy,
        'val_accuracy': best_val_acc,
        'classes': train_dataset.classes,
        'config': config,
        'model_type': 'ResNet50'
    }, model_path)

    # Salvar informações
    model_info = {
        'model_name': model_name,
        'test_accuracy': float(overall_accuracy),
        'val_accuracy': float(best_val_acc),
        'config': config,
        'classes': train_dataset.classes,
        'description': 'Treinado com SGD, 20 épocas, lr=1e-3, momentum=0.9, batch_size=32.',
        'timestamp': datetime.now().isoformat()
    }
    with open(os.path.join(model_dir, f"{model_name}_info.json"), 'w') as f:
        json.dump(model_info, f, indent=2)

    print(f"\n=== TREINAMENTO CONCLUÍDO ===")
    print(f"Modelo salvo em: {model_path}")
    print(f"Acurácia de validação: {best_val_acc:.4f}")
    print(f"Acurácia de teste: {overall_accuracy:.4f}")
    clear_cache()

if __name__ == "__main__":
    main()