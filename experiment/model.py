import os
import time
import random

import torch
import torch.nn as nn
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

from torchvision.models import (
    resnet18,
    efficientnet_v2_m,
    convnext_base,
    vit_b_32,
    densenet169,
)


def set_seed(seed=42):
    """Фиксирует генераторы случайных чисел для воспроизводимости.

    Требование методических рекомендаций: фиксировать seed и параметры
    разбиения, чтобы эксперимент можно было повторить.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model_size_mb(model):
    """Возвращает размер модели в МБ (сумма параметров, float32)."""
    num_params = sum(p.numel() for p in model.parameters())
    size_mb = num_params * 4 / (1024 ** 2)
    return round(size_mb, 2)


def create_model(model_name, num_classes=2):
        
    if model_name == "resnet18":
        model = resnet18(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "efficientnet_v2_m":
        model = efficientnet_v2_m(weights="DEFAULT")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == "convnext_base":
        model = convnext_base(weights="DEFAULT")
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    elif model_name == "vit_b_32":
        model = vit_b_32(weights="DEFAULT")
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    elif model_name == "densenet169":
        model = densenet169(weights="DEFAULT")
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    else:
        raise ValueError("Unknown model")

    return model


def train_model(
    model, train_loader, val_loader, 
    optimizer, criterion, epochs=10
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    for epoch in range(epochs):
        
        model.train()
        train_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        
        model.eval()
        preds, targets = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = model(x)

                pred = torch.argmax(logits, dim=1)

                preds.extend(pred.cpu().numpy())
                targets.extend(y.numpy())

        f1 = f1_score(targets, preds)

        print(f"Epoch {epoch+1}: loss={train_loss:.3f}, F1={f1:.3f}")

    torch.save(model.state_dict(), f"models/{model.__class__.__name__}.pth")


def evaluate_model(model, test_loader):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    total_time = 0
    use_cuda_events = device == 'cuda'

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)

            if use_cuda_events:
                # Точный замер на GPU.
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                outputs = model(x)
                end.record()

                torch.cuda.synchronize()
                total_time += start.elapsed_time(end)
            else:
                # Fallback для машин без CUDA (раньше здесь падало).
                start = time.perf_counter()
                outputs = model(x)
                total_time += (time.perf_counter() - start) * 1000

            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(y.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    cm = confusion_matrix(all_targets, all_preds)

    avg_time_per_batch = total_time / len(test_loader)

    return {
        "acc": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "cm": cm,
        "time": avg_time_per_batch
    }