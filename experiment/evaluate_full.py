import json
from pathlib import Path

import cv2
import torch
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from model import (
    create_model,
    evaluate_model,
    get_model_size_mb,
    set_seed
)

SEED = 42
set_seed(SEED)

CLASS_NAMES = ["free", "occupied"]

# Краткие выводы о сильных и слабых сторонах каждой архитектуры.
COMMENTS = {
    "resnet18": (
        "Надёжный лёгкий baseline: высокая скорость и малый размер. "
        "Слабая сторона — более простая архитектура, уступает современным "
        "моделям по качеству."
    ),
    "efficientnet_v2_m": (
        "Лучший баланс качество/размер/скорость — самая высокая точность. "
        "Слабая сторона — заметно тяжелее и медленнее ResNet18."
    ),
    "convnext_base": (
        "Современная CNN трансформерного типа, высокое качество. "
        "Слабая сторона — самая медленная и тяжёлая модель в сравнении."
    ),
    "vit_b_32": (
        "Vision Transformer: на малых данных при коротком дообучении "
        "уступает CNN — самая низкая точность и медленный инференс."
    ),
    "densenet169": (
        "Хорошее переисользование признаков, высокое качество. "
        "Слабая сторона — большой размер и потребление памяти."
    ),
}


class ParkingDataset(Dataset):
    def __init__(self, image_dir, file_names, labels, transform=None):
        self.image_dir = Path(image_dir)
        self.file_names = file_names
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        image_path = self.image_dir / self.file_names[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = int(self.labels[idx])
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, torch.tensor(label, dtype=torch.long)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_transform = A.Compose([
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2()
    ])

    with open("dataset/annotations.json") as f:
        ann = json.load(f)

    test_files = ann["test"]["file_names"]
    test_labels = ann["test"]["occupancy_list"]

    test_dataset = ParkingDataset(
        image_dir="dataset/images",
        file_names=test_files,
        labels=test_labels,
        transform=test_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model_names = [
        "resnet18",
        "efficientnet_v2_m",
        "convnext_base",
        "vit_b_32",
        "densenet169",
    ]

    results = {}
    for name in model_names:
        print(f"\n===== Evaluating {name} =====")
        model = create_model(name)
        model.load_state_dict(
            torch.load(f"models/{model.__class__.__name__}.pth")
        )

        metrics = evaluate_model(model, test_loader)
        metrics["size_mb"] = get_model_size_mb(model)
        results[name] = metrics

    # Полная таблица сравнения (соответствует шаблону методички).
    df = pd.DataFrame({
        name: {
            "accuracy": results[name]["acc"],
            "f1": results[name]["f1"],
            "precision": results[name]["precision"],
            "recall": results[name]["recall"],
            "time_ms": results[name]["time"],
            "size_mb": results[name]["size_mb"],
            "comment": COMMENTS[name]
        }
        for name in model_names
    }).T

    df.index.name = "model"
    df.to_csv("models_comparison.csv", encoding="utf-8-sig")
    print("\n=== Таблица сравнения ===")
    print(df[["accuracy", "f1", "precision", "recall", "time_ms", "size_mb"]])

    # Выбор лучшей модели по F1.
    best_name = df["f1"].idxmax()
    print(f"\nЛучшая модель по F1: {best_name}")

    # ---- Анализ ошибок лучшей модели ----
    best_model = create_model(best_name)
    best_model.load_state_dict(
        torch.load(f"models/{best_model.__class__.__name__}.pth")
    )
    best_model.to(device)
    best_model.eval()

    predictions = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            logits = best_model(x)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            for i in range(len(pred)):
                predictions.append((pred[i].item(), conf[i].item()))

    correct = []
    incorrect = []
    for i, (pred, conf) in enumerate(predictions):
        true_label = int(test_labels[i])
        record = (test_files[i], true_label, pred, conf)
        if pred == true_label:
            correct.append(record)
        else:
            incorrect.append(record)

    # Уверенные правильные и ошибочные примеры.
    correct.sort(key=lambda t: t[3], reverse=True)
    incorrect.sort(key=lambda t: t[3], reverse=True)

    out_dir = Path("error_analysis")
    (out_dir / "correct").mkdir(parents=True, exist_ok=True)
    (out_dir / "incorrect").mkdir(parents=True, exist_ok=True)

    def save_example(record, folder, n):
        fname, true_label, pred, conf = record
        img = cv2.imread(f"dataset/images/{fname}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(
            f"Истинный класс: {CLASS_NAMES[true_label]}\n"
            f"Прогноз: {CLASS_NAMES[pred]}\n"
            f"Уверенность: {conf:.3f}"
        )
        plt.tight_layout()
        plt.savefig(
            out_dir / folder / f"{n:02d}_{Path(fname).stem}.png",
            dpi=100
        )
        plt.close()

    for i, rec in enumerate(correct[:3], start=1):
        save_example(rec, "correct", i)

    for i, rec in enumerate(incorrect[:3], start=1):
        save_example(rec, "incorrect", i)

    print(
        f"\nАнализ ошибок сохранён в папку 'error_analysis' "
        f"(по 3 удачных и 3 ошибочных примера лучшей модели)."
    )


if __name__ == "__main__":
    main()
