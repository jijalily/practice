# predictor.py

import time
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_v2_m


class Predictor:

    def __init__(self, weights_path):

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = efficientnet_v2_m()

        self.model.classifier[1] = nn.Linear(
            self.model.classifier[1].in_features,
            2
        )

        self.model.load_state_dict(torch.load(
            weights_path,
            map_location=self.device
        ))

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ])

    @torch.no_grad()
    def predict(self, image_path):

        image, tensor = self.preprocess(image_path)
        tensor = tensor.to(self.device)

        start = time.perf_counter()

        output = self.model(tensor)
        inference_time = (time.perf_counter() - start) * 1000

        probs = torch.softmax(output, dim=1)
        confidence, pred = probs.max(1)

        return {
            "image": image,
            "tensor": tensor,
            "prediction": pred.item(),
            "confidence": confidence.item(),
            "free_prob": float(probs[0][0]),
            "occupied_prob": float(probs[0][1]),
            "time_ms": round(inference_time, 2)
        }

    def preprocess(self, image_path):

        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0)

        return image, tensor