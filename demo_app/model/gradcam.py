import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


class GradCAMGenerator:

    def __init__(self, model, target_layer, device="cpu"):

        self.model = model
        self.device = device

        self.cam = GradCAM(
            model=model,
            target_layers=[target_layer]
        )

    def generate(self, image_tensor, original_image, class_idx):

        targets = [ClassifierOutputTarget(class_idx)]

        grayscale_cam = self.cam(
            input_tensor=image_tensor,
            targets=targets
        )[0]

        img = np.array(original_image).astype(np.float32) / 255.0
        img = cv2.resize(img, (grayscale_cam.shape[1], grayscale_cam.shape[0]))

        visualization = show_cam_on_image(
            img,
            grayscale_cam,
            use_rgb=True
        )

        return visualization