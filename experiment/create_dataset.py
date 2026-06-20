import json
import cv2
import numpy as np
import os

def warp_roi(image, roi, width=224, height=224, scale=1.15):

    img_h, img_w = image.shape[0], image.shape[1]
    
    # Преобразование координат в абсолютные значения
    src = np.array([
        [x * img_w, y * img_h]
        for x, y in roi
    ], dtype=np.float32)

    # Масштабирование относительно центра
    center = np.mean(src, axis=0)  
    src_scaled = center + (src - center) * scale

    # Преобразование изображения в соотвествии с поданным размером
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(src_scaled, dst)

    warped = cv2.warpPerspective(
        image,
        matrix,
        (width, height)
    )

    return warped


os.makedirs('dataset', exist_ok=True)
os.makedirs('dataset/images', exist_ok=True)

with open('ACPDS/annotations.json', 'r') as f:
    all_annotations = json.load(f)

new_annotations = {}

for ds_type in all_annotations.keys():
    new_annotations[ds_type] = {
        'file_names': [], 
        'occupancy_list': []
    }
    for i in range(len(all_annotations[ds_type]['file_names'])):
        filename = all_annotations[ds_type]['file_names'][i]
        image = cv2.imread(f"ACPDS/images/{filename}")

        for j in range(min(20, len(all_annotations[ds_type]['rois_list'][i]))):
            roi = all_annotations[ds_type]['rois_list'][i][j]
            occupancy = all_annotations[ds_type]['occupancy_list'][i][j]

            crop = warp_roi(
                image,
                roi,
                width=224,
                height=224,
                scale=1.15
            )

            cv2.imwrite(f"dataset/images/{j}_{filename}", crop)
            new_annotations[ds_type]['file_names'].append(f'{j}_{filename}')
            new_annotations[ds_type]['occupancy_list'].append(occupancy)
    
with open('dataset/annotations.json', 'w') as f:
    json.dump(new_annotations, f)