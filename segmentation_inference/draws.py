

import random
import colorsys

import numpy as np
import matplotlib.pyplot as plt

from skimage.measure import find_contours
from matplotlib.patches import Polygon
from matplotlib import patches


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        x1, y1, width, height = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), width, height, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        plt.show()
        
        
# import cv2
# import numpy as np

# # Suponiendo que tienes la clase SegmentationInference definida 
# # en un archivo segmentation_inference.py (por ejemplo) 
# from inference import SegmentationInference

# # Ruta al modelo TorchScript
# model_path = "models/segmentation_21_08_2023.ts" 

# # Crear una instancia de la clase con el modelo cargado
# seg_inf = SegmentationInference(model_path)

# # Cargar una imagen (por ejemplo con OpenCV)
# img_path = "images/bass.jpg"
# np_img_src = cv2.imread(img_path)

# # Asegúrate de que la imagen esté en el formato que el modelo espera. 
# # Por lo general, OpenCV carga en BGR. Muchos modelos funcionan con RGB, 
# # aunque este código parece usar OpenCV internamente, por lo que BGR debería ser correcto.
# # Si fuera necesario convertir a RGB:
# # np_img_src = cv2.cvtColor(np_img_src, cv2.COLOR_BGR2RGB)

# # Ejecutar la inferencia
# polygons, masks = seg_inf.inference(np_img_src)


# # 'polygons' contendrá la lista de polígonos detectados en el formato dict ("x1", "y1", "x2", "y2", etc.)
# # 'masks' serán las máscaras resultantes.
# # cv2.fillPoly()
# polys = [[polygons[0][f'x{i}'], polygons[0][f'y{i}']] for i in range(1, int(len(polygons[0])/2))]

# pols_arr = np.array(polys)
# print("Polígonos detectados:", polygons)
# print("Cantidad de máscaras:", len(masks))
# print("\n\nPolys: \n", pols_arr)

# cv2.fillPoly(np_img_src, [pols_arr], 255)

# cv2.imwrite("img1.jpg", np_img_src)