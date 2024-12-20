import logging
import time

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
import torch
from torch.nn import functional as F
class SegmentationInference:
    
    def __init__(self, model_path):        
        self.model = torch.jit.load(model_path)
        self.MIN_SIZE_TEST = 800
        self.MAX_SIZE_TEST = 1333
        
        self.SCORE_THRESHOLD = 0.3
        self.MASK_THRESHOLD = 0.5
        self.NMS_THRESHOLD = 0.9
        
    def get_set_up(self):
        return {
            'MIN_SIZE_TEST': self.MIN_SIZE_TEST,
            'MAX_SIZE_TEST': self.MAX_SIZE_TEST,
            'SCORE_THRESHOLD': self.SCORE_THRESHOLD,
            'MASK_THRESHOLD': self.MASK_THRESHOLD,
            'NMS_THRESHOLD': self.NMS_THRESHOLD,
        }
    
    def re_init_model(self, threshold):
        try:
            self.SCORE_THRESHOLD = max(0.01, min(0.9, threshold))
        except Exception as e:
            logging.warning('exception', extra={'custom_dimensions': {'error_message': str(e), 'place': 're_init_model'}})

    def inference(self, np_img_src):
        start_time = time.time()
        
        #resize img 
        np_img_resized = resize_img_by_shortest_endge(np_img_src, self.MIN_SIZE_TEST, self.MAX_SIZE_TEST)
        
        #get scales of x&y after scaling
        scales = np.divide(np_img_src.shape[:2], np_img_resized.shape[:2])
        
        img_torch_tensor = torch.as_tensor(np_img_resized.astype("float32").transpose(2, 0, 1))
        
        segm_output = self.model(img_torch_tensor)
        mask_and_poly = self.convert_output_to_mask_and_polygons(segm_output, np_img_resized, scales)
        polygons, masks = self.__process_output(mask_and_poly)
        logging.info("Inference time by Mask RCNN models has taken {} [s]".format(round(time.time() - start_time, 2)))
        
        return polygons, masks
    
    def __process_output(self, output):
        
        def is_valid_polygon(polygon_array):
            try:
                Polygon(polygon_array)
                return True
            except Exception as e:
                logging.info(f"[PROCESSING][SEGMENTATION] polygon is broken for current mask - skip it: {e}")
                return False

        poly_instances = [[Polygon(polygon_array), polygon_array, mask] for mask, polygon_array in output if is_valid_polygon(polygon_array)]
        
        poly_instances = sorted(poly_instances, key=lambda x: x[0].area, reverse=True)
        # Create a list of indices to keep
        keep_indices = [0]

        for i in range(1, len(poly_instances)):
            keep = True
            for j in keep_indices:
                keep = self.filter_by_iou_threshold(poly_instances[i][0], poly_instances[j][0])
            if keep:
                keep_indices.append(i)

        logging.info(f"[PROCESSING][SEGMENTATION] After applying custom NMS method removed N = {(len(poly_instances) - len(keep_indices))} indexes")
        
        polygons = [SegmentationInference.poly_array_to_dict(poly_instances[i][1]) for i in keep_indices]
        masks = [poly_instances[i][2] for i in keep_indices]
        
        return polygons, masks
    
    def convert_output_to_mask_and_polygons(self, mask_rcnn_output, np_img_resized, scales):

        def rescale_polygon_to_src_size(poly, start_pont, scales):
            return [[int((start_pont[0] + point[0]) * scales[0]), 
                     int((start_pont[1] + point[1]) * scales[1])] for point in poly]

        boxes, classes, masks, scores, img_size = mask_rcnn_output
        processed = []

        for ind in range(len(masks)):
            if scores[ind] <= self.SCORE_THRESHOLD: continue
            x1, y1, x2, y2 = int(boxes[ind][0]), int(boxes[ind][1]), int(boxes[ind][2]) , int(boxes[ind][3])  
            mask_h, mask_w = y2 - y1, x2 - x1

            np_mask = do_paste_mask(masks[ind, None, :, :], mask_h, mask_w).numpy()[0][0]

            np_mask = np.where(np_mask > self.MASK_THRESHOLD, 255, 0).astype(np.uint8)
            crop_image = np_img_resized[y1:y1 + mask_h, x1:x1+mask_w]
            res = cv2.bitwise_and(crop_image, crop_image, mask = np_mask)

            contours = bitmap_to_polygon(np_mask)

            if len(contours) < 1 or len(contours[0]) < 10:
                continue

            polygon_full = rescale_polygon_to_src_size(contours[0], (x1, y1), scales)

            processed.append([res, polygon_full])
        return processed
    
    def segmentate_img(self, img_path, out_path, color=(110, 0, 0, 128)):
   
        np_img_src = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) 


        polygons, masks = self.inference(np_img_src)

    
        if np_img_src.shape[2] != 4:
        
            b_channel, g_channel, r_channel = cv2.split(np_img_src)
            alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255 
            np_img_src = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

        # print("Polys:\n", polygons, flush=True)
        # print("Masks:\n", len(masks))
        # print("311 exists", "x311" in str(polygons[1]))
        # print("310 exists", "x310" in str(polygons[1]))
        polys = []
        for polygon in polygons:    
            for j in range(1, int(len(polygon)/2)):
                if f"x{j}" in str(polygon):
                    polys.append([polygon[f'x{j}'], polygon[f'y{j}']])      
        # polys = [[polygons[1][f'x{i}'], polygons[1][f'y{i}']] for i in range(1, int(math.floor(len(polygons[0])/2)))]
        pols_arr = np.array(polys, dtype=np.int32)

        
    
        overlay = np.zeros_like(np_img_src, dtype=np.uint8) 
        cv2.fillPoly(overlay, [pols_arr], color) 

    
        np_img_src = cv2.addWeighted(np_img_src, 1.0, overlay, 1.0, 1.0)

        
        cv2.imwrite(out_path, np_img_src)
    
    @staticmethod
    def poly_array_to_dict(poly):
        polygons_dict = {}
        for i in range(len(poly)):
            polygons_dict.update({
                "x{}".format(i + 1): poly[i][0],
                "y{}".format(i + 1): poly[i][1]
            })
        return polygons_dict
    
    def filter_by_iou_threshold(self, poly_a, poly_b):
        """
        Checks if the two given polygons intersect.

        Args:
            - poly_a: shapely.geometry.Polygon, first polygon
            - poly_b: shapely.geometry.Polygon, second polygon
            - threshold: float, threshold above which the polygons intersect

        Returns:
            - True if the polygons intersect, False otherwise
        """
        intersection_area = poly_a.intersection(poly_b).area
        union_area = poly_a.union(poly_b).area

        iou = intersection_area / union_area
        if iou > self.NMS_THRESHOLD:
            return False
        return True

# Image utils
def resize_img_by_shortest_endge(img_np, MIN_SIZE_TEST, MAX_SIZE_TEST):
    
    src_h, src_w = img_np.shape[:2]
    new_h, new_w = get_output_shape(src_h, src_w, MIN_SIZE_TEST, MAX_SIZE_TEST)
    img_resized_np = apply_image(img_np, new_h, new_w)
    
    return img_resized_np

def get_output_shape(
    oldh: int, oldw: int, short_edge_length: int, max_size: int) -> [int, int]: # type: ignore
    """
    Compute the output size given input size and target short edge length.
    """
    h, w = oldh, oldw
    size = short_edge_length * 1.0
    scale = size / min(h, w)
    if h < w:
        newh, neww = size, scale * w
    else:
        newh, neww = scale * h, size
    if max(newh, neww) > max_size:
        scale = max_size * 1.0 / max(newh, neww)
        newh = newh * scale
        neww = neww * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)

def apply_image(img, new_h, new_w, interp_method=Image.BILINEAR):
    assert len(img.shape) <= 4
    
    if img.dtype == np.uint8:
        if len(img.shape) > 2 and img.shape[2] == 1:
            pil_image = Image.fromarray(img[:, :, 0], mode="L")
        else:
            pil_image = Image.fromarray(img)
        pil_image = pil_image.resize((new_w, new_h), interp_method)
        ret = np.asarray(pil_image)
        if len(img.shape) > 2 and img.shape[2] == 1:
            ret = np.expand_dims(ret, -1)
    else:
        # PIL only supports uint8
        if any(x < 0 for x in img.strides):
            img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)
        shape = list(img.shape)
        shape_4d = shape[:2] + [1] * (4 - len(shape)) + shape[2:]
        img = img.view(shape_4d).permute(2, 3, 0, 1)  # hw(c) -> nchw
        _PIL_RESIZE_TO_INTERPOLATE_MODE = {
            Image.NEAREST: "nearest",
            Image.BILINEAR: "bilinear",
            Image.BICUBIC: "bicubic",
        }
        mode = _PIL_RESIZE_TO_INTERPOLATE_MODE[interp_method]
        align_corners = None if mode == "nearest" else False
        img = F.interpolate(
            img, (new_h, new_w), mode=mode, align_corners=align_corners
        )
        shape[:2] = (new_h, new_w)
        ret = img.permute(2, 3, 0, 1).view(shape).numpy()  # nchw -> hw(c)

    return ret

def do_paste_mask(masks, img_h: int, img_w: int):
    """
    Args:
        masks: N, 1, H, W
        boxes: N, 4
        img_h, img_w (int):
        skip_empty (bool): only paste masks within the region that
            tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.

    Returns:
        if skip_empty == False, a mask of shape (N, img_h, img_w)
        if skip_empty == True, a mask of shape (N, h', w'), and the slice
            object for the corresponding region.
    """
    device = masks.device

    x0_int, y0_int = 0, 0
    x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 =  torch.Tensor([[0]]), torch.Tensor([[0]]), torch.Tensor([[img_w]]), torch.Tensor([[img_h]])

    N = masks.shape[0]

    img_y = torch.arange(y0_int, y1_int, device=device, dtype=torch.float32) + 0.5
    img_x = torch.arange(x0_int, x1_int, device=device, dtype=torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)
    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)

    resized_mask = F.grid_sample(masks, grid.to(masks.dtype), align_corners=False)

    return resized_mask
    
def bitmap_to_polygon(bitmap):
    """Convert masks from the form of bitmaps to polygons.

    Args:
        bitmap (ndarray): masks in bitmap representation.

    Return:
        list[ndarray]: the converted mask in polygon representation.
        bool: whether the mask has holes.
    """
    bitmap = np.ascontiguousarray(bitmap).astype(np.uint8)
    # cv2.RETR_CCOMP: retrieves all of the contours and organizes them
    #   into a two-level hierarchy. At the top level, there are external
    #   boundaries of the components. At the second level, there are
    #   boundaries of the holes. If there is another contour inside a hole
    #   of a connected component, it is still put at the top level.
    # cv2.CHAIN_APPROX_NONE: stores absolutely all the contour points.
    outs = cv2.findContours(bitmap, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = outs[-2]
    hierarchy = outs[-1]
    if hierarchy is None:
        return [], False
    # hierarchy[i]: 4 elements, for the indexes of next, previous,
    # parent, or nested contours. If there is no corresponding contour,
    # it will be -1.
    contours = [c.reshape(-1, 2) for c in contours]
    return sorted(contours, key=len, reverse = True)












    # def inference(self, np_img_src, output_image_path=None):
    #     start_time = time.time()
        
    #     # Resize image
    #     np_img_resized = resize_img_by_shortest_endge(np_img_src, self.MIN_SIZE_TEST, self.MAX_SIZE_TEST)
        
    #     # Get scales of x & y after scaling
    #     scales = np.divide(np_img_src.shape[:2], np_img_resized.shape[:2])
        
    #     img_torch_tensor = torch.as_tensor(np_img_resized.astype("float32").transpose(2, 0, 1))
        
    #     segm_output = self.model(img_torch_tensor)
    #     mask_and_poly = self.convert_output_to_mask_and_polygons(segm_output, np_img_resized, scales)
    #     polygons, masks = self.__process_output(mask_and_poly)
        
    #     # Draw masks on the image
    #     if output_image_path:
    #         output_image = self.draw_masks(np_img_src, masks, polygons)
    #         cv2.imwrite(output_image_path, output_image)
    #         logging.info(f"Output image saved to {output_image_path}")
        
    #     logging.info("Inference time by Mask RCNN models has taken {} [s]".format(round(time.time() - start_time, 2)))
        
    #     return polygons, masks

    # def draw_masks(self, img, masks, polygons):
    #     """
    #     Draw masks and polygons on the image.
    #     Args:
    #         img: Original image (numpy array).
    #         masks: List of binary masks.
    #         polygons: List of polygon coordinates.
    #     Returns:
    #         Annotated image with masks and polygons drawn.
    #     """
    #     annotated_img = img.copy()

    #     for mask, poly in zip(masks, polygons):
    #         # Resize mask to match the original image size
    #         resized_mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            
    #         # Ensure the mask is binary
    #         binary_mask = (resized_mask > 0).astype(np.uint8)
            
    #         # Draw the mask with transparency
    #         mask_colored = np.zeros_like(annotated_img, dtype=np.uint8)
    #         mask_colored[:, :, 0] = binary_mask * 255  # Use green channel for the mask
    #         annotated_img = cv2.addWeighted(annotated_img, 1.0, mask_colored, 0.5, 0)

    #         # Draw the polygon border
    #         polygon_points = np.array([[point["x{}".format(i + 1)], point["y{}".format(i + 1)]] for i in range(len(poly))], dtype=np.int32) # type: ignore
    #         cv2.polylines(annotated_img, [polygon_points], isClosed=True, color=(0, 0, 255), thickness=2)

    #     return annotated_img
