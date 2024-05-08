import random

import numpy as np
import cv2
from PIL import Image

def is_image_file(file_path):
    return any(file_path.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def to_pil(image):
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    elif isinstance(image, Image.Image):
        return image
    else:
        raise NotImplementedError

def to_cv2(image):
    if isinstance(image, np.ndarray):
        return image
    elif isinstance(image, Image.Image):
        return np.array(image)
    else:
        raise NotImplementedError

def adjust_background_size(image, min_size):
    """
        Return resized image with min(h, w) = min_size
        Return original image if min(h, w) > min_size
    """
    image_cv2 = to_cv2(image)
    
    h, w, _ = image_cv2.shape 
    short_edge = min(h, w) 
    if short_edge >= min_size: 
        return to_pil(image_cv2)
    
    new_h = int(h / short_edge * min_size)
    new_w = int(w / short_edge * min_size)

    resized_image = cv2.resize(image_cv2, (new_w, new_h))
    resized_image = to_pil(resized_image)

    return resized_image

def random_resize(image, start_size = 480, stop_size = 640):
    assert start_size <= stop_size

    image_cv2 = to_cv2(image)

    h, w, c = image_cv2.shape
    long_edge = max(h, w)
    min_size = random.randrange(start_size, stop_size)
    new_h = int(h / long_edge * min_size)
    new_w = int(w / long_edge * min_size)
    resized_image = cv2.resize(image_cv2, (new_w, new_h))
    resized_image = to_pil(resized_image)
    return resized_image

def random_reduce_transparency(image, rate = (0.5, 1.0)):
    image_cv2 = to_cv2(image)

    rate_ = random.uniform(rate[0], rate[1])

    a_channel = image_cv2[:, :, 3]
    a_channel = np.where(a_channel > 0, a_channel * rate_, 0)
    image_cv2[:, :, 3] = a_channel

    image_pil = to_pil(image_cv2)

    return image_pil

def random_rotate(image, degree = (-45, 45)):
    image_pil = to_pil(image)

    angle = random.randint(degree[0], degree[1])
    rotated_image = image.rotate(angle, expand=True)

    image_pil = to_pil(rotated_image)

    return image_pil

def random_perspective_transform(image, transform_range=(3, 10)):
    image = to_cv2(image)
    
    h, w, _ = image.shape
    split_number = random.randint(transform_range[0], transform_range[1])

    new_x1 = random.randint(0, int(w / split_number))
    new_y1 = random.randint(0, int(h / split_number))

    new_x2 = random.randint(int(w / split_number) * (split_number - 1), w)
    new_y2 = random.randint(0, int(h / split_number))

    new_x3 = random.randint(0, int(w / split_number))
    new_y3 = random.randint(int(h / split_number) * (split_number - 1), h)

    new_x4 = random.randint(int(w / split_number) * (split_number - 1), w)
    new_y4 = random.randint(int(h / split_number) * (split_number - 1), h)

    pts1 = np.float32([[0, 0], [w, 0],
                       [0, h], [w, h]])
    
    pts2 = np.float32([[new_x1, new_y1], [new_x2, new_y2],
                       [new_x3, new_y3], [new_x4, new_y4]])
    
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped_image = cv2.warpPerspective(image, matrix, (w, h))
    new_x_min = min(new_x1, new_x2, new_x3, new_x4)
    new_x_max = max(new_x1, new_x2, new_x3, new_x4)
    new_y_min = min(new_y1, new_y2, new_y3, new_y4)
    new_y_max = max(new_y1, new_y2, new_y3, new_y4)
    warped_image = warped_image[new_y_min:new_y_max, new_x_min:new_x_max, :]
    warped_image = to_pil(warped_image)
    return warped_image

def seamless_clone(background, content, x_center, y_center):
    background_cv2 = to_cv2(background)
    content_cv2 = to_cv2(content)
    content = content_cv2[:, :, 0:3]
    mask = content_cv2[:, :, 3]
    image_cv2 = cv2.seamlessClone(content, background_cv2, mask, (x_center, y_center), cv2.NORMAL_CLONE)
    image_pil = to_pil(image_cv2)
    return image_pil

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def get_iob(bb1, bb2):
    """
    Calculate the Intersection over Boundingbox (IoB) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iob1 = intersection_area / float(bb1_area)
    iob2 = intersection_area / float(bb2_area)
    iob = max(iob1, iob2)
    assert iob >= 0.0
    assert iob <= 1.0
    return iob

def bboxes_to_yolo_labels(bboxes, class_mapping_dict):
    save_str = ""
    
    for bbox in bboxes:
        bbox["label"] = bbox["label"].lower()
        if bbox["label"].lower() not in class_mapping_dict.keys():
            new_idx = len(class_mapping_dict.keys())
            class_mapping_dict[bbox["label"]] = new_idx
        x_center = (bbox["x1"] + bbox["x2"]) / 2.0
        y_center = (bbox["y1"] + bbox["y2"]) / 2.0
        w = abs(bbox["x2"] - bbox["x1"])
        h = abs(bbox["y2"] - bbox["y1"])
        save_str += f'{class_mapping_dict[bbox["label"]]} {x_center / bbox["bg_w"]} {y_center / bbox["bg_h"]} {w / bbox["bg_w"]} {h / bbox["bg_h"]}\n'
    
    return save_str, class_mapping_dict