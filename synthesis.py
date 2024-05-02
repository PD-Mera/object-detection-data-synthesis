import random
import os
import argparse
import glob
import uuid
import json

import cv2
from PIL import Image

from helpers import is_image_file, to_pil, to_cv2, adjust_background_size, \
                    random_resize, random_reduce_transparency, random_rotate, random_perspective_transform, \
                    seamless_clone, \
                    get_iob, bboxes_to_yolo_labels

MAX_OVERLAP_IOB = 0.2
MAX_OVERLAP_RETRY = 10

def paste_object_to_background(object_image, 
                               background_image, 
                               object_min_ratio = 0.1,
                               object_max_ratio = 0.6):
    object_image_pil = to_pil(object_image)
    background_image_pil = to_pil(background_image)

    background_w, background_h = background_image_pil.size

    min_object_size = int(min(background_h, background_w) * object_min_ratio)
    max_object_size = int(min(background_h, background_w) * object_max_ratio)
    object_image_pil = random_resize(object_image_pil, start_size = min_object_size, stop_size = max_object_size)
    
    object_w, object_h = object_image_pil.size

    x_start = random.randrange(0, background_w - object_w)
    y_start = random.randrange(0, background_h - object_h)
    bbox = {"x1": x_start, "y1": y_start, "x2": x_start + object_image_pil.size[0], "y2": y_start + object_image_pil.size[1]}
    background_image.paste(object_image_pil, (x_start, y_start), object_image_pil)
    return background_image, bbox

def paste_list_object_to_background(list_object_path, 
                               background_image, 
                               object_min_ratio = 0.2,
                               object_max_ratio = 0.4):
    list_object_image = [Image.open(object_path).convert("RGBA") for object_path in list_object_path]
    
    bboxes = []
    for object_image, object_path in zip(list_object_image, list_object_path):
        class_name = object_path.split("/")[-2]

        object_image_pil = to_pil(object_image)
        background_image_pil = to_pil(background_image)

        background_w, background_h = background_image_pil.size

        min_object_size = int(min(background_h, background_w) * object_min_ratio)
        max_object_size = int(min(background_h, background_w) * object_max_ratio)
        object_image_pil = random_resize(object_image_pil, start_size = min_object_size, stop_size = max_object_size)
        object_image_pil = random_rotate(object_image_pil, degree = (-20, 20))
        object_image_pil = random_reduce_transparency(object_image_pil, rate=(0.7, 1.0))
        object_image_pil = random_perspective_transform(object_image_pil)
        
        object_w, object_h = object_image_pil.size

        for _ in range(MAX_OVERLAP_RETRY):
            x_start = random.randrange(0, background_w - object_w)
            y_start = random.randrange(0, background_h - object_h)

            current_bbox = {"label": class_name, 
                            "x1": x_start, 
                            "y1": y_start, 
                            "x2": x_start + object_image_pil.size[0], 
                            "y2": y_start + object_image_pil.size[1],
                            "bg_w": background_w,
                            "bg_h": background_h,}
            iob = 0
            for bbox in bboxes:
                iob = get_iob(current_bbox, bbox)
                if iob > MAX_OVERLAP_IOB:
                    break

            if iob > MAX_OVERLAP_IOB:
                continue

            break
                
        if random.random() > 0.9:
            x_center = int(x_start + object_image_pil.size[0] / 2.0)
            y_center = int(y_start + object_image_pil.size[1] / 2.0)
            background_image = seamless_clone(background_image, object_image_pil, x_center, y_center)     
        else:
            background_image.paste(object_image_pil, (x_start, y_start), object_image_pil)
        bboxes.append(current_bbox)
    return background_image, bboxes
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Object Detection Data Synthesis')
    parser.add_argument('--backgrounds', default='./backgrounds', type=str, help='Path to background images directory')
    parser.add_argument('--objects', default='./objects', type=str, help='Path to objects images directory')
    parser.add_argument('--savename', default='./synthesis', type=str, help='Path to save synthesis images directory')
    parser.add_argument('--number', default=1, type=int, help='Number of generate labels for each class')
    parser.add_argument('--class_mapping', default='./class_mapping.json', type=str, help='Path to class mapping file')

    args = parser.parse_args()

    save_images_dir = os.path.join(args.savename, "images")
    save_labels_dir = os.path.join(args.savename, "labels")
    os.makedirs(save_images_dir, exist_ok=True)
    os.makedirs(save_labels_dir, exist_ok=True)
    
    list_object_path = glob.glob(f"{args.objects}/*/*")
    list_background_path = glob.glob(f"{args.backgrounds}/*")

    list_object_path = list_object_path * args.number
    random.shuffle(list_object_path)

    try:
        with open(args.class_mapping, 'r') as f:
            class_mapping_dict = json.load(f)
    except:
        with open(args.class_mapping, 'w') as f:
            f.write(R"{}")
            class_mapping_dict = {}

    while len(list_object_path) > 0:
        random_number_labels = random.randint(1, 5)
        random_number_labels = random_number_labels if random_number_labels < len(list_object_path) else len(list_object_path)
        list_process_object_path = []
        for _ in range(random_number_labels):
            list_process_object_path.append(list_object_path.pop())


        background_path = random.choice(list_background_path)
        background_image_pil = Image.open(background_path)

        background_image, bboxes = paste_list_object_to_background(list_process_object_path, background_image_pil)

        print(bboxes)
        save_name = str(uuid.uuid4())
        save_image_path = os.path.join(save_images_dir, f"{save_name}.jpg")
        save_label_path = os.path.join(save_labels_dir, f"{save_name}.txt")
        save_label_str, class_mapping_dict = bboxes_to_yolo_labels(bboxes, class_mapping_dict)

        background_image.save(save_image_path)
        with open(save_label_path, "w") as f:
            f.write(save_label_str)

    with open(args.class_mapping, 'w') as f:
        json.dump(class_mapping_dict, f)




