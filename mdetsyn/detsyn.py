import random
import os
import argparse
import glob
import uuid
import json
from typing import Union

from PIL import Image

from .helpers import to_pil, adjust_background_size, \
                    random_resize, random_reduce_transparency, random_rotate, random_perspective_transform, \
                    seamless_clone, \
                    get_iob, bboxes_to_yolo_labels

def create_args():
    parser = argparse.ArgumentParser(description='Object Detection Data Synthesis')
    parser_base = parser.add_argument_group('Specify base arguments')
    parser_base.add_argument('--backgrounds', default='./backgrounds', type=str, help='Path to background images directory')
    parser_base.add_argument('--objects', default='./objects', type=str, help='Path to objects images directory')
    parser_base.add_argument('--savename', default='./synthesis', type=str, help='Path to save synthesis images directory')
    parser_base.add_argument('--number', default=1, type=int, help='Number of generate labels for each class')
    parser_base.add_argument('--class_mapping', default=None, type=str, help='Path to class mapping file')
    parser_base.add_argument('--class_txt', default=None, type=str, help='Path to classes.txt file')
    parser_base.add_argument('--min_background_size', default=640, type=int, help='Min image size of result')
    parser_base.add_argument('--min_object_per_image', default=1, type=int, help='Min Number of generate labels for each image')
    parser_base.add_argument('--max_object_per_image', default=5, type=int, help='Max Number of generate labels for each image')

    parser_aug = parser.add_argument_group('Specify augmentation arguments')
    parser_aug.add_argument('--resize_min_ratio', default=0.2, type=float, help='Min object size ratio')
    parser_aug.add_argument('--resize_max_ratio', default=0.4, type=float, help='Max object size ratio')

    parser_aug.add_argument('--rotate_max_degree', default=180, type=float, help='Max rotate degree')
    parser_aug.add_argument('--rotate_prob', default=1, type=float, help='Probability of using rotate')

    parser_aug.add_argument('--transparency_min_ratio', default=0.7, type=float, help='Min transparency size ratio')
    parser_aug.add_argument('--transparency_max_ratio', default=1.0, type=float, help='Max transparency size ratio')
    parser_aug.add_argument('--transparency_prob', default=1, type=float, help='Probability of using transparency')

    parser_aug.add_argument('--perspective_min_value', default=3, type=float, help='Min value of x (Split image to x part, lower mean more perspective)')
    parser_aug.add_argument('--perspective_max_value', default=10, type=float, help='Max value of x (Split image to x part, higher mean less perspective)')
    parser_aug.add_argument('--perspective_prob', default=1, type=float, help='Probability of using perspective')
    
    parser_aug.add_argument('--seamless_clone_prob', default=0.1, type=float, help='Probability of using seamless clone')
    parser_aug.add_argument('--grayscale_prob', default=0.15, type=float, help='Probability of using grayscale image')

    parser_other = parser.add_argument_group('Other arguments')
    parser_other.add_argument('--max_overlap_iob', default=0.2, type=float, help='max ratio of intersaction over box')
    parser_other.add_argument('--max_overlap_retry', default=10, type=int, help='Max creates objects retry time')
    
    args = parser.parse_args()
    return args

def paste_list_object_to_background(list_object_path, background_image, args):
    list_object_image = [Image.open(object_path).convert("RGBA") for object_path in list_object_path]
    
    bboxes = []
    for object_image, object_path in zip(list_object_image, list_object_path):
        class_name = object_path.split("/")[-2]

        object_image_pil = to_pil(object_image)
        background_image_pil = to_pil(background_image)

        background_w, background_h = background_image_pil.size

        min_object_size = int(min(background_h, background_w) * args.resize_min_ratio)
        max_object_size = int(min(background_h, background_w) * args.resize_max_ratio)
        object_image_pil = random_resize(object_image_pil, start_size = min_object_size, stop_size = max_object_size)

        if random.random() < args.perspective_prob:
            object_image_pil = random_perspective_transform(object_image_pil, transform_range=(args.perspective_min_value, args.perspective_max_value))
        if random.random() < args.rotate_prob:
            object_image_pil = random_rotate(object_image_pil, degree = (-args.rotate_max_degree, args.rotate_max_degree))
        if random.random() < args.transparency_prob:
            object_image_pil = random_reduce_transparency(object_image_pil, rate=(args.transparency_min_ratio, args.transparency_max_ratio))
        object_w, object_h = object_image_pil.size

        for _ in range(args.max_overlap_retry):
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
                if iob > args.max_overlap_iob:
                    break

            if iob > args.max_overlap_iob:
                continue

            break
                
        if random.random() < args.seamless_clone_prob:
            x_center = int(x_start + object_image_pil.size[0] / 2.0)
            y_center = int(y_start + object_image_pil.size[1] / 2.0)
            background_image = seamless_clone(background_image, object_image_pil, x_center, y_center)     
        else:
            background_image.paste(object_image_pil, (x_start, y_start), object_image_pil)

        bboxes.append(current_bbox)

        if random.random() < args.grayscale_prob:
            background_image = background_image.convert("L").convert("RGB")

    return background_image, bboxes


def run_synthesis(args: argparse.Namespace):
    """
        backgrounds: path to background image folder (contain background images)
        objects: path to objects image folder (contain each object images in seperate subfolder)
        savename: path to save synthetic data in YOLO format (images, labels)
        number: Number of each labels image to generate
        class_mapping: path to json file contain class mapping
        class_txt: path to txt file contain classnames
    """
    os.makedirs(args.backgrounds, exist_ok=True)
    save_images_dir = os.path.join(args.savename, "images")
    save_labels_dir = os.path.join(args.savename, "labels")
    os.makedirs(save_images_dir, exist_ok=True)
    os.makedirs(save_labels_dir, exist_ok=True)
    
    list_object_path = glob.glob(f"{args.objects}/*/*")
    list_background_path = glob.glob(f"{args.backgrounds}/*")

    list_object_path = list_object_path * args.number
    random.shuffle(list_object_path)

    if args.class_mapping is not None:
        try:
            with open(args.class_mapping, 'r') as f:
                class_mapping_dict = json.load(f)
        except:
            with open(args.class_mapping, 'w') as f:
                f.write(R"{}")
                class_mapping_dict = {}

    all_length = len(list_object_path)
    process_length = 0
    while len(list_object_path) > 0:
        assert args.min_object_per_image <= args.max_object_per_image
        random_number_labels = random.randint(args.min_object_per_image, args.max_object_per_image)
        random_number_labels = random_number_labels if random_number_labels < len(list_object_path) else len(list_object_path)
        list_process_object_path = []
        for _ in range(random_number_labels):
            list_process_object_path.append(list_object_path.pop())
            process_length += 1

        print(f"Processing {process_length}/{all_length}")

        background_path = random.choice(list_background_path)
        background_image_pil = Image.open(background_path).convert("RGB")
        background_image_pil = adjust_background_size(background_image_pil, args.min_background_size)

        background_image, bboxes = paste_list_object_to_background(list_process_object_path, background_image_pil, args)

        # print(bboxes)
        save_name = str(uuid.uuid4())
        save_image_path = os.path.join(save_images_dir, f"{save_name}.jpg")
        save_label_path = os.path.join(save_labels_dir, f"{save_name}.txt")
        save_label_str, class_mapping_dict = bboxes_to_yolo_labels(bboxes, class_mapping_dict)

        background_image.save(save_image_path)
        with open(save_label_path, "w") as f:
            f.write(save_label_str)

    if args.class_mapping is not None:
        with open(args.class_mapping, 'w') as f:
            json.dump(class_mapping_dict, f)

    if args.class_txt is not None:
        with open(args.class_txt, 'w') as f:
            swap_dict = {}
            for key in class_mapping_dict.keys():
                swap_dict[class_mapping_dict[key]] = key

            for i in range(len(swap_dict.keys())):
                f.write(swap_dict[i])
                f.write("\n")
    

if __name__ == "__main__":
    args = create_args()
    run_synthesis(args)
