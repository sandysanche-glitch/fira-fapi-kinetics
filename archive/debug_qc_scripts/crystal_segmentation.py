# Author: Christophe Gisler (iCoSys, HEIA-FR)
# Creation year: 2024

import os, sys, json, glob
import copy
import numpy as np
import cv2
import torch
import torchvision

from datetime import date, datetime
from pathlib import Path
from tqdm import tqdm
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from src.crystal_annotations import (CRYSTAL_SUPERCATEGORY, CRYSTAL_ANNOTATION_TO_REMOVE,
                                     CRYSTAL_CELL_ANNOTATION, CRYSTAL_NUCLEUS_ANNOTATION, CRYSTAL_DEFECT_ANNOTATION)
import src.utils.coco as coco


# Define experiment constants and paths, used to pre-categorize the COCO RLE annotations as "Crystal Nucleus" and "Crystal Cell"
DATA_PATH = Path("../") / "data"
SAM_MODEL_PATH = DATA_PATH / "sam_models" / "sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"  # or "vit_l" or "vit_b"
CRYSTAL_NUCLEUS_COLOR_HSV_MAX_VALUE = 0.43
CRYSTAL_NUCLEUS_MAX_AREA = 50 * 50    # Masks can not be a nucleus if their area is bigger than this value
ANNOTATION_MIN_AREA = 10 * 10
ANNOTATION_MAX_AREA = 1941 * 1024 / 3  # = image width * image height / 3
IOU_THRESHOLD = 0.5


def filter_and_label_annotations_as_nuclei_and_cells(
        image,
        annotations,
        annotation_min_area:int = 0,
        annotation_max_area: int = sys.maxsize,
        crystal_nucleus_max_area: int = sys.maxsize,
        crystal_nucleus_color_hsv_max_value=0.43,
        iou_threshold: float = 0.5,
        decode_coco_rle_masks: bool = False
    ):
    """
    Simple method for filtering and pre-categorizeing input COCO RLE annotations as "Crystal Nucleus" and "Crystal Cell"
    """
    # Add field 'area' if missing
    for annotation in annotations:
        if 'area' not in annotation and 'segmentation' in annotation:
            annotation['area'] = coco.get_area_from_coco_rle(annotation['segmentation'])
    # Sort COCO annotations (segments) by area from the smallest to the biggest
    annotations_to_keep = sorted(annotations, key=lambda x: x['area'], reverse=False)
    # min_area, max_area = sys.maxsize, 0  # For stats only
    for i in range(len(annotations_to_keep)):
        annotation_i = annotations_to_keep[i]
        annotation_i_area = annotation_i['area']

        # For stats only
        # if annotation_i_area > max_area:
        #     max_area = annotation_i_area
        # elif annotation_i_area < min_area:
        #     min_area = annotation_i_area

        # Ignore annotations too small or too big
        if annotation_i_area < annotation_min_area or annotation_i_area > annotation_max_area:
            # print("Segment too small or too big found and ignored")
            annotation_i['category_id'] = CRYSTAL_ANNOTATION_TO_REMOVE.ID
            continue

        # If annotation area is greater than nucleus maximum area, then label the annotation as a crystal cell, else label it as a crystal nucleus
        if annotation_i_area > crystal_nucleus_max_area:
            annotation_i['category_id'] = CRYSTAL_CELL_ANNOTATION.ID
        else:
            annotation_i['category_id'] = CRYSTAL_NUCLEUS_ANNOTATION.ID

        annotation_i['has_nucleus'] = False  # By default

        for j in range(i):
            annotation_j = annotations_to_keep[j]

            # If annotation j was set to be removed, then continue. Otherwise is a either a crystal cell (1) or a crystal nucleus (2)
            if annotation_j['category_id'] == CRYSTAL_ANNOTATION_TO_REMOVE.ID:
                continue

            # Remove annotation j if IoU between annotation i and annotation j is too high, i.e. both annotations are considered to be the same
            # iou = compute_iou_between_masks(annotation_i, annotation_j, decode_coco_rle_masks)  # Very runtime consuming!
            iou = coco.compute_iou_between_bboxes(annotation_i, annotation_j)  # Less runtime consuming!
            if iou > iou_threshold:
                # print("IoU:", iou)
                # We keep the biggest mask, i.e. annotation i
                annotation_i['category_id'] = annotation_j['category_id']
                annotation_i['has_nucleus'] = annotation_j['has_nucleus']
                annotation_j['category_id'] = CRYSTAL_ANNOTATION_TO_REMOVE.ID
                continue

            # If annotation j is inside annotation i,...
            if coco.is_bbox1_inside_bbox2(annotation_j['bbox'], annotation_i['bbox']):
                # If annotation j (which is smaller than annotation i) is a crystal cell, then annotation j must be removed
                if annotation_j['category_id'] == CRYSTAL_CELL_ANNOTATION.ID:
                    annotation_i['category_id'] = annotation_j['category_id']
                    annotation_i['has_nucleus'] = annotation_j['has_nucleus']
                    annotation_j['category_id'] = CRYSTAL_ANNOTATION_TO_REMOVE.ID
                # if annotation j (which is smaller) is a crystal nucleus, then it is a nucleus of annotation i which therefore is a cell for sure
                else:  # if annotation_j['category_id'] == 2:  # <-- this is the case, because != 0 (see test above)
                    # If annotation i is also a crystal nucleus, then it cannot contain another annotation, so annotation j must be a cell
                    if annotation_i['category_id'] == CRYSTAL_NUCLEUS_ANNOTATION.ID:
                        annotation_i['category_id'] = CRYSTAL_CELL_ANNOTATION.ID
                    annotation_i['has_nucleus'] = True

        # For stats only
        # crystal_nucleus_area_mean += annotation_i['area']
        # crystal_nucleus_nbr += 1

    # print("Min. annotation area:", min_area, "| Max. annotation area:", max_area)  # For stats only
    # print("crystal_nucleus_area_mean:", crystal_nucleus_area_mean/crystal_nucleus_nbr)  # For stats only

    for annotation in annotations_to_keep:
        # Set all crystal cells which are small enough to be nuclei  and which do not have a nucleus as crystal nuclei
        if annotation['category_id'] == CRYSTAL_CELL_ANNOTATION.ID and not annotation['has_nucleus'] and annotation['area'] <= crystal_nucleus_max_area:
            annotation['category_id'] = CRYSTAL_NUCLEUS_ANNOTATION.ID
        # If HSV value of average color in image annotation is greater than given threshold (i.e. clearer) in a nucleus, set it as a cell (runtime consuming!)
        if annotation['category_id'] == CRYSTAL_NUCLEUS_ANNOTATION.ID and coco.get_average_hsv_color_in_mask(image, annotation, decode_coco_rle_masks)[2] > crystal_nucleus_color_hsv_max_value:
            annotation['category_id'] = CRYSTAL_CELL_ANNOTATION.ID

    annotations_to_keep = [annotation for annotation in annotations_to_keep if annotation['category_id'] != CRYSTAL_ANNOTATION_TO_REMOVE.ID]

    annotations_to_keep.reverse()
    return annotations_to_keep


def create_coco_datasets_from_label_studio_annotations(annotations_path, fapi_output_path, fapi_tempo_output_path, json_indent=3):
    # Load the JSON data
    with open(annotations_path, 'r') as file:
        data = json.load(file)

    # Base structure for the new JSON files
    def create_base_structure(info):
        return {
            "images": [],
            "categories": [
                {"id": CRYSTAL_CELL_ANNOTATION.ID, "name": CRYSTAL_CELL_ANNOTATION.label, "supercategory": CRYSTAL_SUPERCATEGORY},
                {"id": CRYSTAL_NUCLEUS_ANNOTATION.ID, "name": CRYSTAL_NUCLEUS_ANNOTATION.label, "supercategory": CRYSTAL_SUPERCATEGORY},
                {"id": CRYSTAL_DEFECT_ANNOTATION.ID, "name": CRYSTAL_DEFECT_ANNOTATION.label, "supercategory": CRYSTAL_SUPERCATEGORY}
            ],
            "annotations": [],
            "info": info
        }

    # Initialize the two datasets
    fapi_data = create_base_structure(data["info"])
    fapi_tempo_data = create_base_structure(data["info"])

    # Process each image, modify the file name, and distribute into FAPI or FAPI_TEMPO
    def process_and_distribute_images(image, fapi_data, fapi_tempo_data):
        image_copy = copy.deepcopy(image)
        image_copy["file_name"] = Path(image_copy["file_name"]).name

        if "FAPI_TEMPO" in image_copy["file_name"]:
            fapi_tempo_data["images"].append(image_copy)
        else:
            fapi_data["images"].append(image_copy)

    # Apply the process to all images
    for image in data["images"]:
        process_and_distribute_images(image, fapi_data, fapi_tempo_data)

    # Distribute annotations based on the image_id and update category_id
    def distribute_annotations(annotation, fapi_data, fapi_tempo_data):
        updated_annotation = copy.deepcopy(annotation)
        updated_annotation['category_id'] += 1

        target_data = fapi_tempo_data if any(img["id"] == updated_annotation["image_id"] for img in fapi_tempo_data["images"]) else fapi_data
        target_data["annotations"].append(updated_annotation)

    # Apply the distribution to all annotations
    for annotation in data["annotations"]:
        distribute_annotations(annotation, fapi_data, fapi_tempo_data)

    # Save the resulting datasets into separate files
    coco.save_coco_dataset_to_coco_dataset_file(fapi_data, fapi_output_path, json_indent)
    coco.save_coco_dataset_to_coco_dataset_file(fapi_tempo_data, fapi_tempo_output_path, json_indent)


def create_coco_dataset_file_from_coco_rle_files(
    coco_rle_annotations_folder_path: Path,
    coco_dataset_file_path: Path,
    coco_dataset_description: str,
    coco_dataset_url:str,
    coco_dataset_version: str,
    coco_dataset_year: int,
    coco_dataset_contributor: str,
    coco_dataset_license_name: str,
    coco_dataset_license_url: str,
    image_folder_path: str,
    remove_annotations_touching_borders: bool = False,
    filter_and_precategorize_annotations: bool = False,
    default_annotation_category_id: int = CRYSTAL_CELL_ANNOTATION.ID,
    flatten_annotations_by_category: bool = False,
    coco_rle_annotation_score_field: str = 'score',
    coco_image_names_and_ids: dict[str, int] = None,
    json_indent: int = 0
    ):
    """
    Create a COCO dataset JSON file from the given COCO RLE files.
    1. If remove_annotations_touching_borders is True => Remove the annotations whose mask touches a border of the image;
    2. If filter_and_precategorize_annotations, a simple and basic function will be called to filter
       and pre-categorize annotations either as crystal cell or nucleus;
    3. If default_annotation_category_id is set to 1 or 2 and filter_and_precategorize_annotations set to False,
       the category ID of a crystal cell or nucleus will be given by default to all annotations;
    4. If flatten_annotations_by_category is True, flatten masks in annotations by merging overlapping masks of the same category;
    """
    coco_dataset = dict()
    datetime_format = '%Y-%m-%d'
    # Create dataset information part
    coco_dataset['info'] = {
        'description': coco_dataset_description,
        'url': coco_dataset_url,
        'version': coco_dataset_version,
        'year': coco_dataset_year,
        'contributor': coco_dataset_contributor,
        'date_created': date.today().strftime(datetime_format)
    }
    dataset_license_id = 1
    coco_dataset['licenses'] = [
        {'url': coco_dataset_license_url, 'id': dataset_license_id, 'name': coco_dataset_license_name}
    ]
    coco_dataset['images'] = []
    coco_dataset['categories'] = [
        {'id': CRYSTAL_CELL_ANNOTATION.ID, 'name': CRYSTAL_CELL_ANNOTATION.label, 'supercategory': CRYSTAL_SUPERCATEGORY},
        {'id': CRYSTAL_NUCLEUS_ANNOTATION.ID, 'name': CRYSTAL_NUCLEUS_ANNOTATION.label, 'supercategory': CRYSTAL_SUPERCATEGORY},
        {'id': CRYSTAL_DEFECT_ANNOTATION.ID, 'name': CRYSTAL_DEFECT_ANNOTATION.label, 'supercategory': CRYSTAL_SUPERCATEGORY}
    ]
    coco_dataset['annotations'] = []
    # Retrieve COCO RLE annotation JSON file paths
    json_pattern = os.path.join(coco_rle_annotations_folder_path, '*.json')
    annotations_file_paths = [Path(fp) for fp in glob.glob(json_pattern)]
    image_id = 0
    annotation_id = 1
    annotations_file_progress_bar = tqdm(annotations_file_paths)
    for annotations_file_path in annotations_file_progress_bar:
        annotations_file_progress_bar.set_description(f"Parsing COCO RLE file: {annotations_file_path.as_posix()}")
        # Open the JSON file
        with open(annotations_file_path, 'r') as annotations_file:
            # Load JSON data from file
            annotations = json.load(annotations_file)
            # Load the image (required to filter annotations)
            image_name = f"{annotations_file_path.stem}.jpg"
            if coco_image_names_and_ids is None:
                image_id += 1
            else:
                image_id = coco_image_names_and_ids[image_name]
            image_path = image_folder_path / image_name
            image = cv2.imread(image_path.as_posix())
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Create image information part
            image_size = annotations[0]['segmentation']['size']
            image_info = {
                'id': image_id,
                'file_name': image_name,
                'coco_url': annotations_file_path.name,
                'width': image_size[1],
                'height': image_size[0],
                'license': dataset_license_id,  # ID of the image license from the "licenses" section
                'date_captured': datetime.fromtimestamp(annotations_file_path.stat().st_ctime).strftime(datetime_format)  # When the photo was created
            }
            coco_dataset['images'].append(image_info)
            # If wanted, remove the annotations whose mask touches a border of the image
            if remove_annotations_touching_borders:
                annotations = coco.remove_masks_touching_borders_in_annotations(annotations, decode_coco_rle_masks = True)
            # If wanted, filter and pre-categorize annotations (crystal cell/nucleus pre-detection/labeling)
            if filter_and_precategorize_annotations:
                annotations = filter_and_label_annotations_as_nuclei_and_cells(
                    image = image,
                    annotations = annotations,
                    annotation_min_area = ANNOTATION_MIN_AREA,
                    annotation_max_area = ANNOTATION_MAX_AREA,  # Or: original_width * original_height / 3
                    crystal_nucleus_max_area = CRYSTAL_NUCLEUS_MAX_AREA,
                    crystal_nucleus_color_hsv_max_value = CRYSTAL_NUCLEUS_COLOR_HSV_MAX_VALUE,
                    iou_threshold = IOU_THRESHOLD,
                    decode_coco_rle_masks = True
                )
            # If needed, flatten annotations by category, i.e. deal with overlapping masks in each category (class)
            if flatten_annotations_by_category:
                annotations = coco.flatten_masks_in_annotations_by_category(annotations, decode_coco_rle_masks = True)
            for annotation in annotations:
                # annotation = rle_to_coco(annotation)  # No need to uncompress compressed COCO RLE mask
                annotation['id'] = annotation_id
                annotation['image_id'] = image_id
                # If category_id has not already been set in filter_and_label_annotations_as_nuclei_and_cells(...)
                if not filter_and_precategorize_annotations:
                    annotation['category_id'] = CRYSTAL_NUCLEUS_ANNOTATION.ID if default_annotation_category_id == CRYSTAL_NUCLEUS_ANNOTATION.ID else CRYSTAL_CELL_ANNOTATION.ID
                if coco_rle_annotation_score_field != 'score':
                    annotation['score'] = annotation[coco_rle_annotation_score_field]
                annotation['ignore'] = 0
                annotation['iscrowd'] = 0
                coco_dataset['annotations'].append(annotation)
                annotation_id += 1
    # Save the whole dataset as JSON file
    with open(coco_dataset_file_path, "w") as dataset_file:
        if json_indent > 0:
            json.dump(coco_dataset, dataset_file, indent=json_indent, sort_keys = False, separators = (',', ':'))  # Readable but heavy
        else:
            json.dump(coco_dataset, dataset_file)  # Compact and light
    return coco_dataset


# TODO To be replaced by create_label_studio_preannotations_from_coco_dataset(coco_dataset_file_path, label_studio_preannotations_folder) in coco.py
def create_label_studio_preannotations_from_coco_rle_files(
        coco_rle_annotations_folder_path: Path,
        label_studio_preannotations_folder_path: Path,
        image_folder_path: str,
        image_s3_folder_path: str,
        model_version: str,
        remove_annotations_touching_borders: bool = False,
        filter_and_precategorize_annotations: bool = False,
        default_annotation_category_id: int = CRYSTAL_CELL_ANNOTATION.ID,
        flatten_annotations_by_category: bool = False,
        coco_image_names_and_ids: dict[str, int] = None
    ):
    """
    Parse the given input COCO RLE annotations and create a Label Studio preannotation task file for each COCO RLE annotation file).
    1. If remove_annotations_touching_borders is True => Remove the annotations whose mask touches a border of the image;
    2. If filter_and_precategorize_annotations, a simple and basic function will be called to filter
       and pre-categorize annotations either as crystal cell or nucleus;
    3. If default_annotation_category_id is set to 1 or 2 and filter_and_precategorize_annotations set to False,
       the category ID of a crystal cell or nucleus will be given by default to all annotations;
    4. If flatten_annotations_by_category is True, flatten masks in annotations by merging overlapping masks of the same category;
    """
    # datetime_format = '%Y-%m-%d'
    #'date_created': date.today().strftime(datetime_format)
    # Create the Label Studio preannotation folder if necessary
    label_studio_preannotations_folder_path.mkdir(parents=True, exist_ok=True)
    json_pattern = os.path.join(coco_rle_annotations_folder_path, '*.json')
    # Retrieve COCO RLE annotation JSON file paths
    annotations_file_paths = [Path(fp) for fp in glob.glob(json_pattern)]
    image_id = 0
    annotation_id = 1
    annotations_file_progress_bar = tqdm(annotations_file_paths)
    for annotations_file_path in annotations_file_progress_bar:
        annotations_file_progress_bar.set_description(f"Parsing COCO RLE file: {annotations_file_path.as_posix()}")
        # Open the JSON file
        with open(annotations_file_path, 'r') as annotations_file:
            # To compute the mean crystal nucleus area
            # crystal_nucleus_area_mean = 0
            # crystal_nucleus_nbr = 0
            # Load JSON data from file
            annotations = json.load(annotations_file)
            # Load the image (required to filter annotations)
            image_name = f"{annotations_file_path.stem}.jpg"
            if coco_image_names_and_ids is None:
                image_id += 1
            else:
                image_id = coco_image_names_and_ids[image_name]
            image_path = image_folder_path / image_name
            image = cv2.imread(image_path.as_posix())
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Get image information
            image_size = annotations[0]['segmentation']['size']
            original_width = image_size[1]
            original_height = image_size[0]
            # If needed, remove the annotations whose mask touches a border of the image
            if remove_annotations_touching_borders:
                annotations = coco.remove_masks_touching_borders_in_annotations(annotations, decode_coco_rle_masks = True)
            # If needed, filter and pre-categorize annotations as crystal cell or nucleus
            if filter_and_precategorize_annotations:
                annotations = filter_and_label_annotations_as_nuclei_and_cells(
                    image = image,
                    annotations = annotations,
                    annotation_min_area = ANNOTATION_MIN_AREA,
                    annotation_max_area = ANNOTATION_MAX_AREA,  # Or: original_width * original_height / 3
                    crystal_nucleus_max_area = CRYSTAL_NUCLEUS_MAX_AREA,
                    crystal_nucleus_color_hsv_max_value = CRYSTAL_NUCLEUS_COLOR_HSV_MAX_VALUE,
                    iou_threshold = IOU_THRESHOLD,
                    decode_coco_rle_masks = True
                )
            # If needed, flatten annotations by category, i.e. deal with overlapping masks in each category (class)
            if flatten_annotations_by_category:
                annotations = coco.flatten_masks_in_annotations_by_category(annotations, decode_coco_rle_masks = True)
            annotations_to_add = []
            for annotation in annotations:
                if filter_and_precategorize_annotations:
                    annotation_category_label = CRYSTAL_NUCLEUS_ANNOTATION.label if annotation['category_id'] == CRYSTAL_NUCLEUS_ANNOTATION.ID else CRYSTAL_CELL_ANNOTATION.label
                else:
                    annotation_category_label = CRYSTAL_NUCLEUS_ANNOTATION.label if default_annotation_category_id == CRYSTAL_NUCLEUS_ANNOTATION.ID else CRYSTAL_CELL_ANNOTATION.label
                annotation = {
                    "original_width": original_width,
                    "original_height": original_height,
                    "image_rotation": 0,
                    "value": {
                        "points": coco.rle_to_points(annotation, original_width, original_height),  # Time consumming task!
                        "closed": True,
                        "polygonlabels": [annotation_category_label]
                    },
                    "id": f"{image_id}_{annotation_id}",
                    "from_name": "label",  # "polygon",
                    "to_name": "image",
                    "type": "polygonlabels",  # "polygon",
                    # "origin": "manual",
                }
                annotations_to_add.append(annotation)
                annotation_id += 1
        # Create image information part
        label_studio_preannotations = {}
        label_studio_preannotations["data"] = {"image": image_s3_folder_path + '/' + annotations_file_path.name.split('.')[0] + '.jpg'}
        label_studio_preannotations["predictions"] = [{
            "model_version": model_version,
            #"score": 0.5,
            "result": annotations_to_add
        }]
        # Save the whole dataset as JSON file
        with open(label_studio_preannotations_folder_path / annotations_file_path.name, "w") as label_studio_preannotations_file:
            json.dump(label_studio_preannotations, label_studio_preannotations_file)  # Compact and light
            # json.dump(label_studio_preannotations, label_studio_preannotations_file, indent=3, sort_keys = False, separators = (',', ':'))  # Readable but heavy


def run_segment_anything_on_images_in_folder_and_save_annotations_to_folder(
        image_folder_path: Path,
        coco_rle_annotations_folder_path: Path,
        points_per_side=64,                # Default: 32
        points_per_batch=128,              # Default: 64
        pred_iou_thresh=0.86,              # Default: 0.88
        stability_score_thresh=0.92,       # Default: 0.95
        stability_score_offset=1.0,        # Default: 1.0
        box_nms_thresh=0.7,                # Default: 0.7
        crop_n_layers=2,                   # Default: 0
        crop_nms_thresh=0.7,               # Default: 0.7
        crop_overlap_ratio=512/1024,       # Default: 512/1500
        point_grids=None,                  # Default: None
        crop_n_points_downscale_factor=2,  # Default: 1
        min_mask_region_area=100,          # Default: 0, requires open-cv to run post-processing
    ):
    # If it doesn't exist, create the directory for the RLE annotations generated with Segment Anything (SAM)
    os.makedirs(coco_rle_annotations_folder_path, exist_ok=True)

    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("CUDA is available:", torch.cuda.is_available())
    print("MPS (Mac ARM GPU) is available:", torch.backends.mps.is_available())
    print("PyTorch was built with MPS activated:", torch.backends.mps.is_built())

    # To automatically generate masks (COCO annotations), give a SAM model to the SamAutomaticMaskGenerator class and
    # set its path to the SAM checkpoint.
    # Running SAM on CUDA and with the default model is recommended.

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # To run possibly on Nvidia GPU
    # Or to run on Apple M1/M2/... GPU [not working]
    #dtype = torch.float  # float32
    #device = torch.device('mps')

    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_MODEL_PATH.as_posix())
    sam.to(device=device)
    #sam.to(torch.float32).to(device=device)  # For 'mps' device

    #mask_generator = SamAutomaticMaskGenerator(sam)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,                # Default: 32
        points_per_batch=points_per_batch,              # Default: 64
        pred_iou_thresh=pred_iou_thresh,                # Default: 0.88
        stability_score_thresh=stability_score_thresh,  # Default: 0.95
        stability_score_offset=stability_score_offset,  # Default: 1.0
        box_nms_thresh=box_nms_thresh,                  # Default: 0.7
        crop_n_layers=crop_n_layers,                    # Default: 0
        crop_nms_thresh=crop_nms_thresh,                # Default: 0.7
        crop_overlap_ratio=crop_overlap_ratio,          # Default: 512/1500
        point_grids=point_grids,                        # Default: None
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,  # Default: 1
        min_mask_region_area=min_mask_region_area,      # Default: 0, requires open-cv to run post-processing
    )

    jpg_pattern = os.path.join(image_folder_path, '*.jpg')
    image_file_paths = [Path(ip) for ip in glob.glob(jpg_pattern)]

    image_file_progress_bar = tqdm(image_file_paths)
    for image_file_path in image_file_progress_bar:
        image_file_progress_bar.set_description(f"Running Segment Anything on image: {image_file_path.as_posix()}")

        # Read image from its file path
        image = cv2.imread(image_file_path.as_posix())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Generate annotations from image
        annotations = mask_generator.generate(image)

        # Save annotations to COCO RLE file
        annotation_file_name = f"{image_file_path.stem}.json"
        coco_rle_annotations_file_path = coco_rle_annotations_folder_path / annotation_file_name
        coco.save_annotations_to_coco_rle_file(annotations, coco_rle_annotations_file_path)


def get_crystal_nucleus_contours_and_annotations_in_image(gray_image, binarization_threshold=127):
    # Thresholding grayscaled image to obtain a binary image
    _, binary_image = cv2.threshold(gray_image, binarization_threshold, 255, cv2.THRESH_BINARY_INV)

    # Use morphology techniques to improve segmentation
    kernel = np.ones((5,5), np.uint8)
    morph = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)
    morph = cv2.dilate(morph, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert OpenCV contours to nucleus COCO annotations (with masks).
    coco_annotations = coco.get_coco_annotations_from_open_cv_contours(
        image=binary_image,
        contours=contours,
        category_id=CRYSTAL_NUCLEUS_ANNOTATION.ID,
        score=1.0,
        encode_masks_to_coco_rle=False
    )

    # Return OpenCV contours and COCO annotations
    return contours, coco_annotations


def run_opencv_crystal_nucleus_segmentation_on_images_in_folder_and_save_annotations_to_folder(
        image_folder_path: Path,
        coco_rle_annotations_folder_path: Path,
        binarization_threshold: int = 40
    ):
    # If it doesn't exist, create the directory for the RLE annotations generated with Segment Anything (SAM)
    os.makedirs(coco_rle_annotations_folder_path, exist_ok=True)

    jpg_pattern = os.path.join(image_folder_path, '*.jpg')
    image_file_paths = [Path(ip) for ip in glob.glob(jpg_pattern)]

    image_file_progress_bar = tqdm(image_file_paths)
    for image_file_path in image_file_progress_bar:
        image_file_progress_bar.set_description(f"Running opencv crystal nucleus segmentation on image: {image_file_path.as_posix()}")

        # Read image in gray levels from its file path
        gray_image = cv2.imread(image_file_path.as_posix(), cv2.IMREAD_GRAYSCALE)

        # Generate nucleus annotations from image
        _ , nucleus_annotations = get_crystal_nucleus_contours_and_annotations_in_image(gray_image, binarization_threshold)

        # Save annotations to COCO RLE file
        annotation_file_name = f"{image_file_path.stem}.json"
        coco_rle_annotations_file_path = coco_rle_annotations_folder_path / annotation_file_name
        coco.save_annotations_to_coco_rle_file(nucleus_annotations, coco_rle_annotations_file_path)


def get_crystal_defects_contours_and_annotations_in_image(mask_image, binarization_threshold=127, min_defect_area=10):
    """
    Detect defects in crystal masks using OpenCV.
    Defects are identified as clear areas (high intensity regions) within crystal masks.
    Only detects defects within the opaque parts of the image, ignoring transparent background.

    Args:
        mask_image: RGB, RGBA, or grayscale image of the crystal mask (uint8 or float)
        binarization_threshold: Pixel intensity threshold to identify clear areas (defects)
                              For uint8 images: 0-255 range
                              For float images: 0.0-1.0 range (will be auto-scaled)
        min_defect_area: Minimum area in pixels for a region to be considered a defect

    Returns:
        Tuple containing:
        - List of contours for crystal defects
        - List of COCO RLE annotations for crystal defects
    """

    # Normalize input image to uint8 if it's float
    def normalize_to_uint8(img):
        """Convert float image to uint8, handling different input ranges."""
        if img.dtype in [np.float32, np.float64]:
            # Assume float images are in [0, 1] range
            img_normalized = np.clip(img, 0, 1)
            return (img_normalized * 255).astype(np.uint8)
        elif img.dtype == np.uint8:
            return img
        else:
            # Handle other integer types
            if img.max() <= 1:
                return (img * 255).astype(np.uint8)
            else:
                return np.clip(img, 0, 255).astype(np.uint8)

    # Normalize the input image
    mask_image_uint8 = normalize_to_uint8(mask_image)

    # Adjust threshold for float input
    if mask_image.dtype in [np.float32, np.float64] and binarization_threshold <= 1.0:
        # Threshold is already in [0, 1] range for float images
        threshold_uint8 = int(binarization_threshold * 255)
    elif mask_image.dtype in [np.float32, np.float64] and binarization_threshold > 1.0:
        # Threshold is in [0, 255] range, scale it down for the original comparison
        threshold_uint8 = int(binarization_threshold)
    else:
        # uint8 image with uint8 threshold
        threshold_uint8 = int(binarization_threshold)

    # Check if the image has an alpha channel (transparency)
    has_alpha = mask_image_uint8.ndim >= 3 and mask_image_uint8.shape[2] == 4

    # Create a mask of non-transparent areas (the crystal cell)
    if has_alpha:
        # Extract alpha channel - this will be 0 for transparent pixels and 255 for opaque
        alpha_channel = mask_image_uint8[:, :, 3]

        # Create a binary mask of the crystal cell (non-transparent areas)
        # Use a threshold to account for semi-transparent pixels if needed
        _, crystal_mask = cv2.threshold(alpha_channel, 10, 255, cv2.THRESH_BINARY)
    else:
        # If there's no alpha channel, assume the entire image is the crystal cell
        crystal_mask = np.ones(mask_image_uint8.shape[:2], dtype=np.uint8) * 255

    # Convert the RGB(A) image to grayscale for defect detection
    if has_alpha:
        # Convert RGBA to RGB first, then to grayscale
        rgb_image = cv2.cvtColor(mask_image_uint8, cv2.COLOR_RGBA2RGB)
        gray_mask = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    elif len(mask_image_uint8.shape) > 2 and mask_image_uint8.shape[2] >= 3:
        # Convert RGB to grayscale
        gray_mask = cv2.cvtColor(mask_image_uint8, cv2.COLOR_RGB2GRAY)
    elif len(mask_image_uint8.shape) > 2 and mask_image_uint8.shape[2] == 1:
        # Single channel with extra dimension
        gray_mask = mask_image_uint8[:, :, 0]
    else:
        # Already grayscale
        gray_mask = mask_image_uint8.copy()

    # Threshold to identify bright areas (defects) within the crystal
    _, binary_defects = cv2.threshold(gray_mask, threshold_uint8, 255, cv2.THRESH_BINARY)

    # Ensure that the 2 masks have the same type
    # print(f"binary_defects - Shape: {binary_defects.shape}, Type: {binary_defects.dtype}")
    # print(f"crystal_mask - Shape: {crystal_mask.shape}, Type: {crystal_mask.dtype}")
    # binary_defects = binary_defects.astype(np.float64)
    # crystal_mask = crystal_mask.astype(np.float64)

    # Ensure matching shape and dtype for bitwise operation
    # if crystal_mask.shape != binary_defects.shape:
    #     crystal_mask = cv2.resize(crystal_mask, (binary_defects.shape[1], binary_defects.shape[0]))
    # if crystal_mask.dtype != binary_defects.dtype:
    #     crystal_mask = crystal_mask.astype(binary_defects.dtype)

    # Only consider defects within the crystal cell (not in transparent areas)
    # Perform a bitwise AND between the defect mask and the crystal mask
    defects_in_crystal = cv2.bitwise_and(binary_defects, crystal_mask)

    # Use morphological operations to clean up the defect mask
    kernel = np.ones((3, 3), np.uint8)
    # Opening to remove small noise
    morph = cv2.morphologyEx(defects_in_crystal, cv2.MORPH_OPEN, kernel, iterations=1)
    # Closing to connect nearby defect regions
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Ensure binary image is in correct format
    if morph.dtype != np.uint8:
        morph = (morph * 255 if morph.max() <= 1.0 else morph).astype(np.uint8)

    # Find contours of defects
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by area
    filtered_contours = []
    for contour in contours:
        if cv2.contourArea(contour) >= min_defect_area:
            filtered_contours.append(contour)
    contours = filtered_contours

    # Convert OpenCV contours to nucleus COCO annotations (with masks).
    coco_annotations = coco.get_coco_annotations_from_open_cv_contours(
        image=morph,  # Use the morphologically cleaned mask for annotations
        contours=contours,
        category_id=CRYSTAL_DEFECT_ANNOTATION.ID,
        score=1.0,
        encode_masks_to_coco_rle=False
    )

    # Return the contours and annotations
    return contours, coco_annotations


