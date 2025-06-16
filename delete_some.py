import os
import cv2
import numpy as np
import supervision as sv
from pathlib import Path
from tqdm import tqdm

def deliter(IMAGE_DIR_PATH: str, number_of_files=800):
    """
    Функция для удаления файлов в папке с изображениями, если номер файла > number_of_files.
    """

    image_dir = Path(IMAGE_DIR_PATH)


    for filename in os.listdir(image_dir):

        if filename.startswith("frame_") and filename.endswith(".jpg"):
            try:

                number_str = filename.split("_")[1].split(".")[0]
                number = int(number_str)


                if number > number_of_files:
                    file_path = image_dir / filename
                    os.remove(file_path)
                    print(f"Удалено: {filename}")
            except (IndexError, ValueError):

                continue

def check(IMAGE_DIR_PATH):
    """
    Функция для проверки количества изображений в директории.
    """

    image_paths = list(IMAGE_DIR_PATH.glob('*.[pj]*[ng]*'))  # Получаем все изображения в формате png и jpg

    print('image count:', len(image_paths))

def show_pics(flag: bool = True,
              IMAGE_DIR_PATH = Path('C:/obj_labeling/labeling/images'),
              SAMPLE_SIZE = 16,
              SAMPLE_GRID_SIZE = (4, 4),
              SAMPLE_PLOT_SIZE = (16, 16)):
    """
    Функция для отображения примеров изображений.
    """


    image_files = list(IMAGE_DIR_PATH.glob('*.jpg'))

    if flag:

        titles = [image_file.stem for image_file in image_files[:SAMPLE_SIZE]]
        images = [cv2.imread(str(image_file)) for image_file in image_files[:SAMPLE_SIZE]]


        sv.plot_images_grid(images=images, titles=titles, grid_size=SAMPLE_GRID_SIZE, size=SAMPLE_PLOT_SIZE)

def show_results(dataset,
                 SAMPLE_SIZE=16,
                 SAMPLE_GRID_SIZE=(4, 4),
                 SAMPLE_PLOT_SIZE=(16, 16)):
    """
    Функция для отображения аннотированных изображений с результатами.
    """

    image_names = list(dataset.keys())[:SAMPLE_SIZE]

    mask_annotator = sv.MaskAnnotator()
    box_annotator = sv.BoxAnnotator()

    images = []
    for image_name in image_names:

        image = dataset[image_name]["image"]
        detections = dataset[image_name]["detections"]


        if hasattr(detections, "class_id") and detections.class_id is not None:
            labels = [str(class_id) for class_id in detections.class_id]
        else:
            labels = ["object"] * len(detections.xyxy)


        annotated_image = mask_annotator.annotate(
            scene=image.copy(),
            detections=detections
        )
        annotated_image = box_annotator.annotate(
            scene=annotated_image,
            detections=detections,
            labels=labels
        )

        images.append(annotated_image)


    sv.plot_images_grid(
        images=images,
        titles=image_names,
        grid_size=SAMPLE_GRID_SIZE,
        size=SAMPLE_PLOT_SIZE
    )

import os
import itertools


SIZE_THRESHOLD = 0.65
SIMILARITY_THRESHOLD = 0.1

def parse_yolo_annotation(line):
    """
    Разбор строки аннотации YOLO: [class_id, x_center, y_center, width, height]
    """
    parts = line.strip().split()
    class_id = int(parts[0])
    x_center, y_center, width, height = map(float, parts[1:])
    return class_id, x_center, y_center, width, height

def calculate_area(width, height):
    """
    Площадь бокса
    """
    return width * height

def are_similar(ann1, ann2):
    """
    Сравниваем 2 аннотации по центрам и размерам
    """
    _, xc1, yc1, w1, h1 = ann1
    _, xc2, yc2, w2, h2 = ann2

    xc_diff = abs(xc1 - xc2)
    yc_diff = abs(yc1 - yc2)
    w_diff = abs(w1 - w2)
    h_diff = abs(h1 - h2)

    return (xc_diff < SIMILARITY_THRESHOLD and
            yc_diff < SIMILARITY_THRESHOLD and
            w_diff < SIMILARITY_THRESHOLD and
            h_diff < SIMILARITY_THRESHOLD)

def process_annotation_file(file_path):
    """
    Полная обработка одного файла аннотаций
    """
    annotations = []


    with open(file_path, 'r') as file:
        for line in file:
            if line.strip() == '':
                continue
            annotations.append(parse_yolo_annotation(line))


    annotations = [
        ann for ann in annotations
        if calculate_area(ann[3], ann[4]) < SIZE_THRESHOLD
    ]

    if len(annotations) == 0:

        return None

    if len(annotations) == 1:

        return [annotations[0]]


    for ann1, ann2 in itertools.combinations(annotations, 2):
        if not are_similar(ann1, ann2):

            return None


    return [annotations[0]]

def save_annotations(annotations, output_path):
    """
    Сохраняем обработанные аннотации обратно в файл
    """
    with open(output_path, 'w') as file:
        for ann in annotations:
            class_id, x_center, y_center, width, height = ann
            file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")



