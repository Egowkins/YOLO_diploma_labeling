from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
import os
import cv2
import numpy as np
import supervision as sv
from pathlib import Path
from tqdm import tqdm

class FixedGroundingDINO(GroundingDINO):
    def label(self, input_folder: str, extension: str = ".jpg", output_folder: str = None) -> dict:
        def upscale_image(image, scale=2):
            return cv2.resize(image, (image.shape[1] * scale, image.shape[0] * scale), interpolation=cv2.INTER_LINEAR)

        def filter_detections(detections, min_area=800, min_width=20, min_height=20, max_aspect_ratio=3.0):
            if detections.xyxy.size == 0:
                return sv.Detections(xyxy=np.empty((0, 4)), confidence=np.empty(0), class_id=np.empty(0))

            print(f"Shape of detections.xyxy: {detections.xyxy.shape}")  # Для отладки

            boxes = detections.xyxy
            scores = detections.confidence
            class_ids = detections.class_id

            filtered_boxes = []
            filtered_scores = []
            filtered_class_ids = []

            for box, score, class_id in zip(boxes, scores, class_ids):
                x_min, y_min, x_max, y_max = box
                width = x_max - x_min
                height = y_max - y_min
                area = width * height
                aspect_ratio = max(width / height, height / width)

                if area >= min_area and width >= min_width and height >= min_height and aspect_ratio <= max_aspect_ratio:
                    filtered_boxes.append(box)
                    filtered_scores.append(score)
                    filtered_class_ids.append(class_id)

            return sv.Detections(
                xyxy=np.array(filtered_boxes),
                confidence=np.array(filtered_scores),
                class_id=np.array(filtered_class_ids)
            )
        image_paths = [
                os.path.join(input_folder, f)
                for f in os.listdir(input_folder)
                if f.endswith(extension)
        ]

        results = {}

        for path in tqdm(image_paths, desc="Labeling images"):
            image = cv2.imread(path)
            if image is None:
                    print(f" Failed to load image: {path}")
                    continue

            image = upscale_image(image, scale=2)

            detections = self.predict(image)
            detections = filter_detections(detections)

            image_name = Path(path).name
            results[image_name] = {
                "image": image,
                "detections": detections
            }

            if output_folder:
                os.makedirs(output_folder, exist_ok=True)
                annotated = sv.BoxAnnotator().annotate(image.copy(), detections)
                cv2.imwrite(os.path.join(output_folder, image_name), annotated)
        return results


base_model = FixedGroundingDINO(
    ontology=CaptionOntology({
        "tank": "abrams",
        "battle tank": "abrams",
        "military tank": "abrams",
        "armored vehicle": "abrams",
        "vehicle": "abrams"
    }),
    box_threshold=0.20,
    text_threshold=0.25
)
def lets_go(IMAGE_DIR_PATH: str = 'images',DATASET_DIR_PATH: str = 'dataset' ):

    dataset = base_model.label(
        input_folder=IMAGE_DIR_PATH,
        extension=".jpg",
        output_folder=DATASET_DIR_PATH
    )
    return dataset

def save_detections_to_txt(dataset: dict, output_folder: str, class_name_to_id: dict):


    os.makedirs(output_folder, exist_ok=True)

    for image_name, data in dataset.items():
        detections = data["detections"]
        image = data["image"]
        height, width = image.shape[:2]

        # Имя файла для записи
        txt_filename = os.path.splitext(image_name)[0] + ".txt"
        txt_path = os.path.join(output_folder, txt_filename)

        lines = []

        for bbox, class_id in zip(detections.xyxy, detections.class_id):
            x_min, y_min, x_max, y_max = bbox


            x_center = (x_min + x_max) / 2 / width
            y_center = (y_min + y_max) / 2 / height
            bbox_width = (x_max - x_min) / width
            bbox_height = (y_max - y_min) / height
            class_id = class_name_to_id.get(class_id, 0)
            line = f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"
            lines.append(line)

        with open(txt_path, "w") as f:
            f.write("\n".join(lines))





