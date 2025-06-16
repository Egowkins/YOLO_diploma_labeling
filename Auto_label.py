import os
from delete_some import  deliter, check, show_pics, show_results, process_annotation_file, save_annotations
from make_auto_label import lets_go, save_detections_to_txt
import supervision as sv
from pathlib import Path




if __name__ == '__main__':
    # IMAGE_DIR_PATH = Path('C:/obj_labeling/labeling/images')
    # DATASET_DIR_PATH = Path('C:/obj_labeling/labeling/dataset')
    # check(IMAGE_DIR_PATH)
    # deliter(IMAGE_DIR_PATH, 1200)
    # show_pics(True)
    # result = lets_go()
    # show_results(result)
    # class_name_to_id = {
    #     "abrams": 2
    # }
    # ANNOTATIONS_DIR = "./labels"
    # save_detections_to_txt(result, ANNOTATIONS_DIR, class_name_to_id)
    annotations_dir = 'labels'
    output_dir = 'new_labels'
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(annotations_dir):
        if not filename.endswith('.txt'):
            continue

        input_path = os.path.join(annotations_dir, filename)
        output_path = os.path.join(output_dir, filename)

        cleaned_annotations = process_annotation_file(input_path)


        if cleaned_annotations is None:
            print(f" Пропущен {filename} (None)")
            continue


        save_annotations(cleaned_annotations, output_path)
        print(f" Сохранён {filename}")


    show_pics(True)



