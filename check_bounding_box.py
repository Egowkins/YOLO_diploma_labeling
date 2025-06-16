import cv2
import os


img_dir = 'images'
labels_dir = 'new_labels'
output_dir = 'check'


os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(img_dir):
    raise FileNotFoundError(f"Директория с изображениями не найдена: {img_dir}")
if not os.path.exists(labels_dir):
    raise FileNotFoundError(f"Директория с метками не найдена: {labels_dir}")

print(f"Обработка изображений из: {img_dir}")


for filename in os.listdir(img_dir):

    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    image_path = os.path.join(img_dir, filename)
    label_filename = os.path.splitext(filename)[0] + '.txt'
    label_path = os.path.join(labels_dir, label_filename)
    output_path = os.path.join(output_dir, filename)


    if not os.path.exists(label_path):
        print(f"Нет меток для: {filename}")
        continue


    image = cv2.imread(image_path)
    if image is None:
        print(f"Не удалось прочитать файл: {image_path}")
        continue


    h, w = image.shape[:2]
    has_boxes = False


    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    class_id, cx, cy, bw, bh = map(float, parts)
                    x1 = int((cx - bw/2) * w)
                    y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w)
                    y2 = int((cy + bh/2) * h)


                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image,
                                str(int(class_id)),
                                (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                1)
                    has_boxes = True
                except Exception as e:
                    print(f"[ОШИБКА] Ошибка обработки строки '{line}' в файле {label_path}: {str(e)}")


    if has_boxes:
        try:
            cv2.imwrite(output_path, image)
            print(f"Сохранено: {filename}")
        except Exception as e:
            print(f"Не удалось сохранить {filename}: {str(e)}")
    else:
        print(f"Нет боксов в: {filename}")

