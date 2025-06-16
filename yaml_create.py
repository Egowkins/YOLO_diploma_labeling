import os
import random
import shutil

def split_and_make_yaml(
    dataset_path: str,
    output_path: str,
    val_ratio: float = 0.3,
    seed: int = 42,
    class_names: list[str] | None = None
):
    """
    Разбивает датасет на train/val и пишет data.yaml для YOLOv11.

    Args:
        dataset_path: папка с merged_dataset, содержащая *.jpg + *.txt.
        output_path: корень, куда положим train/, val/ и data.yaml.
        val_ratio: доля валидации (0.0–1.0).
        seed: сид для рандома.
        class_names: опционально, список имён классов;
                     если None — будет сгенерирован автоматически.
    """
    random.seed(seed)
    os.makedirs(output_path, exist_ok=True)

    # 1) Найти все изображения и их метки
    imgs = sorted(f for f in os.listdir(dataset_path)
                  if f.lower().endswith(('.jpg','.jpeg','.png')))
    lbls = {os.path.splitext(f)[0]: f for f in os.listdir(dataset_path)
            if f.lower().endswith('.txt')}

    # Оставляем только те изображения, у которых есть .txt
    pairs = [img for img in imgs if os.path.splitext(img)[0] in lbls]
    if not pairs:
        raise RuntimeError("Не найдено ни одного пары .jpg + .txt в " + dataset_path)

    # 2) Перемешиваем и делим
    random.shuffle(pairs)
    n_val = int(len(pairs) * val_ratio)
    val_set = set(pairs[:n_val])
    train_set = set(pairs[n_val:])

    # 3) Создаём папки
    for split in ('train', 'val'):
        for sub in ('images','labels'):
            os.makedirs(os.path.join(output_path, split, sub), exist_ok=True)

    # 4) Копируем
    for img in train_set:
        name = os.path.splitext(img)[0]
        shutil.copy(
            os.path.join(dataset_path, img),
            os.path.join(output_path, 'train', 'images', img)
        )
        shutil.copy(
            os.path.join(dataset_path, lbls[name]),
            os.path.join(output_path, 'train', 'labels', lbls[name])
        )

    for img in val_set:
        name = os.path.splitext(img)[0]
        shutil.copy(
            os.path.join(dataset_path, img),
            os.path.join(output_path, 'val', 'images', img)
        )
        shutil.copy(
            os.path.join(dataset_path, lbls[name]),
            os.path.join(output_path, 'val', 'labels', lbls[name])
        )

    # 5) Автоматический поиск классов в .txt
    class_ids = set()
    for txt in os.listdir(dataset_path):
        if not txt.lower().endswith('.txt'):
            continue
        with open(os.path.join(dataset_path, txt), 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    try:
                        class_ids.add(int(parts[0]))
                    except ValueError:
                        pass
    class_ids = sorted(class_ids)
    nc = len(class_ids)

    # 6) Формируем names: либо переданный, либо auto ["class0",...]
    if class_names is None:
        names = [f"class{cid}" for cid in class_ids]
    else:
        if len(class_names) != nc:
            raise ValueError(f"Передано {len(class_names)} имён, "
                             f"а найдено {nc} классов.")
        names = class_names

    # 7) Пишем YAML
    yaml_path = os.path.join(output_path, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(f"train: {os.path.abspath(os.path.join(output_path, 'train', 'images'))}\n")
        f.write(f"val:   {os.path.abspath(os.path.join(output_path, 'val', 'images'))}\n\n")
        f.write(f"nc: {nc}\n")
        # YAML-список
        names_list = "[" + ", ".join(f"'{n}'" for n in names) + "]"
        f.write(f"names: {names_list}\n")

    print(f"✅ Разбивка завершена:\n"
          f"   train: {len(train_set)} примеров\n"
          f"   val:   {len(val_set)} примеров\n"
          f"YAML сохранён в {yaml_path}")

# Пример запуска:
if __name__ == '__main__':
    split_and_make_yaml(
        dataset_path="C:/Users/Егошка/Documents/__диплом/merged_dataset",
        output_path="C:/Users/Егошка/Documents/__диплом/yolov11_data",
        val_ratio=0.3,
        seed=123,
        # class_names=['tank','bmp','truck']  # опционально, если хотите свои названия
    )
