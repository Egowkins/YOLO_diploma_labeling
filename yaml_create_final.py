import os
import yaml
from glob import glob

# Укажи вручную путь к папке с train/ и val/
dataset_path = r"C:\Users\Егошка\Documents\__диплом\COMBINED_DATASET"

train_images = os.path.join(dataset_path, 'train/images')
val_images = os.path.join(dataset_path, 'val/images')
train_labels = os.path.join(dataset_path, 'train/labels')

# Считаем количество классов
class_ids = set()
for label_file in glob(os.path.join(train_labels, '*.txt')):
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                class_id = int(line.strip().split()[0])
                class_ids.add(class_id)
nc = max(class_ids) + 1 if class_ids else 0

# Собираем данные
data = {
    'train': train_images.replace('\\', '/'),
    'val': val_images.replace('\\', '/'),
    'nc': nc,
    'names': [f'class{i}' for i in range(nc)]
}

# Сохраняем в dataset.yaml
with open('dataset.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(data, f, allow_unicode=True)

print("✅ YAML сохранён как dataset.yaml")
