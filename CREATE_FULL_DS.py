import os
import shutil
import random
from pathlib import Path

src_folders = {
    Path(r'C:/Users/Егошка/Documents/__диплом/FULL_DATASET/FULL_ABRAMS_FOR_YOLO_0'): 0,
    Path(r'C:/Users/Егошка/Documents/__диплом/FULL_DATASET/FULL_MERKAVA_FOR_YOLO_1'): 1,
    Path(r'C:/Users/Егошка/Documents/__диплом/FULL_DATASET/FULL_T_90_FOR_YOLO_2'): 2,
    Path(r'C:/Users/Егошка/Documents/__диплом/FULL_DATASET/FULL_T_90_SCRIMMED_FOR_YOLO_3'): 3,
    Path(r'C:/Users/Егошка/Documents/__диплом/FULL_DATASET/FULL_T_90_COPECAGE_FOR_YOLO_4'): 4,
    Path(r'C:/Users/Егошка/Documents/__диплом/FULL_DATASET/FULL_MERKAVA_SCRIMMED_FOR_YOLO_5'): 5,
    Path(r'C:/Users/Егошка/Documents/__диплом/FULL_DATASET/FULL_MERKAVA_COPECAGE_FOR_YOLO_6'): 6,
    Path(r'C:/Users/Егошка/Documents/__диплом/FULL_DATASET/FULL_ABRAMS_SCRIMMED_FOR_YOLO_7'): 7,
    Path(r'C:/Users/Егошка/Documents/__диплом/FULL_DATASET/FULL_ABRAMS_COPECAGE_FOR_YOLO_8'): 8,
}
dst_root = Path(r'C:/Users/Егошка/Documents/__диплом/COMBINED_DATASET')
train_ratio = 0.7

train_img = dst_root / 'train' / 'Img'
train_lbl = dst_root / 'train' / 'labels'
val_img = dst_root / 'val' / 'Img'
val_lbl = dst_root / 'val' / 'labels'
for p in [train_img, train_lbl, val_img, val_lbl]:
    p.mkdir(parents=True, exist_ok=True)

all_items = []
for folder, cls_id in src_folders.items():
    imgs = list((folder / 'Img').glob('*.jpg'))
    for img in imgs:
        lbl = folder.parent / 'labels' / f"{img.stem}.txt"
        if not lbl.exists():
            lbl = folder / 'labels' / f"{img.stem}.txt"
        if not lbl.exists():
            print(f"Пропуск {img}: нет метки")
            continue
        all_items.append((img, lbl, cls_id))

grandom = random.Random(42)
grandom.shuffle(all_items)
split_idx = int(len(all_items) * train_ratio)
train_items = all_items[:split_idx]
val_items = all_items[split_idx:]


def copy_and_relabel(items, img_out, lbl_out, start_idx=1):
    idx = start_idx
    for img, lbl, cls in items:
        new_img = img_out / f"{idx}.jpg"
        new_lbl = lbl_out / f"{idx}.txt"
        shutil.copy(img, new_img)
        with lbl.open('r', encoding='utf-8') as f_in, new_lbl.open('w', encoding='utf-8') as f_out:
            for line in f_in:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                f_out.write(str(cls) + ' ' + ' '.join(parts[1:]) + '\n')
        idx += 1
    return idx

next_idx = 1
next_idx = copy_and_relabel(train_items, train_img, train_lbl, next_idx)
copy_and_relabel(val_items, val_img, val_lbl, next_idx)

print(f"Готово. Train: {len(train_items)}, Val: {len(val_items)}")
