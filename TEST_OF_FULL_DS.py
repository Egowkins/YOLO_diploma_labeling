import os
import shutil
from pathlib import Path


src_root = Path('C:/Users/Егошка/Documents/__диплом/FULL_DATASET/FULL_T_90_2')
dst_root = Path('C:/Users/Егошка/Documents/__диплом/FULL_DATASET/FULL_T_90_FOR_YOLO_2')
fixed_class = '2'
sample_step = 2

img_out = dst_root / 'Img'
lbl_out = dst_root / 'labels'
img_out.mkdir(parents=True, exist_ok=True)
lbl_out.mkdir(parents=True, exist_ok=True)


def convert_polygon_to_bbox(in_path: Path, out_path: Path, cls: str):
    with in_path.open('r', encoding='utf-8') as inf, out_path.open('w', encoding='utf-8') as outf:
        for line in inf:
            parts = line.strip().split()
            if len(parts) < 3 or len(parts) % 2 == 0:
                continue
            coords = list(map(float, parts[1:]))
            xs, ys = coords[::2], coords[1::2]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            x_c = (x_min + x_max) / 2
            y_c = (y_min + y_max) / 2
            w = x_max - x_min
            h = y_max - y_min
            outf.write(f"{cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")


def is_yolo_format(txt_file: Path) -> bool:
    try:
        with txt_file.open('r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    return False

                for val in parts[1:]:
                    float(val)
        return True
    except Exception:
        return False


def process_dir(path: Path, start_idx: int, step: int = 1, convert_polygons: bool = True):
    idx = start_idx
    images = sorted([f for f in path.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']], key=lambda f: f.name)
    texts = {f.stem.lower(): f for f in path.iterdir() if f.is_file() and f.suffix.lower() == '.txt'}

    for i, img_file in enumerate(images):
        if (i % step) != 0:
            continue
        stem = img_file.stem.lower()
        txt_file = texts.get(stem)

        shutil.copy(img_file, img_out / f"{idx}.jpg")

        if convert_polygons:

            convert_polygon_to_bbox(txt_file, lbl_out / f"{idx}.txt", fixed_class)
        else:

            with txt_file.open('r', encoding='utf-8') as inf, (lbl_out / f"{idx}.txt").open('w', encoding='utf-8') as outf:
                for line in inf:
                    elems = line.strip().split()
                    if len(elems) != 5:
                        continue
                    outf.write(f"{fixed_class} {' '.join(elems[1:])}\n")

        idx += 1
    return idx


def find_all_src_dirs(root: Path):
    cascade_dirs = []
    simple_dirs = []
    for dirpath, _, _ in os.walk(root):
        p = Path(dirpath)
        if p.name == 'obj_train_data':
            cascade_dirs.append(p)
        else:
            files = list(p.iterdir()) if p.is_dir() else []
            if any(f.suffix.lower() in ['.jpg', '.jpeg', '.png'] for f in files) and any(f.suffix.lower() == '.txt' for f in files):
                simple_dirs.append(p)
    simple_clean = [d for d in simple_dirs if 'obj_train_data' not in str(d)]
    cascade_dirs = sorted(set(cascade_dirs), key=lambda x: str(x))
    simple_clean = sorted(set(simple_clean), key=lambda x: str(x))
    return cascade_dirs, simple_clean



cascade_dirs, simple_dirs = find_all_src_dirs(src_root)


next_idx = 1


for d in cascade_dirs:
    next_idx = process_dir(d, next_idx, step=1, convert_polygons=True)


for d in simple_dirs:
    next_idx = process_dir(d, next_idx, step=sample_step, convert_polygons=False)


