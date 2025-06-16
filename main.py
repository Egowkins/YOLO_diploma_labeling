import os
import re
import shutil

def merge_filtered_dataset(base_path, pattern, output_path, target_class):

    os.makedirs(output_path, exist_ok=True)

    items = os.listdir(base_path)
    print("Содержимое base_path:")
    for it in items:
        print("   ", it)
    print()


    candidate_folders = []
    print(f"Проверяем папки на соответствие шаблону `{pattern}`:")
    for name in items:
        full = os.path.join(base_path, name)
        if os.path.isdir(full):
            m = re.search(pattern, name)
            print(f"{name:30} - директория - {'MATCH' if m else 'NO MATCH'}")
            if m:
                candidate_folders.append(name)
        else:
            print(f"{name:30} - не директория")
    print()

    if not candidate_folders:
        print("Не найдено ни одной папки по заданному шаблону")
        return

    candidate_folders.sort()
    print("Папки, подходящие под шаблон:", candidate_folders, "\n")

    global_index = 1


    for folder in candidate_folders:
        outer_path = os.path.join(base_path, folder)
        print(f"Обработка папки: {folder}")

        subfolders = [d for d in os.listdir(outer_path) if os.path.isdir(os.path.join(outer_path, d))]
        if not subfolders:
            print("Pass.\n")
            continue

        for sub in subfolders:
            sub_path = os.path.join(outer_path, sub)
            data_path = os.path.join(sub_path, 'data')
            data_path = os.path.join(data_path, 'obj_train_data')
            if not os.path.isdir(data_path):
                print(f"{sub_path} Pass")
                continue

            print(f"Найдена папка : {data_path}")
            files = os.listdir(data_path)
            images = [f for f in files if f.lower().endswith(('.jpg','.jpeg','.png'))]

            images.sort(key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else float('inf'))
            print(f"Найдено изображений: {len(images)}")

            for img in images:
                base_name = os.path.splitext(img)[0]
                txt = base_name + '.txt'
                img_path = os.path.join(data_path, img)
                txt_path = os.path.join(data_path, txt)

                if not os.path.exists(txt_path):
                    print(f"Пропущено: нет аннотации {txt} для {img}")
                    continue

                with open(txt_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                if not lines:
                    print(f"Аннотация {txt} PASS")
                    continue

                filtered = [l for l in lines if l.strip().startswith(f"{target_class} ")]
                if not filtered:
                    print(f"В {txt} нет строк с классом {target_class}, PASS.")
                    continue

                new_img = f"{global_index}.jpeg"
                new_txt = f"{global_index}.txt"
                shutil.copy(img_path, os.path.join(output_path, new_img))
                with open(os.path.join(output_path, new_txt), 'w', encoding='utf-8') as f:
                    f.writelines(filtered)

                print(f"Скопировано: {new_img}")
                global_index += 1
            print()

    print(f"Всего скопировано: {global_index - 1} файлов в {output_path}")

if __name__ == '__main__':
    merge_filtered_dataset(
        base_path="C:/Users/Егошка/Documents/__диплом/Датасет/cesium/Abrams",
        pattern=r'.*wo_mask_.*',      # ваша регулярка
        output_path="C:/Users/Егошка/Documents/__диплом/merged_dataset/",
        target_class=5
    )
