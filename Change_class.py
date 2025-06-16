import os
import glob

# Жёстко заданные параметры (меняй здесь)
FOLDER_PATH = "C:/Users/Егошка/Documents/__диплом/FULL_DATASET/FULL_T_90_SCRIMMED_FOR_YOLO_3/labels"  # <- Вставь свой путь
NEW_CHAR = "3"  # <- Вставь символ для замены


def replace_first_char():
    # Поиск .txt файлов
    txt_files = glob.glob(os.path.join(FOLDER_PATH, "*.txt"))

    for file_path in txt_files:
        try:
            # Чтение файла
            with open(file_path, "r+", encoding="utf-8") as file:
                content = file.read()

                if not content:
                    print(f"⚠️ Пустой файл: {file_path}")
                    continue

                # Модификация контента
                new_content = NEW_CHAR + content[1:]

                # Перезапись
                file.seek(0)
                file.write(new_content)
                file.truncate()

            print(f"✅ Успех: {os.path.basename(file_path)}")

        except Exception as e:
            print(f"🔥 Ошибка в {os.path.basename(file_path)}: {str(e)}")


if __name__ == "__main__":
    replace_first_char()