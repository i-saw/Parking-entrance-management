import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO
import torch
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from difflib import SequenceMatcher
import json

settings_file = "settings.json"  # Путь к файлу настроек
initial_settings = None  # Для отслеживания изменений настроек
allowed_numbers_list = "allowed_numbers_list.txt"

with open(allowed_numbers_list, "r", encoding="utf-8") as f:
    allowed_numbers = f.read().split(",")
    # Удаляем пустые строки из списка номеров
    allowed_numbers = [num.strip() for num in allowed_numbers if num.strip()]

root = tk.Tk()   # Создаем основное окно

# Получение максимальных размеров экрана
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Устанавливаем начальные размеры окна
root.state('zoomed')
root.geometry(f"{screen_width}x{screen_height}")

# Классы для распознавания транспортных средств и символов
vehicle_classes = ['car', 'truck', 'bus']
classification_classes = ['ambulance', 'firetrucks', 'normal_car', 'police']
#classification_classes = ['Emergency_service', 'Fire_Department', 'MCHS', 'Med_help', 'Normal_car', 'Police']
plate_classes = ['black', 'blue', 'normal', 'red', 'square', 'yellow']
letters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'E', 'H', 'K', 'M', 'O', 'P', 'T', 'X', 'Y']

# Инициализация глобальных переменных для чекбоксов (все чекбоксы по умолчанию включены)
plate_vars = [tk.IntVar(value=1) for _ in plate_classes]

# Загрузка моделей YOLO
model_vehicle = YOLO("yolov8n.pt")
model_classify = YOLO("classify_4_max_last.pt")
model_np = YOLO("detect_number_model+best.pt")
model_char = YOLO("symbol_detect_416_long_time_best.pt")
model_plate_classify = YOLO("model_classific_type_number_last.pt")

# Проверка наличия GPU и перемещение моделей на GPU, если доступен
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_vehicle.to(device)
model_classify.to(device)
model_np.to(device)
model_char.to(device)
model_plate_classify.to(device)

# Глобальные переменные для исходного размера видео
VIDEO_WIDTH = None
VIDEO_HEIGHT = None
allowed_plate_types = set()
allow_all = False
allowed_errors = 0
video_capture = None
is_video_loaded = False
is_paused = False
PLATE_WIDTH = 300
PLATE_HEIGHT = 100
# Инициализация глобальной переменной для хранения таймера
gate_timer_id = None
time_open = 10000

# Параметры обработки видео
top = tk.DoubleVar(value=38)
bottom = tk.DoubleVar(value=60)
left = tk.DoubleVar(value=10)
right = tk.DoubleVar(value=60)
alpha = tk.DoubleVar(value=1.5)
beta = tk.IntVar(value=20)
frame_skip_var = tk.IntVar(value=3)
source_var = tk.StringVar(value="Camera")
time_wait = tk.DoubleVar(value=3)
allow_all_var = tk.IntVar(value=0)
errors_var = tk.IntVar(value=0)
time_open = tk.IntVar(value=10000)

# Параметры пороговых значений для моделей YOLO
threshold_vehicle = tk.DoubleVar(value=0.60)
threshold_classify = tk.DoubleVar(value=0.50)
threshold_np = tk.DoubleVar(value=0.40)
threshold_char = tk.DoubleVar(value=0.70)
threshold_plate_classify = tk.DoubleVar(value=0.50)

# Функция для получения текущих настроек
def get_current_settings():
    return {
        "top": top.get(),
        "bottom": bottom.get(),
        "left": left.get(),
        "right": right.get(),
        "alpha": alpha.get(),
        "beta": beta.get(),
        "frame_skip_var": frame_skip_var.get(),
        "source_var": source_var.get(),
        "time_wait": time_wait.get(),
        "threshold_vehicle": threshold_vehicle.get(),
        "threshold_classify": threshold_classify.get(),
        "threshold_np": threshold_np.get(),
        "threshold_char": threshold_char.get(),
        "threshold_plate_classify": threshold_plate_classify.get(),
        "allowed_plate_types": [plate_type for var, plate_type in zip(plate_vars, plate_classes) if var.get()],
        "allow_all": bool(allow_all_var),
        "time_open": time_open.get(),
        "allowed_errors": int(errors_var.get())
    }


# Функция для сохранения настроек в JSON файл
def save_settings_to_json():
    settings_data = get_current_settings()
    print(f"Сохранение настроек: {settings_data}")
    # Сохраняем данные в файл
    with open(settings_file, 'w', encoding='utf-8') as f:
        json.dump(settings_data, f, ensure_ascii=False, indent=4)


# Функция для загрузки настроек из JSON файла
def load_settings_from_json():
    global allow_all, allowed_errors, allowed_plate_types

    try:
        with open(settings_file, 'r', encoding='utf-8') as f:
            settings_data = json.load(f)
            print(f"Загруженные настройки: {settings_data}")

            # Применяем загруженные данные к переменным
            top.set(settings_data.get("top", 38))
            bottom.set(settings_data.get("bottom", 60))
            left.set(settings_data.get("left", 10))
            right.set(settings_data.get("right", 60))
            alpha.set(settings_data.get("alpha", 1.5))
            beta.set(settings_data.get("beta", 20))
            frame_skip_var.set(settings_data.get("frame_skip_var", 3))
            source_var.set(settings_data.get("source_var", "Camera"))
            time_wait.set(settings_data.get("time_wait", 3))
            threshold_vehicle.set(settings_data.get("threshold_vehicle", 0.60))
            threshold_classify.set(settings_data.get("threshold_classify", 0.50))
            threshold_np.set(settings_data.get("threshold_np", 0.40))
            threshold_char.set(settings_data.get("threshold_char", 0.70))
            time_open.set(settings_data.get("time_open", 10000))
            threshold_plate_classify.set(settings_data.get("threshold_plate_classify", 0.50))
            allowed_plate_types.clear()

            # Восстанавливаем состояние чекбоксов
            for var, plate_type in zip(plate_vars, plate_classes):
                var.set(1 if plate_type in settings_data.get("allowed_plate_types", []) else 0)

            allow_all_var.set(settings_data.get("allow_all", False))
            errors_var.set(settings_data.get("allowed_errors", 0))

            print("Настройки успешно загружены")
    except FileNotFoundError:
        messagebox.showwarning("Файл настроек не найден", "Файл настроек не найден. Используются настройки по умолчанию.")
    except json.JSONDecodeError:
        messagebox.showerror("Ошибка загрузки", "Не удалось загрузить настройки. Проверьте формат файла settings.json.")


# Функция для сброса настроек на значения по умолчанию
def reset_settings_to_default():
    # Выводим сообщение с подтверждением
    if messagebox.askyesno("Сброс настроек", "Вы действительно хотите сбросить все настройки?"):
        top.set(38)
        bottom.set(60)
        left.set(10)
        right.set(60)
        alpha.set(1.5)
        beta.set(20)
        frame_skip_var.set(3)
        source_var.set("Camera")
        time_wait.set(3)
        threshold_vehicle.set(0.60)
        threshold_classify.set(0.50)
        threshold_np.set(0.40)
        threshold_char.set(0.70)
        threshold_plate_classify.set(0.50)
        allowed_plate_types.clear()
        allow_all_var.set(False)
        errors_var.set(0)
        time_open.set(10000)
        messagebox.showinfo("Сброс настроек", "Все настройки сброшены на значения по умолчанию.")


# Функция для установки начальных настроек при запуске программы
def initialize_settings():
    global initial_settings
    load_settings_from_json()  # Загружаем настройки из файла, если он существует
    initial_settings = get_current_settings()  # Сохраняем начальные настройки для отслеживания изменений
    print(f"Начальные настройки: {initial_settings}")


# Функция для настройки контрастности и яркости изображения
def adjust_contrast_brightness(image, alpha, beta):
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image


# Функция для детекции транспортных средств на изображении
def detect_vehicles(image):
    results = model_vehicle(image, conf=threshold_vehicle.get())
    vehicle_images = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            bbox = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].item()  # Получаем значение conf для этого bbox
            class_id = int(box.cls[0].item())
            if model_vehicle.names[class_id] in vehicle_classes:
                vehicle_images.append((bbox, conf))  # Добавляем кортеж bbox и conf
    return vehicle_images


# Функция для рисования прямоугольника вокруг обнаруженного объекта и отображения conf
def draw_bbox_on_image(image, bbox, conf, thickness=2):
    x1, y1, x2, y2 = bbox
    image_with_bbox = image.copy()

    # Рисуем рамку
    cv2.rectangle(image_with_bbox, (x1, y1), (x2, y2), (0, 255, 0), thickness)

    # Отображаем conf над рамкой
    conf_text = f"{conf:.2f}"  # Преобразуем conf в строку с двумя знаками после запятой
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_thickness = 3
    text_size, _ = cv2.getTextSize(conf_text, font, font_scale, font_thickness)

    # Располагаем текст над рамкой
    text_x = x1
    text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10  # Если место над bbox есть, рисуем выше, иначе ниже
    cv2.putText(image_with_bbox, conf_text, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness)

    return image_with_bbox


# Функция для классификации типа транспортного средства
def classify_vehicle(vehicle_image):
    results = model_classify(vehicle_image)  # Используем модель без conf, так как это классификация

    if results:
        for r in results:
            if r.probs is not None:
                probs = r.probs.data.cpu().numpy()
                class_id = probs.argmax()  # Определение класса с максимальной вероятностью

                # Проверяем, превышает ли вероятность порог уверенности
                if probs[class_id] >= threshold_classify.get() and 0 <= class_id < len(classification_classes):
                    print(
                        f"Классифицированный тип транспорта: {classification_classes[class_id]} (вероятность: {probs[class_id]:.2f})")
                    return classification_classes[class_id], probs[class_id]  # Возвращаем тип и вероятность
                else:
                    print(
                        f"Классификация не удалась: вероятность {probs[class_id]:.2f} ниже порога {threshold_classify.get():.2f}")
                    return "Классификация не удалась! Conf меньше порога!", probs[class_id]  # Если вероятность ниже порога, классификация не удалась
            else:
                print("Не удалось получить вероятности классов.")  # Диагностика
    else:
        print("Результаты классификации отсутствуют.")  # Диагностика

    return "Неизвестно", 0.0  # Возвращаем "Неизвестно" и вероятность 0.0 в случае ошибки


# Функция для распознавания номера на транспортном средстве
def detect_number_plate(vehicle_image):
    results = model_np(vehicle_image, conf=threshold_np.get())
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            bbox = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = bbox
            number_plate_image = vehicle_image[y1:y2, x1:x2]
            return number_plate_image
    return None


# Функция для распознавания символов на номере
def detect_characters(number_plate_image, plate_type):
    results = model_char(number_plate_image, conf=threshold_char.get())
    char_results = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            bbox = box.xyxy[0].cpu().numpy().astype(int)
            cls = int(box.cls[0].item())
            char_results.append((bbox, cls))

    if plate_type == 'square':
        if len(char_results) >= 8:
            top_row = sorted(char_results, key=lambda y: min(y[0][1], y[0][3]))[:4]
            bottom_row = sorted(char_results, key=lambda y: max(y[0][1], y[0][3]))[4:]

            top_row_sorted = sorted(top_row, key=lambda x: x[0][0])
            bottom_row_sorted = sorted(bottom_row, key=lambda x: x[0][0])

            top_text = ''.join([letters[char[1]] for char in top_row_sorted if char[1] < len(letters)])
            bottom_text = ''.join([letters[char[1]] for char in bottom_row_sorted if char[1] < len(letters)])

            return f"{top_text}{bottom_text}"
        else:
            return "Не корректно распознаны символы номера."
    else:
        char_results = sorted(char_results, key=lambda x: x[0][0])
        return ''.join([letters[char[1]] for char in char_results if char[1] < len(letters)])


# Функция для классификации номера
def classify_number_plate(number_plate_image):
    results = model_plate_classify(number_plate_image, conf=threshold_plate_classify.get())

    if results:
        for r in results:
            if r.probs is not None:
                probs = r.probs.data.cpu().numpy()
                class_id = probs.argmax()
                if 0 <= class_id < len(plate_classes):
                    print(
                        f"Распознанный тип номера: {plate_classes[class_id]} (вероятность: {probs[class_id]})")  # Диагностика
                    return plate_classes[class_id], probs[class_id]
            else:
                print("Не удалось получить вероятности классов.")  # Диагностика
    else:
        print("Результаты классификации отсутствуют.")  # Диагностика

    return "Неизвестно"


# Функция для рисования фиолетового прямоугольника для распознавания на изображении
def draw_violet_rectangle(image, top, bottom, left, right, thickness=2):
    image_with_rectangle = image.copy()
    cv2.rectangle(image_with_rectangle, (left, top), (right, bottom), (255, 0, 255), thickness)
    return image_with_rectangle


# Функция для загрузки видео из файла
def load_video():
    global video_capture, VIDEO_WIDTH, VIDEO_HEIGHT, is_video_loaded, is_paused
    is_paused = False
    if video_capture:
        video_capture.release()
    video_path = filedialog.askopenfilename(title="Выберите видеофайл",
                                            filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
    if video_path:
        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened():
            messagebox.showerror("Ошибка", "Не удалось открыть видеофайл!")
        else:
            is_video_loaded = True
            VIDEO_WIDTH = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            VIDEO_HEIGHT = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            adjust_video_size()
            process_frame()
    else:
        messagebox.showwarning("Внимание", "Видео не выбрано!")


# Функция для загрузки видеопотока с камеры
def load_camera(index=0):
    global video_capture, VIDEO_WIDTH, VIDEO_HEIGHT, is_video_loaded, is_paused
    is_paused = False
    if video_capture:
        video_capture.release()
    try:
        video_capture = cv2.VideoCapture(index)
        if not video_capture.isOpened():
            raise ValueError(f"Не удалось открыть камеру с индексом {index}")
        is_video_loaded = False
        VIDEO_WIDTH = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        VIDEO_HEIGHT = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        adjust_video_size()
        process_frame()
    except Exception as e:
        messagebox.showwarning("Предупреждение", str(e) + ". Пожалуйста, загрузите видео.")


# Функция для масштабирования видео в соответствии с размерами окна, сохраняя пропорции
def adjust_video_size():
    global VIDEO_WIDTH, VIDEO_HEIGHT, screen_width, screen_height

    frame_width = frame_video.winfo_width()
    frame_height = frame_video.winfo_height()

    aspect_ratio = VIDEO_WIDTH / VIDEO_HEIGHT

    if frame_width / frame_height > aspect_ratio:
        new_height = frame_height
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = frame_width
        new_height = int(new_width / aspect_ratio)

    VIDEO_WIDTH = new_width
    VIDEO_HEIGHT = new_height


# Функция для загрузки допустимых номеров из текстового файла
def load_allowed_numbers():
    global allowed_numbers
    file_path = filedialog.askopenfilename(title="Выберите txt файл с допустимыми номерами",
                                           filetypes=[("Text files", "*.txt")])
    if file_path:
        with open(file_path, "r", encoding="utf-8") as f:
            allowed_numbers = f.read().split(",")
            # Удаляем пустые строки из списка номеров
            allowed_numbers = [num.strip() for num in allowed_numbers if num.strip()]
    else:
        if allowed_numbers_list:
            with open(allowed_numbers_list, "r", encoding="utf-8") as f:
                allowed_numbers = f.read().split(",")
                # Удаляем пустые строки из списка номеров
                allowed_numbers = [num.strip() for num in allowed_numbers if num.strip()]




# Функция для обновления доступных камер
def update_camera_list():
    camera_list = []
    index = 0
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            break
        camera_list.append(f"Камера {index}")
        cap.release()
        index += 1
    if camera_list:
        camera_var.set(camera_list[0])
        camera_menu['values'] = camera_list
    else:
        camera_var.set('')
        messagebox.showwarning("Предупреждение",
                               "Не найдены доступные камеры. Пожалуйста, загрузите видео для обработки.")


# Функция для изменения размера изображения номерного знака
def resize_plate_image(plate_image, target_width, target_height):
    return cv2.resize(plate_image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


# Функция для сравнения номеров с допустимыми ошибками в символах
def is_allowed_number(detected_number, allowed_numbers, allowed_errors):
    if not allowed_numbers:  # Если список номеров пустой
        print("Список допустимых номеров пуст. Запрет доступа.")
        return False  # Запрещаем доступ, если список номеров пуст

    for allowed_number in allowed_numbers:
        similarity = SequenceMatcher(None, detected_number, allowed_number).ratio()
        print(f"Сравнение: {detected_number} с {allowed_number} | Сходство: {similarity:.2f}")

        # Вычисляем максимальное количество допустимых ошибок
        max_errors = allowed_errors
        actual_errors = int(len(allowed_number) * (1 - similarity))

        if abs(len(detected_number) - len(allowed_number)) == 1:
            actual_errors += 1
            print("Несоответствие длины: добавлена 1 ошибка из-за отсутствующего символа")

        print(f"Фактические ошибки: {actual_errors}, Допустимые ошибки: {max_errors}")

        if actual_errors <= max_errors:
            return True

    return False  # Если ни один номер не подошел, запрещаем доступ


# Функция для открытия окна настроек доступа
def open_settings_window():
    global allowed_numbers, allowed_plate_types, allow_all, allowed_errors

    settings_window = tk.Toplevel(root)
    settings_window.title("Настройки доступа на парковку")
    settings_window.geometry("400x450")

    def load_txt_file():
        global allowed_numbers
        file_path = filedialog.askopenfilename(title="Настройки доступа на парковку",
                                               filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, "r", encoding="utf-8") as f:
                allowed_numbers = f.read().split(",")
                allowed_numbers = [num.strip() for num in allowed_numbers]
            settings_window.lift()
            settings_window.focus_force()
        else:
            allowed_numbers = None

    # Обновляем глобальную переменную при изменении значения ползунка
    def update_allowed_errors():
        global allowed_errors
        allowed_errors = errors_var.get()

    # Обновляем глобальную переменную при изменении чекбокса
    def update_allow_all():
        global allow_all
        allow_all = allow_all_var.get()

    # Обновляем глобальную переменную для allowed_plate_types при изменении чекбоксов
    def update_allowed_plate_types():
        global allowed_plate_types
        allowed_plate_types = {plate_type for var, plate_type in zip(plate_vars, plate_classes) if var.get()}

    lbl_file_path = tk.Label(settings_window, text="Загрузите txt файл с номерами", wraplength=380, justify="left")
    lbl_file_path.pack(pady=10)

    btn_load_txt = tk.Button(settings_window, text="Загрузить файл", command=load_txt_file, font=("Helvetica", 12),
                             bg="#3498db", fg="white", activebackground="#2980b9", activeforeground="white", bd=0,
                             padx=5, pady=5)
    btn_load_txt.pack(pady=5)

    tk.Label(settings_window, text="Выберите типы номерных знаков", font=("Helvetica", 12)).pack(anchor="w", pady=(10, 0), padx=5)

    # Используем уже созданные plate_vars, добавляя command для немедленного обновления
    for var, plate_type in zip(plate_vars, plate_classes):
        chk = tk.Checkbutton(settings_window, text=plate_type, variable=var, command=update_allowed_plate_types)
        chk.pack(anchor="w", padx=20)

    allow_all_var = tk.IntVar(value=allow_all)
    chk_allow_all = tk.Checkbutton(settings_window, text="Разрешить доступ для всех", variable=allow_all_var, command=update_allow_all)
    chk_allow_all.pack(anchor="w", pady=(10, 0), padx=5)

    tk.Label(settings_window, text="Допустимое количество ошибок в номере", font=("Helvetica", 12)).pack(anchor="w", pady=(10, 0), padx=5)

    # Ползунок с обновлением значения в глобальной переменной
    errors_scale = tk.Scale(settings_window, from_=0, to=3, orient=tk.HORIZONTAL, variable=errors_var, length=280, command=lambda v: update_allowed_errors())
    errors_scale.pack(padx=5, pady=5)


# Функция для открытия окна с настройками порогов
def open_threshold_settings_window():
    threshold_window = tk.Toplevel(root)
    threshold_window.title("Настройки порогов")
    threshold_window.geometry("400x400")

    tk.Label(threshold_window, text="Порог детекции транспортных средств", font=("Helvetica", 10)).pack(anchor="w", pady=(5, 0), padx=5)
    tk.Scale(threshold_window, from_=0.0, to=1.0, orient=tk.HORIZONTAL, resolution=0.01, variable=threshold_vehicle, length=280).pack(padx=5, fill=tk.X)

    tk.Label(threshold_window, text="Порог классификации транспортных средств", font=("Helvetica", 10)).pack(anchor="w", pady=(5, 0), padx=5)
    tk.Scale(threshold_window, from_=0.0, to=1.0, orient=tk.HORIZONTAL, resolution=0.01, variable=threshold_classify, length=280).pack(padx=5, fill=tk.X)

    tk.Label(threshold_window, text="Порог детекции номеров", font=("Helvetica", 10)).pack(anchor="w", pady=(5, 0), padx=5)
    tk.Scale(threshold_window, from_=0.0, to=1.0, orient=tk.HORIZONTAL, resolution=0.01, variable=threshold_np, length=280).pack(padx=5, fill=tk.X)

    tk.Label(threshold_window, text="Порог детекции символов", font=("Helvetica", 10)).pack(anchor="w", pady=(5, 0), padx=5)
    tk.Scale(threshold_window, from_=0.0, to=1.0, orient=tk.HORIZONTAL, resolution=0.01, variable=threshold_char, length=280).pack(padx=5, fill=tk.X)

    tk.Label(threshold_window, text="Порог классификации номеров", font=("Helvetica", 10)).pack(anchor="w", pady=(5, 0), padx=5)
    tk.Scale(threshold_window, from_=0.0, to=1.0, orient=tk.HORIZONTAL, resolution=0.01, variable=threshold_plate_classify, length=280).pack(padx=5, fill=tk.X)


# Функция для открытия шлагбаума на 10 секунд
def open_gate(text="Проезд разрешён! Шлагбаум открыт.", fg="green"):
    global gate_timer_id

    # Если уже запущен таймер закрытия, отменяем его
    if gate_timer_id is not None:
        root.after_cancel(gate_timer_id)  # Отменяем предыдущий таймер

    # Обновляем текст и цвет для статуса открытого шлагбаума
    lbl_access.config(text=text, fg=fg)

    # Запускаем новый таймер на для закрытия шлагбаума
    gate_timer_id = root.after(time_open.get(), close_gate)


# Функция для закрытия шлагбаума
def close_gate(text="Шлагбаум закрыт", fg="red"):
    global gate_timer_id

    # Закрываем шлагбаум, обновляем текст и цвет
    lbl_access.config(text=text, fg=fg)

    # Сбрасываем идентификатор таймера, так как шлагбаум закрыт
    gate_timer_id = None

# Основная функция для обработки каждого кадра видео
def process_frame():
    global video_capture, allowed_numbers, allowed_vehicle_types, allowed_plate_types
    global frame_skip_count, VIDEO_WIDTH, VIDEO_HEIGHT, lbl_plate_image, is_paused, allow_all

    # Определяем допустимые типы номеров на основе текущего состояния чекбоксов
    allowed_plate_types = {plate_type for var, plate_type in zip(plate_vars, plate_classes) if var.get()}

    if video_capture is None or not video_capture.isOpened():  # Проверяем, загружено ли видео или камера открыта
        print("Видео или камера не найдены. Ожидание следующего действия...")
        return  # Если видео или камера не загружены, выходим из функции

    if is_paused:  # Проверяем, стоит ли видео на паузе
        root.after(100, process_frame)  # Если на паузе, через 100 мс вызываем функцию снова
        return  # Выходим из функции, чтобы не продолжать обработку

    ret, frame = video_capture.read()  # Читаем текущий кадр из видео или камеры
    if not ret:  # Если не удалось получить кадр
        print("Кадр не получен. Ожидание следующего кадра...")
        root.after(1000, process_frame)  # Пытаемся снова через 1000 мс
        return  # Выходим из функции

    frame_skip = frame_skip_var.get()  # Получаем значение пропуска кадров из настроек

    if frame_skip > 0 and frame_skip_count % frame_skip != 0:  # Если нужно пропустить кадр
        frame_skip_count += 1  # Увеличиваем счётчик пропуска кадров
        root.after(int(time_wait.get()), process_frame)  # Переходим к следующему кадру через time_wait мс
        return  # Выходим из функции

    frame_skip_count += 1  # Увеличиваем счётчик обработанных кадров

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Преобразуем кадр из BGR в RGB для обработки
    frame_with_bbox = frame_rgb  # Создаем копию кадра для дальнейшего отображения

    # Переводим границы в пиксели
    top_value = int(top.get() * frame.shape[0] / 100)
    bottom_value = int(bottom.get() * frame.shape[0] / 100)
    left_value = int(left.get() * frame.shape[1] / 100)
    right_value = int(right.get() * frame.shape[1] / 100)

    # Проверяем корректность границ зоны
    if top_value < bottom_value and left_value < right_value:
        vehicle_images = detect_vehicles(frame_rgb)  # Детектируем транспортные средства на кадре

        if vehicle_images:  # Если обнаружены транспортные средства
            max_y2 = -1  # Инициализируем максимальное значение y2
            best_bbox = None  # Инициализируем лучшую область
            best_conf = 0  # Для хранения лучшего conf

            for bbox, conf in vehicle_images:  # Проходим по всем обнаруженным транспортным средствам
                x1, y1, x2, y2 = bbox  # Получаем координаты текущей области

                if y2 <= bottom_value and y2 < frame.shape[0]:  # Проверяем, не выходит ли область за нижнюю границу
                    intersects = (
                        (x1 < right_value and x2 > left_value and y1 <= bottom_value and y2 >= top_value)
                        or (y1 < bottom_value and y2 > top_value and x1 <= right_value and x2 >= left_value)
                    )

                    if intersects and y2 > max_y2:  # Если область пересекается и ниже текущей максимальной границы
                        max_y2 = y2  # Обновляем максимальное значение y2
                        best_bbox = bbox  # Запоминаем этот транспорт
                        best_conf = conf  # Запоминаем лучшее значение conf

            if best_bbox is not None:  # Если найден подходящий транспорт
                x1, y1, x2, y2 = best_bbox  # Получаем координаты транспорта

                # Проверяем флаг "доступ для всех"
                if allow_all==True:
                    open_gate(text="Проезд разрешён всем!", fg="green")
                else:
                    # Вырезаем изображение транспортного средства
                    vehicle_image = frame_rgb[y1:y2, x1:x2]
                    vehicle_image = adjust_contrast_brightness(vehicle_image, alpha.get(), beta.get())  # Корректируем яркость и контрастность
                    vehicle_class, conf_classv = classify_vehicle(vehicle_image)  # Классифицируем тип транспортного средства
                    lbl_classification.config(text=f"Классифицирован транспорт: {vehicle_class} (conf: {conf_classv:.2f})")

                    # Если транспорт — спецслужбы, разрешаем доступ
                    if vehicle_class in ["police", "ambulance", "firetrucks"]:
                        lbl_plate_class.config(text="Классификация номера: ")
                        lbl_number.config(text="Распознанный номер: ")
                        lbl_plate_image.configure(image=None)
                        lbl_plate_image.image = None  # Удаляем ссылку на изображение
                        open_gate(text="Проезд спецслужбам разрешён!", fg="blue")

                    # Если обычный автомобиль, проверяем номер
                    elif vehicle_class == "normal_car":
                        number_plate_image = detect_number_plate(vehicle_image)
                        if number_plate_image is not None:
                            resized_plate_image = resize_plate_image(number_plate_image, PLATE_WIDTH, PLATE_HEIGHT)
                            plate_image_pil = Image.fromarray(resized_plate_image)
                            plate_image_tk = ImageTk.PhotoImage(plate_image_pil)
                            lbl_plate_image.configure(image=plate_image_tk)
                            lbl_plate_image.image = plate_image_tk  # Отображаем номер

                            plate_class, conf_cnp = classify_number_plate(number_plate_image)
                            lbl_plate_class.config(text=f"Классификация номера: {plate_class} (conf: {conf_cnp:.2f})")

                            # Проверяем тип номера
                            if plate_class in allowed_plate_types:
                                if plate_class in ["normal", "square"]:
                                    number_plate = detect_characters(number_plate_image, plate_type=plate_class)

                                    if number_plate and 10 >=len(number_plate) >=6  :
                                        lbl_number.config(text=f"Распознанный номер: {number_plate}")

                                        # Проверяем наличие номера в списке
                                        if is_allowed_number(number_plate, allowed_numbers, errors_var.get()):
                                            open_gate(text="Проезд разрешён! Номер в списке!", fg="green")
                                        else:

                                            close_gate(text="Проезд запрещён! Нет в списке!", fg="red")
                                    else:
                                        lbl_number.config(text="Распознанный номер: Не удалось распознать символы")
                                        close_gate(text="Проезд запрещён! Номер не читается!", fg="red")
                                else:
                                    open_gate(text="Проезд разрешён! Такие номера пропускаем!", fg="green")
                            else:
                                close_gate(text="Проезд запрещён! Номера такого типа запрещены!", fg="red")
                        else:
                            lbl_number.config(text="Распознанный номер: Не удалось найти номер.")
                            lbl_plate_class.config(text="Классификация номера: ")
                            lbl_plate_image.configure(image=None)
                            lbl_plate_image.image = None  # Удаляем ссылку на изображение
                            close_gate(text="Проезд запрещён!", fg="red")
                    else:
                          # Если это не обычный автомобиль и не спецтранспорт, доступ запрещен
                        close_gate(text="Проезд запрещён!", fg="red")

                # Отрисовываем рамку с конфидентом вокруг лучшего обнаруженного транспортного средства
                frame_with_bbox = draw_bbox_on_image(frame_rgb, best_bbox, best_conf, thickness=2)
        else:
            frame_with_bbox = frame_rgb  # Используем оригинальный кадр без изменений

        # Рисуем фиолетовый прямоугольник вокруг области интереса
        frame_with_rectangle = draw_violet_rectangle(frame_with_bbox, top_value, bottom_value, left_value, right_value)
    else:
        print("Некорректные границы зоны детекции транспорта.")
        frame_with_rectangle = frame_rgb  # Используем оригинальный кадр без изменений

    # Изменяем размер кадра до размеров видео
    frame_resized = cv2.resize(frame_with_rectangle, (VIDEO_WIDTH, VIDEO_HEIGHT))
    frame_pil = Image.fromarray(frame_resized)  # Преобразуем кадр в объект PIL для отображения в интерфейсе
    frame_tk = ImageTk.PhotoImage(frame_pil)  # Преобразуем изображение PIL в формат, поддерживаемый Tkinter
    lbl_frame.configure(image=frame_tk)  # Обновляем изображение в интерфейсе
    lbl_frame.image = frame_tk  # Сохраняем ссылку на изображение, чтобы избежать его удаления сборщиком мусора

    # Повторно вызываем функцию через time_wait для обработки следующего кадра
    root.after(int(time_wait.get()), process_frame)


# Функция для начала обработки видео или камеры
def start_processing(source="camera"):
    global video_capture, allowed_numbers, frame_skip_count, VIDEO_WIDTH, VIDEO_HEIGHT, is_paused
    frame_skip_count = 0
    is_paused = False

    if source == "camera":
        camera_index = int(camera_var.get().split()[-1]) if camera_var.get() else 0
        load_camera(camera_index)
    elif source == "video":
        load_video()


# Основная функция для закрытия программы
def on_closing():
    global initial_settings
    current_settings = get_current_settings()
    print(f"Текущие настройки: {current_settings}")  # Отладка

    # Сравниваем текущие настройки с первоначальными
    if current_settings != initial_settings:
        if messagebox.askyesnocancel("Сохранение настроек", "Настройки были изменены. Хотите их сохранить перед выходом?"):
            save_settings_to_json()
        elif messagebox.askyesnocancel("Сброс настроек", "Вы уверены, что хотите выйти без сохранения настроек?"):
            pass  # Продолжаем закрытие программы без сохранения
        else:
            return  # Отменить закрытие программы
    root.destroy()  # Закрываем окно


# Функция для обработки нажатия пробела
def toggle_pause(event):
    global is_paused, video_capture
    is_paused = not is_paused
    if is_paused and not is_video_loaded:
        video_capture.release()
    elif not is_paused and not is_video_loaded:
        load_camera(int(camera_var.get().split()[-1]) if camera_var.get() else 0)


# Инициализация настроек при запуске
initialize_settings()  # Загружаем настройки из файла или применяем значения по умолчанию
root.protocol("WM_DELETE_WINDOW", on_closing)  # Устанавливаем обработчик закрытия окна

root.configure(bg="#2c3e50")  # Устанавливаем цвет фона главного окна

paned_window_horizontal = tk.PanedWindow(root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, bg="#2c3e50")  # Создаем панель, которая делит окно на две части
paned_window_horizontal.pack(fill=tk.BOTH, expand=True)  # Заполняем окно панелью

frame_controls = tk.Frame(paned_window_horizontal, bg="#f0f0f0")  # Создаем фрейм для элементов управления (левая часть)
frame_controls.pack_propagate(True)  # Разрешаем изменять размер фрейма в зависимости от содержимого

left_canvas = tk.Canvas(frame_controls)  # Создаем холст для добавления прокручиваемого содержимого
left_scrollbar = tk.Scrollbar(frame_controls, orient="vertical", command=left_canvas.yview)  # Создаем вертикальный скроллбар и связываем его с холстом
left_canvas.configure(yscrollcommand=left_scrollbar.set)  # Настраиваем холст так, чтобы скроллбар отображал изменения

scrollable_frame = tk.Frame(left_canvas)  # Создаем фрейм внутри холста для прокручиваемого содержимого
left_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")  # Создаем окно внутри холста для размещения фрейма
scrollable_frame.bind("<Configure>", lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all")))  # Настраиваем область прокрутки по размеру содержимого

left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)  # Размещаем холст слева и заполняем все пространство
left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)  # Размещаем скроллбар справа

paned_window_horizontal.add(frame_controls, minsize=1)  # Добавляем левую часть интерфейса в основную панель

right_frame = tk.Frame(paned_window_horizontal, bg="black")  # Создаем фрейм для правой части
right_frame.pack_propagate(True)  # Разрешаем изменять размер фрейма

paned_window_horizontal.add(right_frame, minsize=280)  # Добавляем правую часть интерфейса в основную панель

paned_window_vertical = tk.PanedWindow(right_frame, orient=tk.VERTICAL, sashrelief=tk.RAISED, bg="#2c3e50")  # Создаем панель, которая делит правую часть на две вертикальные части
paned_window_vertical.pack(fill=tk.BOTH, expand=True)  # Заполняем правую часть вертикальной панелью

frame_video = tk.Frame(paned_window_vertical, bg="black")  # Создаем фрейм для отображения видео
frame_video.pack_propagate(False)  # Запрещаем изменять размер фрейма

lbl_frame = tk.Label(frame_video, bg="black")  # Создаем метку для отображения видео
lbl_frame.pack(fill=tk.BOTH, expand=True)  # Размещаем метку и заполняем фрейм

# Создаем фрейм для текста и меток
frame_text = tk.Frame(paned_window_vertical, bg="#f0f0f0")  # Создаем фрейм для текстовых элементов и меток
frame_text.pack_propagate(False)  # Запрещаем изменять размер фрейма

frame_right = tk.Frame(frame_text, width=500, height=100, bg="#f0f0f0")  # Создаем правую часть фрейма для изображения номерного знака
frame_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)  # Размещаем его справа и заполняем пространство

lbl_plate_image = tk.Label(frame_right, bg="#f0f0f0")  # Создаем метку для отображения изображения номерного знака
lbl_plate_image.pack(anchor="e", pady=10, padx=10)  # Размещаем метку с отступами

# Создаем левую часть фрейма для текстовых меток
frame_left = tk.Frame(frame_text, width=600, height=100, bg="#f0f0f0")  # Создаем левую часть фрейма для текстового вывода
frame_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)  # Размещаем его слева и заполняем пространство

lbl_access = tk.Label(frame_left, text="Проезд: ", font=("Arial", 26), wraplength=1000, justify="left")  # Создаем текст для отображения доступа
lbl_access.pack(anchor="w", pady=1)  # Размещаем текст с небольшими отступами

lbl_number = tk.Label(frame_left, text="Распознанный номер: ", font=("Arial", 16), wraplength=800, justify="left")  # Создаем текст для отображения распознанного номера
lbl_number.pack(anchor="w", pady=1)  # Размещаем текст с небольшими отступами

lbl_classification = tk.Label(frame_left, text="Классификация транспорта: ", font=("Arial", 13), wraplength=800, justify="left")  # текст для отображения типа транспорта
lbl_classification.pack(anchor="w", pady=1)  # Размещаем текст с небольшими отступами

lbl_plate_class = tk.Label(frame_left, text="Классификация номера: ", font=("Arial", 13), wraplength=800, justify="left")  # текст для отображения типа номерного знака
lbl_plate_class.pack(anchor="w", pady=1)  # Размещаем текст с небольшими отступами

# Добавляем видео и текстовые блоки в правую вертикальную панель
paned_window_vertical.add(frame_video, stretch="always")  # Видео будет растягиваться и занимать больше пространства
paned_window_vertical.add(frame_text, stretch="never")  # Текстовые элементы не будут растягиваться

paned_window_vertical.add(frame_video, minsize=500)  # Минимальный размер фрейма для видео
paned_window_vertical.add(frame_text, minsize=150)  # Минимальный размер фрейма для текста

# Добавляем элементы управления на левую панель
tk.Label(scrollable_frame, text="Источник видео").pack()  # Метка для источника видео

camera_var = tk.StringVar()  # Переменная для хранения выбранной камеры
camera_menu = ttk.Combobox(scrollable_frame, textvariable=camera_var, state="readonly", width=28)  # Выпадающий список для выбора камеры
camera_menu.pack(anchor=tk.W, padx=[5, 5], pady=(5, 0))  # Размещаем выпадающий список с отступами
update_camera_list()  # Обновляем список доступных камер

# Кнопка для включения камеры
btn_load_camera = tk.Button(scrollable_frame, text="Включить камеру", command=lambda: start_processing(source="camera"), font=("Helvetica", 12), bg="#3498db", fg="white", activebackground="#2980b9", activeforeground="white", bd=0, padx=5, pady=5, width=30)
btn_load_camera.pack(pady=5, padx=[5, 5])  # Размещаем кнопку с отступами

# Кнопка для настройки доступа на парковку
btn_load_numbers = tk.Button(scrollable_frame, text="Настройки доступа на парковку", command=open_settings_window, font=("Helvetica", 12), bg="#FF8C00", fg="white", activebackground="#2980b9", activeforeground="white", bd=0, padx=5, pady=5, width=30)
btn_load_numbers.pack(pady=5, padx=[5, 5])  # Размещаем кнопку с отступами

# Кнопка для открытия настроек порогов нейросетей
btn_threshold_settings = tk.Button(scrollable_frame, text="Настройки порогов нейросетей", command=open_threshold_settings_window, font=("Helvetica", 12), bg="#FF8C00", fg="white", activebackground="#2980b9", activeforeground="white", bd=0, padx=5, pady=5, width=30)
btn_threshold_settings.pack(pady=5, padx=[5, 5])  # Размещаем кнопку с отступами

# Кнопка для загрузки видео
btn_load_video = tk.Button(scrollable_frame, text="Загрузить видео", command=lambda: start_processing(source="video"), font=("Helvetica", 12), bg="#3498db", fg="white", activebackground="#2980b9", activeforeground="white", bd=0, padx=5, pady=5, width=30)
btn_load_video.pack(pady=5, padx=[5, 5])  # Размещаем кнопку с отступами

# Кнопка для сброса настроек
btn_reset_settings = tk.Button(scrollable_frame, text="Сброс настроек", command=reset_settings_to_default, font=("Helvetica", 12), bg="#e74c3c", fg="white", activebackground="#c0392b", activeforeground="white", bd=0, padx=5, pady=5, width=30)
btn_reset_settings.pack(pady=5, padx=[5, 5])  # Размещаем кнопку с отступами


separator = tk.Frame(scrollable_frame, height=2, bd=1, relief=tk.SUNKEN)  # Разделительная линия между элементами
separator.pack(fill=tk.X, pady=5)  # Размещаем разделитель по горизонтали

# Слайдеры для настройки границ зоны детекции
tk.Label(scrollable_frame, text="Верхняя граница (%)", font=("Helvetica", 10)).pack(anchor="w", pady=(5, 0), padx=5)  # Метка для верхней границы
tk.Scale(scrollable_frame, from_=0, to=100, orient=tk.HORIZONTAL, variable=top, length=280).pack(padx=5, fill=tk.X)  # Слайдер для настройки верхней границы

tk.Label(scrollable_frame, text="Нижняя граница (%)", font=("Helvetica", 10)).pack(anchor="w", pady=(5, 0), padx=5)  # Метка для нижней границы
tk.Scale(scrollable_frame, from_=0, to=100, orient=tk.HORIZONTAL, variable=bottom, length=280).pack(padx=5, fill=tk.X)  # Слайдер для настройки нижней границы

tk.Label(scrollable_frame, text="Левая граница (%)", font=("Helvetica", 10)).pack(anchor="w", pady=(5, 0), padx=5)  # Метка для левой границы
tk.Scale(scrollable_frame, from_=0, to=100, orient=tk.HORIZONTAL, variable=left, length=280).pack(padx=5, fill=tk.X)  # Слайдер для настройки левой границы

tk.Label(scrollable_frame, text="Правая граница (%)", font=("Helvetica", 10)).pack(anchor="w", pady=(5, 0), padx=5)  # Метка для правой границы
tk.Scale(scrollable_frame, from_=0, to=100, orient=tk.HORIZONTAL, variable=right, length=280).pack(padx=5, fill=tk.X)  # Слайдер для настройки правой границы

# Слайдеры для пропуска кадров и времени пропуска
tk.Label(scrollable_frame, text="Пропуск кадров", font=("Helvetica", 10)).pack(anchor="w", pady=(5, 0), padx=5)  # Метка для пропуска кадров
tk.Scale(scrollable_frame, from_=0, to=25, orient=tk.HORIZONTAL, variable=frame_skip_var, length=280).pack(padx=5, fill=tk.X)  # Слайдер для пропуска кадров

tk.Label(scrollable_frame, text="Регулировка времени пропуска кадров", font=("Helvetica", 10)).pack(anchor="w", pady=(5, 0), padx=5)  # Метка для времени пропуска
tk.Scale(scrollable_frame, from_=1, to=50, orient=tk.HORIZONTAL, variable=time_wait, length=280).pack(padx=5, fill=tk.X)  # Слайдер для настройки времени пропуска

tk.Label(scrollable_frame, text="Регулировка времени закрытия шлагбаума, мс", font=("Helvetica", 10)).pack(anchor="w", pady=(5, 0), padx=5)  # Метка для времени закрытия шлагбаума
tk.Scale(scrollable_frame, from_=1, to=30000, orient=tk.HORIZONTAL, variable=time_open, length=280).pack(padx=5, fill=tk.X)  # Слайдер для настройки времени закрытия

# Слайдеры для контрастности и яркости изображения номера
tk.Label(scrollable_frame, text="Контрастность номера", font=("Helvetica", 10)).pack(anchor="w", pady=(5, 0), padx=5)  # Метка для контрастности
tk.Scale(scrollable_frame, from_=0.5, to=3.0, orient=tk.HORIZONTAL, resolution=0.1, variable=alpha, length=280).pack(padx=5, fill=tk.X)  # Слайдер для настройки контрастности

tk.Label(scrollable_frame, text="Яркость номера", font=("Helvetica", 10)).pack(anchor="w", pady=(5, 0), padx=5)  # Метка для яркости
tk.Scale(scrollable_frame, from_=-100, to=100, orient=tk.HORIZONTAL, variable=beta, length=280).pack(padx=5, fill=tk.X)  # Слайдер для настройки яркости

# Настройки главного окна
root.state('zoomed')  # Устанавливаем окно в развернутый на весь экран режим
root.geometry(f"{screen_width}x{screen_height}")  # Устанавливаем размеры окна по размеру экрана

# Функция для выхода из полноэкранного режима
def exit_fullscreen(event=None):
    root.attributes('-fullscreen', False)  # Выключаем полноэкранный режим

root.bind("<Escape>", exit_fullscreen)  # Привязываем клавишу "Escape" для выхода из полноэкранного режима
root.bind("<space>", toggle_pause)  # Привязываем клавишу "Space" для паузы/продолжения

root.mainloop()  # Запускаем главный цикл обработки событий Tkinter

