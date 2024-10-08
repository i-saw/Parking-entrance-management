# Управление шлагбаумом парковки

![image](https://github.com/i-saw/Parking-entrance-management/blob/main/GIF.gif)
![image](https://github.com/i-saw/Parking-entrance-management/blob/main/22.png)

# Разработка приложения "Автоматизация шлагбаума для доступа на паровку"

## Введение

В рамках проекта были созданы датасеты, обучены модели, разработана программа для распознавания транспортных средств, классификации их типов и распознавания номерных знаков с последующей проверкой на доступ к парковке. Программа была реализована с использованием компьютерного зрения и моделей YOLO для решения задач детекции, классификации и распознавания символов. Она предоставляет графический интерфейс, написанный с использованием библиотеки `Tkinter`, позволяющий пользователю управлять настройками и параметрами работы приложения.

## Основная цель проекта

Основной целью разработки программы было создание системы автоматического контроля доступа на парковку, которая могла бы:

1. Распознавать транспортные средства на видеопотоке или видеофайле.
2. Определять тип транспортного средства (спецслужбы или обычные автомобили).
3. Распознавать и классифицировать тип номерного знака.
4. Сравнивать распознанные номера с допустимыми номерами и предоставлять доступ в случае совпадения.

## Основные функциональные возможности

1. **Детекция транспортных средств**: Используется модель YOLO для обнаружения автомобилей, грузовиков и автобусов в видеопотоке.
2. **Классификация типов транспортных средств**: Используются модели для классификации транспортных средств по категориям, таким как "ambulance", "firetruck", "police", "normal_car".
3. **Распознавание и классификация номерных знаков**: Специальная модель YOLO используется для распознавания символов на номерных знаках, а также для классификации типов номеров (например, обычные, квадратные, желтые и т.д.).
4. **Проверка на доступ к парковке**: Программа сравнивает распознанные номера с допустимыми номерами из заранее загруженного списка и предоставляет доступ в случае совпадения. Допустимо также задавать количество ошибок в распознавании номера.
5. **Графический интерфейс**: Программа имеет интерфейс для управления настройками, такими как пороговые значения детекции, выбор камеры или видеофайла, настройка границ обработки и управление пропуском кадров.
6. **Сохранение и загрузка настроек**: Пользователь может сохранять настройки в JSON-файл и загружать их при повторном запуске программы.

## Используемые технологии

- **OpenCV**: Библиотека для работы с изображениями и видео, используется для обработки видеопотока, изменения контраста и яркости.
- **YOLO**: Модель для обнаружения объектов и классификации транспортных средств и номеров.
- **PIL (Pillow)**: Используется для работы с изображениями в формате, совместимом с интерфейсом `Tkinter`.
- **Tkinter**: Библиотека для создания графического интерфейса, включая элементы управления, такие как кнопки, чекбоксы, ползунки и текстовые поля.
- **Difflib**: Библиотека для сравнения строк, используется для вычисления допустимых ошибок при проверке номеров.

## Архитектура программы

Программа построена на основе событийно-ориентированной архитектуры, где основное окно приложения управляет всеми действиями пользователя. Основные компоненты включают:

- **Модуль детекции и классификации**: Отвечает за обработку видео и распознавание объектов.
- **Модуль настроек**: Позволяет сохранять и загружать параметры программы.
- **Интерфейс**: Взаимодействует с пользователем через графические элементы и предоставляет визуальные результаты.


## Ссылки для скачивания программы:
[Для CPU (займет около 650 Mb)](https://drive.google.com/drive/folders/1nTGw1Fqpv2WYll4p8UFNWZ9Djkig3S10?usp=drive_link) 

[CPU+GPU (займет около 4,4 Gb)](https://drive.google.com/drive/folders/1ZnXQ8oHEhfl4R9WDozDG6lXb9BpY9MTD?usp=drive_link)

