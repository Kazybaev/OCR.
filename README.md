main 

# Multilingual OCR Ensemble

Распознавание текста и таблиц из изображений на трёх языках: английский, русский, китайский.

## Системные требования

- **ОС:** Windows
- **Python:** 3.10
- **Tesseract OCR:** обязательна установка отдельно

## Что делает

- Загружаете изображение → получаете весь текст сразу от трёх OCR-движков
- Автоматически находит и извлекает таблицы
- Объединяет результаты, убирает дубликаты
- Скачиваете текст (.txt) и таблицы (.csv)

## Технологии

**OCR-движки:**
- **PaddleOCR** — основной движок (английский + китайский)
- **Tesseract** — резервный движок (все три языка)
- **EasyOCR** — дополнительный движок (русский + китайский)

**Таблицы:**
- **PPStructure** (PaddleOCR) — находит структуру таблицы
- **Tesseract + EasyOCR** — извлекают текст из каждой ячейки

**Интерфейс:**
- **Streamlit** — веб-интерфейс

**Обработка:**
- **OpenCV** — предобработка изображений (масштабирование, бинаризация)
- **Pandas** — экспорт таблиц

## Установка

### 1. Установка Tesseract OCR

**Windows:** 
- Скачайте установщик: [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
- При установке выберите языки: **English**, **Russian**, **Chinese - Simplified**
- Стандартный путь установки: `C:\Program Files\Tesseract-OCR\tesseract.exe`

### 2. Установка Python-зависимостей
```bash
pip install -r req.txt
```

### req.txt
```text
streamlit==1.51.0
paddleocr==2.6.1.3
paddlepaddle==3.0.0b1
easyocr==1.7.1
opencv-python==4.6.0.66
opencv-contrib-python==4.6.0.66
pandas==2.3.3
pytesseract==0.3.13
numpy==1.26.4
pillow==12.0.0
torch==2.5.1
torchvision==0.20.1
Shapely==1.8.5.post1
pyclipper==1.3.0.post6
lmdb==1.7.5
tqdm==4.67.1
imgaug==0.4.0
scikit-image==0.25.2
scipy==1.15.3
PyYAML==6.0.3
requests==2.32.5
python-docx==1.2.0
openpyxl==3.1.5
lxml==6.0.2
Cython==3.2.0
```

### 3. Настройка пути к Tesseract

Откройте `main.py` и укажите путь к Tesseract (строка 23):
```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

Если Tesseract установлен в другую папку, измените путь соответственно.

## Запуск
```bash
streamlit run main.py
```

Откроется браузер → загрузите изображение → нажмите "Распознать".

## Как работает

1. Все три OCR читают изображение параллельно
2. Результаты нормализуются и объединяются (дубликаты удаляются)
3. Для таблиц: PPStructure находит ячейки → каждая ячейка распознаётся отдельно
4. Готовые данные: текст блоками, таблицы в DataFrame

## Особенности

- Первый запуск долгий (модели EasyOCR загружаются автоматически ~100 MB)
- Работает на CPU (GPU не требуется)
- При недоступности движка — работает с оставшимися
- Все модели кэшируются для быстрой повторной загрузки
