# main.py — OCR с посимвольным сравнением и подсветкой различий
# PaddleOCR + Tesseract + EasyOCR (EN + RU + ZH)

import streamlit as st
import os
import tempfile
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import re
import pytesseract
from typing import List, Tuple, Dict
from difflib import SequenceMatcher
import time

# ========= EasyOCR =========
try:
    import easyocr

    EASYOCR_OK = True
except:
    EASYOCR_OK = False

# ========= Tesseract =========
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ========= PaddleOCR =========
try:
    from paddleocr import PaddleOCR, PPStructure

    PADDLE_OK = True
except Exception as e:
    PADDLE_OK = False
    st.error(f"❌ PaddleOCR ошибка: {e}")

# ========= UI =========
st.set_page_config(page_title="OCR Comparator", layout="wide")
st.title("📸 OCR Сравнение с подсветкой различий")
st.caption("Посимвольное сравнение результатов разных движков")

# ========= Проверки =========
TESSERACT_OK = True
try:
    ver = pytesseract.get_tesseract_version()
    st.success(f"✅ Tesseract: {ver}")
except:
    TESSERACT_OK = False
    st.error("❌ Tesseract не найден")

if EASYOCR_OK:
    st.success("✅ EasyOCR подключён")


# ========= Загрузка моделей =========
@st.cache_resource
def load_paddle():
    if not PADDLE_OK:
        return None, None, None
    ocr_en = PaddleOCR(
        lang='en',
        use_angle_cls=True,
        use_gpu=False,
        show_log=False,
        det_db_box_thresh=0.3,
        rec_batch_num=6
    )
    ocr_ch = PaddleOCR(
        lang='ch',
        use_angle_cls=True,
        use_gpu=False,
        show_log=False,
        det_db_box_thresh=0.3,
        rec_batch_num=6
    )
    table_engine = PPStructure(
        lang='en',
        layout=True,
        table=True,
        ocr=True,
        use_gpu=False,
        recovery=True,
        return_ocr_result_in_table=True,
        show_log=False
    )
    return ocr_en, ocr_ch, table_engine


@st.cache_resource
def load_easyocr_models():
    if not EASYOCR_OK:
        return None, None
    ru_reader = easyocr.Reader(['ru', 'en'], gpu=False, download_enabled=True)
    ch_reader = easyocr.Reader(['ch_sim', 'en'], gpu=False, download_enabled=True)
    return ru_reader, ch_reader


with st.spinner("⏳ Загрузка моделей..."):
    ocr_en, ocr_ch, table_engine = load_paddle() if PADDLE_OK else (None, None, None)
    easy_ru, easy_ch = load_easyocr_models() if EASYOCR_OK else (None, None)


# ========= Утилиты =========
def preprocess_image(img: np.ndarray, scale: float = 2.0) -> np.ndarray:
    """Улучшенная предобработка"""
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
    return denoised


def align_and_highlight_differences(texts_dict: Dict[str, str]) -> str:
    """
    Выравнивание текстов и подсветка различий цветом
    texts_dict: {"Engine Name": "recognized text"}
    """
    if not texts_dict:
        return ""

    # Цвета для каждого движка
    colors = {
        'PaddleOCR-EN': '#4A90E2',  # Синий
        'PaddleOCR-CH': '#7B68EE',  # Фиолетовый
        'Tesseract': '#50C878',  # Зелёный
        'EasyOCR-RU': '#FF6B6B',  # Красный
        'EasyOCR-CH': '#FFA07A'  # Оранжевый
    }

    engines = list(texts_dict.keys())
    texts = list(texts_dict.values())

    if len(texts) < 2:
        return f"<div style='font-family:monospace; white-space:pre-wrap;'>{texts[0]}</div>"

    # Берём первый текст как эталон
    reference = texts[0]

    html_output = "<div style='font-family:monospace; line-height:1.8;'>"

    for idx, (engine, text) in enumerate(zip(engines, texts)):
        color = colors.get(engine, '#888888')

        html_output += f"<div style='margin-bottom:15px;'>"
        html_output += f"<strong style='color:{color};'>🔹 {engine}</strong><br/>"

        if idx == 0:
            # Эталонный текст без подсветки
            html_output += f"<span style='background:#2d2d2d; padding:5px; display:inline-block; border-radius:3px;'>{text}</span>"
        else:
            # Сравнение с эталоном
            matcher = SequenceMatcher(None, reference, text)
            highlighted = ""

            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'equal':
                    # Совпадающие символы - серый фон
                    highlighted += f"<span style='background:#2d2d2d; padding:2px;'>{text[j1:j2]}</span>"
                elif tag == 'replace':
                    # Замена - яркая подсветка
                    highlighted += f"<span style='background:{color}; color:#000; padding:2px; font-weight:bold;'>{text[j1:j2]}</span>"
                elif tag == 'insert':
                    # Вставка - яркая подсветка
                    highlighted += f"<span style='background:{color}; color:#000; padding:2px; font-weight:bold;'>{text[j1:j2]}</span>"
                elif tag == 'delete':
                    # Удаление - показываем что отсутствует
                    highlighted += f"<span style='background:#555; color:#aaa; padding:2px; text-decoration:line-through;'>{reference[i1:i2]}</span>"

            html_output += f"<div style='padding:5px; display:inline-block; border-radius:3px;'>{highlighted}</div>"

        html_output += "</div>"

    html_output += "</div>"

    return html_output


def calculate_similarity(text1: str, text2: str) -> float:
    """Вычисление процента совпадения"""
    return SequenceMatcher(None, text1, text2).ratio() * 100


def merge_overlapping_cells(cells: List[Dict]) -> List[Dict]:
    """Объединение перекрывающихся ячеек"""
    if not cells:
        return []

    cells_sorted = sorted(cells, key=lambda c: (c.get('row_start', 0), c.get('col_start', 0)))
    merged = []
    skip_indices = set()

    for i, cell in enumerate(cells_sorted):
        if i in skip_indices:
            continue
        current_cell = cell.copy()

        for j in range(i + 1, len(cells_sorted)):
            if j in skip_indices:
                continue
            other = cells_sorted[j]

            if (abs(other.get('row_start', 0) - current_cell.get('row_start', 0)) <= 1 and
                    abs(other.get('col_start', 0) - current_cell.get('col_start', 0)) <= 1):

                bbox1 = current_cell.get('bbox', [0, 0, 0, 0])
                bbox2 = other.get('bbox', [0, 0, 0, 0])

                if len(bbox1) == 4 and len(bbox2) == 4:
                    current_cell['bbox'] = [
                        min(bbox1[0], bbox2[0]),
                        min(bbox1[1], bbox2[1]),
                        max(bbox1[2], bbox2[2]),
                        max(bbox1[3], bbox2[3])
                    ]
                skip_indices.add(j)

        merged.append(current_cell)

    return merged


def extract_cell_text(roi: np.ndarray, methods: Dict) -> Dict[str, str]:
    """Извлечение текста всеми методами для сравнения"""
    results = {}

    if methods['tesseract']:
        try:
            text = pytesseract.image_to_string(
                roi,
                lang="eng+rus+chi_sim",
                config="--psm 6 --oem 1"
            ).strip()
            if text:
                results['Tesseract'] = text
        except:
            pass

    if methods['easy_ru']:
        try:
            text_list = methods['easy_ru'].readtext(roi, detail=0, paragraph=True)
            text = " ".join([t.strip() for t in text_list if t.strip()])
            if text:
                results['EasyOCR-RU'] = text
        except:
            pass

    if methods['easy_ch']:
        try:
            text_list = methods['easy_ch'].readtext(roi, detail=0, paragraph=True)
            text = " ".join([t.strip() for t in text_list if t.strip()])
            if text:
                results['EasyOCR-CH'] = text
        except:
            pass

    return results


def build_table_from_cells(cells: List[Dict], img: np.ndarray, methods: Dict) -> pd.DataFrame:
    """Построение таблицы из ячеек"""
    if not cells:
        return pd.DataFrame()

    cells = merge_overlapping_cells(cells)
    max_row = max(cell.get('row_start', 0) for cell in cells) + 1
    max_col = max(cell.get('col_start', 0) for cell in cells) + 1

    table_data = {}
    progress_bar = st.progress(0)
    total_cells = len(cells)

    for idx, cell in enumerate(cells):
        r = cell.get('row_start', 0)
        c = cell.get('col_start', 0)
        bbox = cell.get('bbox', [])

        text = ""

        if len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            h, w = img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 > x1 and y2 > y1:
                roi = img[y1:y2, x1:x2]

                if roi.size > 0:
                    roi_processed = preprocess_image(roi, scale=2.0)
                    roi_rgb = cv2.cvtColor(roi_processed, cv2.COLOR_GRAY2RGB)

                    # Получаем все результаты
                    all_results = extract_cell_text(roi_rgb, methods)
                    # Берём самый длинный
                    if all_results:
                        text = max(all_results.values(), key=len)

        if r not in table_data:
            table_data[r] = {}
        table_data[r][c] = text

        progress_bar.progress((idx + 1) / total_cells)

    progress_bar.empty()

    rows = []
    for r in sorted(table_data.keys()):
        row = []
        for c in range(max_col):
            row.append(table_data[r].get(c, ""))
        rows.append(row)

    return pd.DataFrame(rows)


# ========= UI загрузки =========
uploaded_file = st.file_uploader("📁 Загрузите изображение", type=["png", "jpg", "jpeg"])

if uploaded_file:
    st.success(f"✅ {uploaded_file.name}")

    if st.button("🔍 Распознать и сравнить", type="primary"):
        start_time = time.time()

        with st.spinner("🔄 Обработка изображения..."):
            file_bytes = uploaded_file.read()
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_bytes)
                img_path = tmp.name

            # Результаты от каждого движка отдельно
            engine_results = {
                'PaddleOCR-EN': [],
                'PaddleOCR-CH': [],
                'Tesseract': [],
                'EasyOCR-RU': [],
                'EasyOCR-CH': []
            }

            tables_data = []

            try:
                img_cv = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

                st.image(img_rgb, caption="Загруженное изображение", use_container_width=True)

                # ===== PaddleOCR EN =====
                if ocr_en:
                    res = ocr_en.ocr(img_cv, cls=True)
                    if res and res[0]:
                        for ln in res[0]:
                            text = ln[1][0].strip()
                            if text:
                                engine_results['PaddleOCR-EN'].append(text)

                # ===== PaddleOCR CH =====
                if ocr_ch:
                    res = ocr_ch.ocr(img_cv, cls=True)
                    if res and res[0]:
                        for ln in res[0]:
                            text = ln[1][0].strip()
                            if text:
                                engine_results['PaddleOCR-CH'].append(text)

                # ===== Tesseract =====
                if TESSERACT_OK:
                    raw = pytesseract.image_to_string(
                        img_rgb,
                        lang="eng+rus+chi_sim",
                        config="--oem 1 --psm 6"
                    )
                    for line in raw.splitlines():
                        line = line.strip()
                        if line:
                            engine_results['Tesseract'].append(line)

                # ===== EasyOCR RU =====
                if easy_ru:
                    out = easy_ru.readtext(img_rgb, detail=0, paragraph=False)
                    for t in out:
                        t = t.strip()
                        if t:
                            engine_results['EasyOCR-RU'].append(t)

                # ===== EasyOCR CH =====
                if easy_ch:
                    out = easy_ch.readtext(img_rgb, detail=0, paragraph=False)
                    for t in out:
                        t = t.strip()
                        if t:
                            engine_results['EasyOCR-CH'].append(t)

                # ===== Таблицы =====
                if table_engine:
                    table_result = table_engine(img_path)
                    methods = {
                        'tesseract': TESSERACT_OK,
                        'easy_ru': easy_ru,
                        'easy_ch': easy_ch
                    }

                    for idx, region in enumerate(table_result):
                        if region.get("type") != "table":
                            continue
                        res = region.get("res", {})
                        cells = res.get("cells", [])

                        if cells:
                            df = build_table_from_cells(cells, img_cv, methods)
                            if not df.empty:
                                tables_data.append(df)
                        else:
                            html = res.get("html", "")
                            if html:
                                try:
                                    df = pd.read_html(html)[0]
                                    tables_data.append(df)
                                except:
                                    pass

            finally:
                try:
                    os.unlink(img_path)
                except:
                    pass

            elapsed_time = time.time() - start_time
            st.success(f"✅ Обработка завершена за {elapsed_time:.2f} сек")

            # ===== РЕЗУЛЬТАТЫ С ПОДСВЕТКОЙ =====
            st.divider()
            st.subheader("🔤 Сравнение результатов OCR")

            # Подготовка текстов для сравнения
            comparison_texts = {}
            for engine, lines in engine_results.items():
                if lines:
                    comparison_texts[engine] = "\n".join(lines)

            if comparison_texts:
                # Статистика совпадений
                st.subheader("📊 Статистика совпадений")

                engines_list = list(comparison_texts.keys())
                if len(engines_list) >= 2:
                    cols = st.columns(len(engines_list) - 1)
                    reference_text = comparison_texts[engines_list[0]]

                    for idx, engine in enumerate(engines_list[1:]):
                        with cols[idx]:
                            similarity = calculate_similarity(reference_text, comparison_texts[engine])
                            st.metric(
                                f"{engines_list[0]} ↔ {engine}",
                                f"{similarity:.1f}%"
                            )

                st.divider()

                # Визуальное сравнение с подсветкой
                st.subheader("🎨 Посимвольное сравнение (подсветка различий)")

                highlighted_html = align_and_highlight_differences(comparison_texts)
                st.markdown(highlighted_html, unsafe_allow_html=True)

                st.divider()

                # Кнопки скачивания для каждого движка
                st.subheader("📥 Скачать результаты")
                cols = st.columns(len(comparison_texts))

                for idx, (engine, text) in enumerate(comparison_texts.items()):
                    with cols[idx]:
                        st.download_button(
                            f"💾 {engine}",
                            text,
                            f"ocr_{engine.lower().replace('-', '_')}.txt",
                            "text/plain"
                        )

            else:
                st.info("Текст не найден")

            # ===== Таблицы =====
            if tables_data:
                st.divider()
                st.subheader("📊 Распознанные таблицы")

                for i, df in enumerate(tables_data, 1):
                    st.write(f"**Таблица {i}** ({df.shape[0]} строк × {df.shape[1]} столбцов)")
                    st.dataframe(df, use_container_width=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        csv = df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            f"📥 CSV таблица {i}",
                            csv,
                            f"table_{i}.csv",
                            "text/csv"
                        )
                    with col2:
                        excel_buffer = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
                        df.to_excel(excel_buffer.name, index=False, engine='openpyxl')
                        with open(excel_buffer.name, 'rb') as f:
                            st.download_button(
                                f"📥 Excel таблица {i}",
                                f.read(),
                                f"table_{i}.xlsx",
                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        try:
                            os.unlink(excel_buffer.name)
                        except:
                            pass