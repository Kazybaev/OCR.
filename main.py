# main.py — OCR с объединёнными движками и посимвольным сравнением
# PaddleOCR (EN+CH) + Tesseract + EasyOCR (RU+CH) - 3 финальных результата

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
import matplotlib.pyplot as plt
import seaborn as sns

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
st.set_page_config(page_title="OCR Comparator Pro", layout="wide")
st.title("📸 OCR Сравнение: 3 движка")
st.caption("PaddleOCR | Tesseract | EasyOCR - посимвольное сравнение с подсветкой")

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


def normalize_text(s: str) -> str:
    """Нормализация текста"""
    return re.sub(r'\s+', ' ', s).strip()


def merge_texts_smart(texts: List[str]) -> str:
    """Умное объединение текстов с дедупликацией"""
    if not texts:
        return ""

    # Удаляем дубликаты с сохранением порядка
    seen = set()
    unique_lines = []

    for text in texts:
        lines = text.split('\n')
        for line in lines:
            line_norm = normalize_text(line)
            if line_norm and line_norm not in seen:
                seen.add(line_norm)
                unique_lines.append(line.strip())

    return "\n".join(unique_lines)


def align_and_highlight_differences(texts_dict: Dict[str, str]) -> str:
    """
    Посимвольное сравнение 3 движков с горизонтальным расположением
    """
    if not texts_dict or len(texts_dict) < 2:
        return "<div>Недостаточно данных для сравнения</div>"

    # Цвета для каждого движка
    colors = {
        'PaddleOCR': '#4A90E2',  # Синий
        'Tesseract': '#50C878',  # Зелёный
        'EasyOCR': '#FF6B6B'  # Красный
    }

    engines = list(texts_dict.keys())
    texts = list(texts_dict.values())

    # Берём первый как эталон
    reference = texts[0]
    reference_engine = engines[0]

    # Контейнер с горизонтальным расположением
    html_output = "<div style='display:flex; gap:15px; background:#1a1a1a; padding:20px; border-radius:8px; overflow-x:auto;'>"

    # Эталонный текст (первая колонка)
    color = colors.get(reference_engine, '#888888')
    html_output += f"<div style='flex:1; min-width:300px; border:2px solid {color}; border-radius:8px; padding:15px; background:#0d0d0d;'>"
    html_output += f"<div style='text-align:center; margin-bottom:10px;'>"
    html_output += f"<strong style='color:{color}; font-size:16px;'>🔹 {reference_engine}</strong>"
    html_output += f"<div style='color:#888; font-size:11px; margin-top:3px;'>ЭТАЛОН</div>"
    html_output += f"</div>"
    html_output += f"<div style='background:#2d2d2d; padding:12px; border-radius:5px; font-family:monospace; line-height:1.8; white-space:pre-wrap; color:#ccc; max-height:600px; overflow-y:auto;'>{reference}</div>"
    html_output += "</div>"

    # Сравнение остальных с эталоном (следующие колонки)
    for idx in range(1, len(engines)):
        engine = engines[idx]
        text = texts[idx]
        color = colors.get(engine, '#888888')

        html_output += f"<div style='flex:1; min-width:300px; border:2px solid {color}; border-radius:8px; padding:15px; background:#0d0d0d;'>"
        html_output += f"<div style='text-align:center; margin-bottom:10px;'>"
        html_output += f"<strong style='color:{color}; font-size:16px;'>🔹 {engine}</strong>"
        html_output += f"<div style='color:#888; font-size:11px; margin-top:3px;'>СРАВНЕНИЕ</div>"
        html_output += f"</div>"

        # Посимвольное сравнение
        matcher = SequenceMatcher(None, reference, text)
        highlighted = ""

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                # Совпадения - тёмный фон
                chunk = text[j1:j2].replace('\n', '<br/>')
                highlighted += f"<span style='color:#ccc;'>{chunk}</span>"
            elif tag == 'replace':
                # Замены - яркая подсветка
                chunk = text[j1:j2].replace('\n', '<br/>')
                highlighted += f"<span style='background:{color}; color:#000; font-weight:bold; padding:2px 4px; border-radius:3px;'>{chunk}</span>"
            elif tag == 'insert':
                # Вставки - яркая подсветка
                chunk = text[j1:j2].replace('\n', '<br/>')
                highlighted += f"<span style='background:{color}; color:#000; font-weight:bold; padding:2px 4px; border-radius:3px;'>{chunk}</span>"
            elif tag == 'delete':
                # Удаления - зачёркнутый текст (показываем что пропущено)
                chunk = reference[i1:i2].replace('\n', '<br/>')
                highlighted += f"<span style='background:#555; color:#999; text-decoration:line-through; padding:2px 4px;'>{chunk}</span>"

        html_output += f"<div style='background:#2d2d2d; padding:12px; border-radius:5px; font-family:monospace; line-height:1.8; white-space:pre-wrap; max-height:600px; overflow-y:auto;'>{highlighted}</div>"
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
                results['EasyOCR'] = text
        except:
            pass

    if methods['easy_ch'] and 'EasyOCR' not in results:
        try:
            text_list = methods['easy_ch'].readtext(roi, detail=0, paragraph=True)
            text = " ".join([t.strip() for t in text_list if t.strip()])
            if text:
                results['EasyOCR'] = text
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

                    all_results = extract_cell_text(roi_rgb, methods)
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

            # Временные результаты для объединения
            paddle_en_lines = []
            paddle_ch_lines = []
            tesseract_lines = []
            easy_ru_lines = []
            easy_ch_lines = []

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
                                paddle_en_lines.append(text)

                # ===== PaddleOCR CH =====
                if ocr_ch:
                    res = ocr_ch.ocr(img_cv, cls=True)
                    if res and res[0]:
                        for ln in res[0]:
                            text = ln[1][0].strip()
                            if text:
                                paddle_ch_lines.append(text)

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
                            tesseract_lines.append(line)

                # ===== EasyOCR RU =====
                if easy_ru:
                    out = easy_ru.readtext(img_rgb, detail=0, paragraph=False)
                    for t in out:
                        t = t.strip()
                        if t:
                            easy_ru_lines.append(t)

                # ===== EasyOCR CH =====
                if easy_ch:
                    out = easy_ch.readtext(img_rgb, detail=0, paragraph=False)
                    for t in out:
                        t = t.strip()
                        if t:
                            easy_ch_lines.append(t)

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

            # ===== ОБЪЕДИНЕНИЕ РЕЗУЛЬТАТОВ =====

            # PaddleOCR = EN + CH
            paddle_combined = merge_texts_smart([
                "\n".join(paddle_en_lines),
                "\n".join(paddle_ch_lines)
            ])

            # Tesseract
            tesseract_combined = "\n".join(tesseract_lines)

            # EasyOCR = RU + CH
            easy_combined = merge_texts_smart([
                "\n".join(easy_ru_lines),
                "\n".join(easy_ch_lines)
            ])

            # Финальные 3 результата
            final_results = {}
            if paddle_combined:
                final_results['PaddleOCR'] = paddle_combined
            if tesseract_combined:
                final_results['Tesseract'] = tesseract_combined
            if easy_combined:
                final_results['EasyOCR'] = easy_combined

            elapsed_time = time.time() - start_time
            st.success(f"✅ Обработка завершена за {elapsed_time:.2f} сек")

            # ===== РЕЗУЛЬТАТЫ =====
            st.divider()
            st.subheader("🔤 Сравнение 3 движков OCR")

            if final_results:

                # Визуальное сравнение с подсветкой (главный результат)
                st.caption("Первый движок — эталон. Цветом выделены различия в остальных.")

                highlighted_html = align_and_highlight_differences(final_results)
                st.markdown(highlighted_html, unsafe_allow_html=True)

                st.divider()

                # Кнопка для статистики (скрываемое меню)
                with st.expander("📊 Показать статистику совпадений и графики точности"):
                    if len(final_results) >= 2:
                        engines_list = list(final_results.keys())

                        # Цвета для движков
                        engine_colors = {
                            'PaddleOCR': '#4A90E2',
                            'Tesseract': '#50C878',
                            'EasyOCR': '#FF6B6B'
                        }

                        # Вычисление всех парных сходств (полная матрица 6 сравнений)
                        st.subheader("📈 Матрица совпадений (все пары)")

                        all_comparisons = []
                        for engine1 in engines_list:
                            for engine2 in engines_list:
                                if engine1 != engine2:
                                    sim = calculate_similarity(
                                        final_results[engine1],
                                        final_results[engine2]
                                    )
                                    all_comparisons.append({
                                        'from': engine1,
                                        'to': engine2,
                                        'similarity': sim
                                    })

                        # Отображение в 3 колонках (по 2 метрики в каждой)
                        cols = st.columns(3)

                        for idx, comp in enumerate(all_comparisons):
                            with cols[idx % 3]:
                                pair_name = f"{comp['from']} → {comp['to']}"
                                st.metric(pair_name, f"{comp['similarity']:.1f}%")

                        st.divider()

                        # График точности для каждого сравнения
                        st.subheader("📊 Графики точности (6 сравнений)")

                        # Настройка стиля seaborn
                        sns.set_style("darkgrid")
                        plt.rcParams['figure.facecolor'] = '#0E1117'
                        plt.rcParams['axes.facecolor'] = '#262730'
                        plt.rcParams['text.color'] = 'white'
                        plt.rcParams['axes.labelcolor'] = 'white'
                        plt.rcParams['xtick.color'] = 'white'
                        plt.rcParams['ytick.color'] = 'white'
                        plt.rcParams['grid.color'] = '#404040'

                        # Создаём графики для каждого сравнения
                        for idx, comp in enumerate(all_comparisons):
                            st.write(f"**{idx + 1}. {comp['from']} → {comp['to']}**")

                            sim = comp['similarity']
                            diff = 100 - sim

                            # Создаём фигуру с 2 графиками (горизонтальный бар + пай-чарт)
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))

                            # График 1: Горизонтальный bar chart
                            categories = ['Совпадение', 'Различие']
                            values = [sim, diff]
                            colors_bar = ['#50C878', '#FF6B6B']

                            bars = ax1.barh(categories, values, color=colors_bar, edgecolor='white', linewidth=1.5)
                            ax1.set_xlim(0, 100)
                            ax1.set_xlabel('Процент (%)', fontsize=11, color='white')
                            ax1.set_title(f'{comp["from"]} → {comp["to"]}', fontsize=12, color='white', pad=10)

                            # Добавляем значения на бары
                            for bar, value in zip(bars, values):
                                width = bar.get_width()
                                ax1.text(width + 2, bar.get_y() + bar.get_height() / 2,
                                         f'{value:.1f}%',
                                         ha='left', va='center', fontsize=10, color='white', fontweight='bold')

                            ax1.grid(axis='x', alpha=0.3)

                            # График 2: Pie chart
                            colors_pie = ['#50C878', '#FF6B6B']
                            explode = (0.05, 0.05)

                            wedges, texts, autotexts = ax2.pie(
                                values,
                                labels=categories,
                                colors=colors_pie,
                                autopct='%1.1f%%',
                                startangle=90,
                                explode=explode,
                                textprops={'color': 'white', 'fontsize': 10, 'fontweight': 'bold'},
                                wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
                            )

                            ax2.set_title('Распределение', fontsize=12, color='white', pad=10)

                            # Индикатор качества
                            if sim >= 80:
                                quality = "✅ Отлично"
                                quality_color = '#50C878'
                            elif sim >= 60:
                                quality = "ℹ️ Хорошо"
                                quality_color = '#4A90E2'
                            elif sim >= 40:
                                quality = "⚠️ Средне"
                                quality_color = '#FFA07A'
                            else:
                                quality = "❌ Низко"
                                quality_color = '#FF6B6B'

                            fig.text(0.5, -0.05, quality, ha='center', fontsize=13,
                                     color=quality_color, fontweight='bold')

                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()

                            st.markdown("---")

                        # Тепловая карта (Heatmap) - общая матрица сходства
                        st.subheader("🔥 Тепловая карта сходства")

                        # Создаём матрицу сходства
                        matrix_size = len(engines_list)
                        similarity_matrix = np.zeros((matrix_size, matrix_size))

                        for i, engine1 in enumerate(engines_list):
                            for j, engine2 in enumerate(engines_list):
                                if i == j:
                                    similarity_matrix[i][j] = 100
                                else:
                                    sim = calculate_similarity(
                                        final_results[engine1],
                                        final_results[engine2]
                                    )
                                    similarity_matrix[i][j] = sim

                        # Создаём heatmap
                        fig_heat, ax_heat = plt.subplots(figsize=(8, 6))

                        sns.heatmap(
                            similarity_matrix,
                            annot=True,
                            fmt='.1f',
                            cmap='RdYlGn',
                            xticklabels=engines_list,
                            yticklabels=engines_list,
                            cbar_kws={'label': 'Совпадение (%)'},
                            linewidths=2,
                            linecolor='white',
                            vmin=0,
                            vmax=100,
                            ax=ax_heat,
                            annot_kws={'fontsize': 12, 'fontweight': 'bold'}
                        )

                        ax_heat.set_title('Матрица сходства между OCR движками',
                                          fontsize=14, color='white', pad=15, fontweight='bold')
                        ax_heat.set_xlabel('Целевой движок', fontsize=11, color='white')
                        ax_heat.set_ylabel('Исходный движок', fontsize=11, color='white')

                        plt.tight_layout()
                        st.pyplot(fig_heat)
                        plt.close()

                st.divider()

                # Кнопки скачивания
                st.subheader("📥 Скачать результаты")
                cols = st.columns(len(final_results))

                for idx, (engine, text) in enumerate(final_results.items()):
                    with cols[idx]:
                        st.download_button(
                            f"💾 {engine}",
                            text,
                            f"ocr_{engine.lower()}.txt",
                            "text/plain",
                            key=f"download_{engine}"
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
                            "text/csv",
                            key=f"csv_{i}"
                        )
                    with col2:
                        excel_buffer = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
                        df.to_excel(excel_buffer.name, index=False, engine='openpyxl')
                        with open(excel_buffer.name, 'rb') as f:
                            st.download_button(
                                f"📥 Excel таблица {i}",
                                f.read(),
                                f"table_{i}.xlsx",
                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key=f"excel_{i}"
                            )
                        try:
                            os.unlink(excel_buffer.name)
                        except:
                            pass