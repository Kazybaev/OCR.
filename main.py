# main.py — OCR Ensemble: PaddleOCR + Tesseract + EasyOCR (Multilingual: EN + RU + ZH)
# Гибридные таблицы: структура — PaddleOCR, текст — Tesseract/EasyOCR
# Полностью автоматический многоязычный режим

import streamlit as st
import os
import tempfile
from pathlib import Path
import cv2
import pandas as pd
import re
import pytesseract

# ========= EasyOCR (проверка доступности) =========
try:
    import easyocr
    EASYOCR_OK = True
except:
    EASYOCR_OK = False
    st.warning("⚠️ EasyOCR недоступен")

# ========= Путь к Tesseract (Windows) =========
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ========= PaddleOCR импорты =========
try:
    from paddleocr import PaddleOCR, PPStructure
    PADDLE_OK = True
except Exception as e:
    PADDLE_OK = False
    st.error(f"❌ PaddleOCR ошибка: {e}")

# ========= UI =========
st.set_page_config(page_title="Multilingual OCR 3-in-1", layout="wide")
st.title("📸 Multilingual OCR: PaddleOCR + Tesseract + EasyOCR")
st.caption("Распознаёт EN + RU + Chinese. Поддерживает таблицы.")

# ========= Проверка Tesseract =========
TESSERACT_OK = True
try:
    ver = pytesseract.get_tesseract_version()
    st.success(f"✅ Tesseract подключён: {ver}")
except:
    TESSERACT_OK = False
    st.error("❌ Tesseract не найден")

# ========= Инициализация моделей =========
@st.cache_resource
def load_paddle():
    if not PADDLE_OK:
        return None, None, None
    ocr_en = PaddleOCR(
        lang='en',
        use_angle_cls=True,
        use_gpu=False,
        show_log=False
    )
    ocr_ch = PaddleOCR(
        lang='ch',
        use_angle_cls=True,
        use_gpu=False,
        show_log=False
    )
    table_engine = PPStructure(
        lang='en',
        layout=True,
        use_gpu=False,
        recovery=True,
        return_ocr_result_in_table=True,
        show_log=False
    )
    return ocr_en, ocr_ch, table_engine

@st.cache_resource
def load_easyocr_ru():
    return easyocr.Reader(['ru','en'], gpu=False, download_enabled=True)

@st.cache_resource
def load_easyocr_ch():
    return easyocr.Reader(['ch_sim','en'], gpu=False, download_enabled=True)

with st.spinner("⏳ Загрузка моделей..."):
    ocr_en, ocr_ch, table_engine = load_paddle() if PADDLE_OK else (None, None, None)
    easy_ru = load_easyocr_ru() if EASYOCR_OK else None
    easy_ch = load_easyocr_ch() if EASYOCR_OK else None

# ========= Утилита =========
def normalize_text(s):
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s\u0400-\u04FF\u4e00-\u9fff]', '', s)).strip().lower()

# ========= UI загрузки =========
uploaded_file = st.file_uploader("📁 Загрузите изображение", type=["png","jpg","jpeg"])

if uploaded_file:
    st.success(f"✅ {uploaded_file.name}")

    if st.button("🔍 Распознать всеми OCR", type="primary"):
        with st.spinner("Обработка..."):
            # временный файл
            file_bytes = uploaded_file.read()
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_bytes)
                img_path = tmp.name

            raw_texts = []   # [(src, text)]
            enhanced_tables = []

            try:
                img_cv = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

                # ===== 1. PaddleOCR EN =====
                if ocr_en:
                    res = ocr_en.ocr(img_cv, cls=True)
                    if res and res[0]:
                        for ln in res[0]:
                            text = ln[1][0].strip()
                            if text:
                                raw_texts.append(("PaddleOCR", text))

                # ===== 2. PaddleOCR CH =====
                if ocr_ch:
                    res = ocr_ch.ocr(img_cv, cls=True)
                    if res and res[0]:
                        for ln in res[0]:
                            text = ln[1][0].strip()
                            if text:
                                raw_texts.append(("PaddleOCR", text))

                # ===== 3. Tesseract (EN + RU + CH) =====
                if TESSERACT_OK:
                    raw = pytesseract.image_to_string(
                        img_rgb,
                        lang="eng+rus+chi_sim",
                        config="--oem 1 --psm 6"
                    )
                    for line in raw.splitlines():
                        line = line.strip()
                        if line:
                            raw_texts.append(("Tesseract", line))

                # ===== 4. EasyOCR: RU + EN =====
                if easy_ru:
                    out = easy_ru.readtext(img_rgb, detail=0, paragraph=False)
                    for t in out:
                        t = t.strip()
                        if t:
                            raw_texts.append(("EasyOCR", t))

                # ===== 5. EasyOCR: CH + EN =====
                if easy_ch:
                    out = easy_ch.readtext(img_rgb, detail=0, paragraph=False)
                    for t in out:
                        t = t.strip()
                        if t:
                            raw_texts.append(("EasyOCR", t))

                # ============================================================
                # =============== ГИБРИДНЫЕ ТАБЛИЦЫ ===========================
                # ============================================================
                if table_engine:
                    table_result = table_engine(img_path)

                    for region in table_result:
                        if region.get("type") != "table":
                            continue

                        res = region.get("res", {})
                        cells = res.get("cells", [])

                        # если PPStructure дал только html
                        if not cells:
                            html = res.get("html", "")
                            if html:
                                enhanced_tables.append(html)
                            continue

                        rows = {}

                        for cell in cells:
                            r = cell.get("row_start", 0)
                            c = cell.get("col_start", 0)
                            bbox = cell.get("bbox", [])
                            text = ""

                            if len(bbox) == 4:
                                x1, y1, x2, y2 = map(int, bbox)

                                roi = img_cv[y1:y2, x1:x2]

                                if roi.size > 0:
                                    roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                                    roi_bin = cv2.threshold(
                                        roi_gray, 0, 255,
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU
                                    )[1]

                                    # --- 1. Tesseract
                                    if TESSERACT_OK:
                                        try:
                                            t1 = pytesseract.image_to_string(
                                                roi_bin,
                                                lang="eng+rus+chi_sim",
                                                config="--psm 6 --oem 1"
                                            ).strip()
                                            if len(t1) > 1:
                                                text = t1
                                        except:
                                            pass

                                    # --- 2. EasyOCR RU/EN
                                    if not text and easy_ru:
                                        try:
                                            r1 = easy_ru.readtext(roi, detail=0)
                                            if r1:
                                                text = " ".join([t.strip() for t in r1 if t.strip()])
                                        except:
                                            pass

                                    # --- 3. EasyOCR CH/EN
                                    if (not text or len(text) < 1) and easy_ch:
                                        try:
                                            r2 = easy_ch.readtext(roi, detail=0)
                                            if r2:
                                                text = " ".join([t.strip() for t in r2 if t.strip()])
                                        except:
                                            pass

                            if r not in rows:
                                rows[r] = {}
                            rows[r][c] = text

                        # ---- Генерация HTML таблицы ----
                        html_lines = ["<table border='1' style='border-collapse:collapse;'>"]
                        for rr in sorted(rows.keys()):
                            html_lines.append("<tr>")
                            for cc in sorted(rows[rr].keys()):
                                cell_text = rows[rr][cc]
                                cell_text = (
                                    cell_text.replace("&","&amp;")
                                             .replace("<","&lt;")
                                             .replace(">","&gt;")
                                )
                                html_lines.append(f"<td style='padding:6px;'>{cell_text}</td>")
                            html_lines.append("</tr>")
                        html_lines.append("</table>")

                        enhanced_tables.append("".join(html_lines))

            finally:
                try: os.unlink(img_path)
                except: pass

            # ===== Объединение текста =====
            seen = set()
            merged = []
            for src, txt in raw_texts:
                n = normalize_text(txt)
                if n and n not in seen:
                    seen.add(n)
                    merged.append((src, txt))

            st.divider()
            st.subheader("🔤 Объединённый текст")

            if merged:
                all_text = "\n".join(t for _, t in merged)
                st.download_button("📥 Скачать текст", all_text, "ocr_text.txt")

                # Группировка
                blocks = {"PaddleOCR": [], "Tesseract": [], "EasyOCR": []}
                for src, txt in merged:
                    blocks[src].append(txt)

                for src in ["PaddleOCR", "Tesseract", "EasyOCR"]:
                    if blocks[src]:
                        with st.expander(f"{src} ({len(blocks[src])})"):
                            st.text("\n".join(blocks[src]))
            else:
                st.info("Текст не найден")

            # ===== Таблицы =====
            st.divider()
            st.subheader("📊 Таблицы")

            if enhanced_tables:
                for i, html in enumerate(enhanced_tables, 1):
                    try:
                        df = pd.read_html(html)[0]
                        st.write(f"Таблица {i}")
                        st.dataframe(df)
                        st.download_button(
                            f"Скачать таблицу {i}",
                            df.to_csv(index=False).encode("utf-8"),
                            f"table_{i}.csv"
                        )
                    except:
                        st.warning(f"Таблица {i}: ошибка парсинга")
                        st.code(html[:1000])
            else:
                st.info("Таблицы не найдены.")
