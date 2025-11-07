# main.py — OCR Ensemble: PaddleOCR + Tesseract + EasyOCR (CPU-only, Windows-ready)
# Гибридные таблицы: структура — PaddleOCR, текст — Tesseract/EasyOCR

import streamlit as st
import os
import tempfile
from pathlib import Path
import cv2
import pandas as pd
import re
import pytesseract

EASYOCR_AVAILABLE = False
try:
    import easyocr
    _ = easyocr.Reader(['en'], gpu=False, download_enabled=False)
    EASYOCR_AVAILABLE = True
except Exception as e:
    st.warning(f"EasyOCR отключен: {e}")

# 🔑 ЯВНО УКАЗЫВАЕМ ПУТЬ К TESSERACT (работает без PATH)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Импорты OCR (отложенные, чтобы избежать ошибок при отсутствии)
try:
    from paddleocr import PaddleOCR, PPStructure
    PADDLE_OK = True
except Exception as e:
    PADDLE_OK = False
    st.error(f"❌ PaddleOCR: {e}")

try:
    import easyocr
    EASYOCR_OK = True
except Exception as e:
    EASYOCR_OK = False
    st.warning(f"⚠️ EasyOCR не установлен или недоступен: {e}")

st.set_page_config(page_title="OCR Ensemble: 3 in 1", layout="wide")
st.title("📸 OCR Ensemble: PaddleOCR + Tesseract + EasyOCR")

# === Проверка Tesseract ===
TESSERACT_OK = True
try:
    ver = pytesseract.get_tesseract_version()
    st.success(f"✅ Tesseract {ver} подключён")
except Exception as e:
    TESSERACT_OK = False
    st.error(f"❌ Tesseract не найден: {e}")

# === Инициализация движков ===
@st.cache_resource
def load_paddle():
    if not PADDLE_OK:
        return None, None
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang='en',
        use_gpu=False,
        show_log=False,
        det_db_thresh=0.3,
        det_db_box_thresh=0.5,
    )
    table_engine = PPStructure(
        lang='en',
        layout=True,
        use_gpu=False,
        show_log=False,
        recovery=True,
        return_ocr_result_in_table=True,
    )
    return ocr, table_engine

@st.cache_resource
def load_easyocr():
    if not EASYOCR_OK:
        return None
    with st.spinner("⏳ Загрузка EasyOCR (первый запуск: ~20 сек)"):
        return easyocr.Reader(['en'], gpu=False, download_enabled=True)

with st.spinner("⏳ Инициализация моделей..."):
    ocr_engine, table_engine = load_paddle() if PADDLE_OK else (None, None)
    easy_reader = load_easyocr() if EASYOCR_OK else None

def normalize_text(s):
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', s)).strip().lower()

# === UI ===
uploaded_file = st.file_uploader("📁 Загрузите изображение (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    st.success(f"✅ `{uploaded_file.name}`")

    if st.button("🔍 Распознать всеми OCR", type="primary"):
        with st.spinner("Обработка изображения..."):
            file_bytes = uploaded_file.read()
            raw_texts = []  # [(source, text), ...]
            enhanced_tables = []

            suffix = Path(uploaded_file.name).suffix.lower()
            suffix = suffix if suffix in [".png", ".jpg", ".jpeg"] else ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_bytes)
                img_path = tmp.name

            try:
                img_cv = cv2.imread(img_path)
                if img_cv is None:
                    st.error("❌ Не удалось прочитать изображение")
                    raise
                img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

                # === 1. PaddleOCR текст ===
                if ocr_engine:
                    res = ocr_engine.ocr(img_cv, cls=True)
                    if res and res[0]:
                        for line in res[0]:
                            text = line[1][0].strip()
                            if text:
                                raw_texts.append(("PaddleOCR", text))

                # === 2. Tesseract текст ===
                if TESSERACT_OK:
                    raw = pytesseract.image_to_string(
                        img_rgb, lang='eng', config='--oem 1 --psm 6'
                    )
                    for line in raw.splitlines():
                        line = line.strip()
                        if line:
                            raw_texts.append(("Tesseract", line))

                # === 3. EasyOCR текст ===
                if easy_reader:
                    try:
                        results = easy_reader.readtext(img_rgb, detail=0, paragraph=False)
                        for text in results:
                            text = text.strip()
                            if text:
                                raw_texts.append(("EasyOCR", text))
                    except Exception as e:
                        st.warning(f"⚠️ EasyOCR: {e}")

                # === 4. Гибридные таблицы (PaddleOCR layout + Tesseract/EasyOCR) ===
                if table_engine:
                    st.info("🔧 Обработка таблиц: структура → PaddleOCR, текст → Tesseract/EasyOCR")
                    table_result = table_engine(img_path)

                    for region in table_result:
                        if region.get("type") != "table":
                            continue

                        res = region.get("res", {})
                        cells = res.get("cells", [])
                        if not cells:
                            html = res.get("html", "")
                            if html:
                                enhanced_tables.append(html)
                            continue

                        # Обновляем текст в ячейках: сначала Tesseract, fallback → EasyOCR
                        rows = {}
                        for cell in cells:
                            row_idx = cell.get("row_start", 0)
                            col_idx = cell.get("col_start", 0)
                            bbox = cell.get("bbox", [])
                            text = cell.get("text", "")

                            # 1. Tesseract
                            if TESSERACT_OK and len(bbox) == 4:
                                x1, y1, x2, y2 = map(int, bbox)
                                roi = img_cv[y1:y2, x1:x2]
                                if roi.size > 0:
                                    try:
                                        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                                        text = pytesseract.image_to_string(
                                            rgb_roi,
                                            lang='eng',
                                            config='--psm 6 --oem 1 -c preserve_interword_spaces=1'
                                        ).strip()
                                    except:
                                        pass

                            # 2. EasyOCR (если Tesseract не сработал или текст пустой)
                            if (not text or len(text) < 2) and easy_reader and len(bbox) == 4:
                                x1, y1, x2, y2 = map(int, bbox)
                                roi = img_cv[y1:y2, x1:x2]
                                if roi.size > 0:
                                    try:
                                        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                                        ocr_res = easy_reader.readtext(rgb_roi, detail=0, paragraph=False)
                                        if ocr_res:
                                            text = " ".join(ocr_res).strip()
                                    except:
                                        pass

                            if row_idx not in rows:
                                rows[row_idx] = {}
                            rows[row_idx][col_idx] = text

                        # Собираем HTML
                        html_lines = ["<table border='1' style='border-collapse:collapse;'>"]
                        for r in sorted(rows.keys()):
                            html_lines.append("<tr>")
                            for c in sorted(rows[r].keys()):
                                txt = rows[r][c].replace("&", "&amp;").replace("<", "<").replace(">", ">")
                                html_lines.append(f"<td style='padding:6px;'>{txt}</td>")
                            html_lines.append("</tr>")
                        html_lines.append("</table>")
                        enhanced_tables.append("".join(html_lines))

            finally:
                if os.path.exists(img_path):
                    os.unlink(img_path)

            # === Объединение текста ===
            seen = set()
            merged = []
            for src, txt in raw_texts:
                norm = normalize_text(txt)
                if norm and norm not in seen:
                    seen.add(norm)
                    merged.append((src, txt))

            # === ВЫВОД ===
            st.divider()
            st.subheader("🔤 Объединённый текст (дубли удалены)")

            if merged:
                full_text = "\n".join(t for _, t in merged)
                st.download_button("📥 Скачать текст (txt)", full_text, "ocr_text.txt", "text/plain")

                grouped = {}
                for src, txt in merged:
                    grouped.setdefault(src, []).append(txt)

                for src in ["PaddleOCR", "Tesseract", "EasyOCR"]:
                    lines = grouped.get(src, [])
                    if lines:
                        with st.expander(f"{src} ({len(lines)} строк)"):
                            st.text("\n".join(lines))
            else:
                st.info("🔹 Текст не распознан ни одним движком")

            st.divider()
            st.subheader("📊 Гибридные таблицы")

            if enhanced_tables:
                all_dfs = []
                for i, html in enumerate(enhanced_tables, 1):
                    try:
                        dfs = pd.read_html(html, header=0)
                        if dfs:
                            df = dfs[0]
                            all_dfs.append(df)
                            st.write(f"**Таблица {i}**")
                            st.dataframe(df, use_container_width=True)

                            # Экспорт CSV
                            csv = df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                f"📥 Скачать таблицу {i} (CSV)",
                                csv,
                                f"table_{i}.csv",
                                "text/csv"
                            )
                    except Exception as e:
                        st.warning(f"Таблица {i}: не удалось распарсить")
                        with st.expander("HTML"):
                            st.code(html[:800], language="html")

                # Экспорт Excel
                if all_dfs:
                    import io
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine="openpyxl") as writer:
                        for idx, df in enumerate(all_dfs, 1):
                            df.to_excel(writer, sheet_name=f"Table_{idx}", index=False)
                    st.download_button(
                        "📥 Скачать ВСЕ таблицы (Excel)",
                        output.getvalue(),
                        "tables.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            else:
                st.info("🔹 Таблицы не найдены")



                #  .\.venv-alll\Scripts\activate
                #  streamlit run main.py
