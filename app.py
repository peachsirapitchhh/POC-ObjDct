# Car Brand Detection - Streamlit Interface
# วิธีรัน:  streamlit run app.py

import io

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Car Brand Detection", page_icon="🚗", layout="wide")

st.title("🚗 Car Brand Detection")
st.caption("อัปโหลดรูปรถ แล้วโมเดล YOLOv8 จะตรวจจับโลโก้/แบรนด์ของรถให้")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ ตั้งค่า")
    model_path = st.text_input(
        "Path ของโมเดล (.pt)",
        value="best.pt",
        help="ไฟล์ weights ที่ได้จากการเทรน เช่น best.pt (จากโฟลเดอร์ runs ที่ copy ไว้ใน Drive)",
    )
    conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)


@st.cache_resource
def load_model(path: str) -> YOLO:
    return YOLO(path)


try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"โหลดโมเดลไม่สำเร็จ: {e}\n\nตรวจสอบ path ของไฟล์ .pt ใน sidebar")
    st.stop()

# ---------- Upload ----------
uploaded_files = st.file_uploader(
    "อัปโหลดรูปภาพ (ได้หลายรูป)",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("⬆️ อัปโหลดรูปเพื่อเริ่มตรวจจับ")
    st.stop()

for uploaded in uploaded_files:
    st.divider()
    st.subheader(f"📷 {uploaded.name}")

    image = Image.open(uploaded).convert("RGB")
    results = model.predict(np.array(image), conf=conf_threshold, verbose=False)
    result = results[0]

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="รูปต้นฉบับ", use_container_width=True)
    with col2:
        annotated = result.plot()[:, :, ::-1]  # BGR -> RGB
        st.image(annotated, caption="ผลการตรวจจับ", use_container_width=True)

    # ---------- Detection output ----------
    if len(result.boxes) == 0:
        st.warning("ไม่พบแบรนด์รถในรูปนี้")
        continue

    rows = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        rows.append(
            {
                "Brand": result.names[cls_id],
                "Confidence": f"{float(box.conf[0]):.2%}",
                "BBox (x1, y1, x2, y2)": [round(v) for v in box.xyxy[0].tolist()],
            }
        )

    brands = sorted({r["Brand"] for r in rows})
    st.success(f"✅ ตรวจพบแบรนด์: **{', '.join(brands)}**")
    st.table(rows)

    # ปุ่มดาวน์โหลดรูปผลลัพธ์
    buf = io.BytesIO()
    Image.fromarray(annotated).save(buf, format="PNG")
    st.download_button(
        "⬇️ ดาวน์โหลดรูปผลลัพธ์",
        data=buf.getvalue(),
        file_name=f"detected_{uploaded.name}.png",
        mime="image/png",
        key=f"dl_{uploaded.name}",
    )
