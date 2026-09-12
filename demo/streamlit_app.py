"""
streamlit_app.py

Demo Streamlit: so sánh trực tiếp Prototypical Network (few-shot)
và Fine-tuning (live) trên cùng một ảnh lá sầu riêng.

CHẠY Ở MÁY LOCAL. Chuẩn bị thư mục demo/ với cấu trúc:
    demo/
      streamlit_app.py          (file này)
      protonet_conv4_best.pt
      prototypes.json
      support_set/
        manifest.json
        Leaf_Algal/00.jpg, 01.jpg, ...
        ... (đủ 6 lớp)

Cài thư viện (1 lần):
    pip install streamlit torch torchvision pillow pandas

Chạy:
    streamlit run streamlit_app.py

LƯU Ý:
Ứng dụng được xây dựng để so sánh hai phương pháp trên cùng tập 6 lớp
đang đánh giá trong đề tài. Mỗi nhánh luôn trả về lớp có xác suất cao nhất
trong 6 lớp này; ứng dụng không thực hiện cơ chế phát hiện hoặc từ chối
ảnh ngoài phân phối (OOD).
"""

import io
import json
import os
from datetime import datetime

import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import torchvision.transforms as T
from PIL import Image

MODEL_PATH = 'protonet_conv4_best.pt'
PROTOTYPES_PATH = 'prototypes.json'
SUPPORT_SET_DIR = 'support_set'
MANIFEST_PATH = os.path.join(SUPPORT_SET_DIR, 'manifest.json')

IMG_SIZE = 84  # PHẢI khớp IMG_SIZE trong data_prep.py

FT_STEPS = 20
FT_LR = 1e-3
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

eval_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Thông tin hiển thị cho từng lớp bệnh — chỉ phục vụ UI, không ảnh hưởng
# logic mô hình. Sửa nhãn tiếng Việt tại đây nếu cần cho đúng thuật ngữ.
CLASS_INFO = {
    'Leaf_Algal':          {'label': 'Đốm rong / tảo lá',        'emoji': '🟢', 'color': '#8BC34A'},
    'Leaf_Blight':         {'label': 'Cháy lá (Blight)',          'emoji': '🟤', 'color': '#8D6E63'},
    'Leaf_Colletotrichum': {'label': 'Thán thư (Colletotrichum)', 'emoji': '🟠', 'color': '#FF7043'},
    'Leaf_Healthy':        {'label': 'Lá khỏe mạnh',              'emoji': '✅', 'color': '#4CAF50'},
    'Leaf_Phomopsis':      {'label': 'Đốm lá Phomopsis',          'emoji': '🟡', 'color': '#FBC02D'},
    'Leaf_Rhizoctonia':    {'label': 'Cháy lá Rhizoctonia',       'emoji': '🔴', 'color': '#E53935'},
}


def class_display(cls: str) -> str:
    info = CLASS_INFO.get(cls)
    if not info:
        return cls
    return f"{info['emoji']} {cls} — {info['label']}"


# ============================================================
# 1. KIẾN TRÚC — PHẢI giống hệt lúc train
# ============================================================

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class Conv4Backbone(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(3, 64), conv_block(64, 64),
            conv_block(64, 64), conv_block(64, embedding_dim),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)


# ============================================================
# 2. NẠP TÀI NGUYÊN
# ============================================================

@st.cache_resource
def load_protonet():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PROTOTYPES_PATH):
        return None

    model = Conv4Backbone()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    with open(PROTOTYPES_PATH, 'r', encoding='utf-8') as f:
        proto_data = json.load(f)

    novel_classes = proto_data['novel_classes']
    prototypes = torch.tensor(
        [proto_data['protonet_prototypes'][c] for c in novel_classes]
    )

    return {
        'model': model,
        'novel_classes': novel_classes,
        'prototypes': prototypes,
    }


@st.cache_resource
def load_finetune_backbone():
    base = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V1)
    base.classifier = nn.Identity()
    base.eval()
    for p in base.parameters():
        p.requires_grad = False
    return base


@st.cache_resource
def load_support_set():
    if not os.path.exists(MANIFEST_PATH):
        return None

    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    novel_classes = manifest['novel_classes']
    class_images = {}
    class_pil = {}
    for c in novel_classes:
        rel_paths = manifest['classes'][c]
        pil_imgs = [Image.open(os.path.join(SUPPORT_SET_DIR, rp)).convert('RGB') for rp in rel_paths]
        class_pil[c] = pil_imgs
        class_images[c] = torch.stack([eval_transform(im) for im in pil_imgs])

    return {
        'novel_classes': novel_classes,
        'class_images': class_images,
        'class_pil': class_pil,
        'n_support_per_class': manifest['n_support_per_class'],
    }


# ============================================================
# 3. DỰ ĐOÁN — ProtoNet
# ============================================================

def predict_protonet(protonet, image: Image.Image):
    x = eval_transform(image.convert('RGB')).unsqueeze(0)
    with torch.no_grad():
        emb = protonet['model'](x)
        dists = torch.pow(emb - protonet['prototypes'], 2).sum(dim=1)
        probs = F.softmax(-dists, dim=0)

    ranked = sorted(
        zip(protonet['novel_classes'], probs.tolist()),
        key=lambda t: -t[1]
    )
    return ranked


# ============================================================
# 4. DỰ ĐOÁN — Fine-tuning LIVE
# ============================================================

def run_live_finetune(backbone, support_set, image: Image.Image):
    novel_classes = support_set['novel_classes']
    n_way = len(novel_classes)

    support_x, support_y = [], []
    for label_idx, c in enumerate(novel_classes):
        imgs = support_set['class_images'][c]
        support_x.append(imgs)
        support_y += [label_idx] * imgs.size(0)
    support_x = torch.cat(support_x, dim=0)
    support_y = torch.tensor(support_y)

    with torch.no_grad():
        support_feat = backbone(support_x)
    feat_dim = support_feat.shape[1]

    classifier = nn.Linear(feat_dim, n_way)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=FT_LR)
    criterion = nn.CrossEntropyLoss()

    classifier.train()
    for _ in range(FT_STEPS):
        optimizer.zero_grad()
        logits = classifier(support_feat)
        loss = criterion(logits, support_y)
        loss.backward()
        optimizer.step()
    classifier.eval()

    x = eval_transform(image.convert('RGB')).unsqueeze(0)
    with torch.no_grad():
        feat = backbone(x)
        logits = classifier(feat)
        probs = F.softmax(logits, dim=1).squeeze(0)

    ranked = sorted(zip(novel_classes, probs.tolist()), key=lambda t: -t[1])
    return ranked


# ============================================================
# 5. GIAO DIỆN
# ============================================================

st.set_page_config(
    page_title="Durian Leaf AI — Nhận diện bệnh lá sầu riêng",
    page_icon="🍃",
    layout="wide",
)

st.markdown("""
<style>
.hero {
    background: linear-gradient(120deg, #1B5E20 0%, #4CAF50 55%, #8BC34A 100%);
    padding: 28px 32px; border-radius: 16px; color: white; margin-bottom: 18px;
}
.hero h1 { margin: 0; font-size: 30px; }
.hero p { margin: 6px 0 0 0; opacity: 0.92; font-size: 15px; }
.result-card {
    border-radius: 14px; padding: 18px 20px; margin-bottom: 10px;
    border: 1px solid rgba(0,0,0,0.08);
}
.result-card.ok { background: #F1F8E9; border-left: 6px solid #4CAF50; }
.result-card.warn { background: #E3F2FD; border-left: 6px solid #1E88E5; }
.class-row { display: flex; justify-content: space-between; font-size: 14px; margin: 4px 0; }
.badge {
    display: inline-block; padding: 2px 10px; border-radius: 999px;
    font-size: 12px; font-weight: 600; color: white; margin-left: 8px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🍃 Durian Leaf AI</h1>
  <p>So sánh trực tiếp Prototypical Network (few-shot, không train lại) và
  Fine-tuning (train lại lớp cuối bằng ảnh support thật mỗi lần chạy) trên
  cùng một ảnh lá sầu riêng.</p>
</div>
""", unsafe_allow_html=True)

protonet = load_protonet()
support_set = load_support_set()

if 'history' not in st.session_state:
    st.session_state.history = []

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Thông tin mô hình")
    if protonet is not None:
        st.metric("Số lớp nhận diện trong demo", len(protonet['novel_classes']))
        st.divider()
        st.subheader("6 lớp được so sánh")
        for c in protonet['novel_classes']:
            info = CLASS_INFO.get(c, {'emoji': '•', 'label': c, 'color': '#999'})
            st.markdown(f"{info['emoji']} **{c}** — {info['label']}")
        st.divider()
        st.info(
            "Ứng dụng dùng cùng 6 lớp và cùng tập support để so sánh trực tiếp "
            "Prototypical Network với Fine-tuning trên một ảnh đầu vào."
        )
    else:
        st.error("Thiếu file mô hình / prototypes.json.")

# ---------------- Tabs ----------------
tab_predict, tab_gallery, tab_history, tab_about = st.tabs(
    ["🔍 Nhận diện", "🖼️ Thư viện mẫu chuẩn", "📊 Lịch sử", "ℹ️ Giới thiệu & Giới hạn"]
)

# ====================== TAB: NHẬN DIỆN ======================
with tab_predict:
    if protonet is None:
        st.error(f"Không tìm thấy `{MODEL_PATH}` hoặc `{PROTOTYPES_PATH}`.")
    elif support_set is None:
        st.error(f"Không tìm thấy `{MANIFEST_PATH}` — cần thư mục support_set/ đầy đủ.")
    else:
        col_upload, col_sample = st.columns([2, 1])
        with col_upload:
            uploaded_file = st.file_uploader(
                "Upload ảnh lá sầu riêng (jpg/png)", type=['jpg', 'jpeg', 'png'], key="uploader"
            )
        with col_sample:
            st.caption("...hoặc thử nhanh bằng ảnh mẫu có sẵn:")
            sample_class = st.selectbox("Lớp", support_set['novel_classes'], key="sample_class")
            sample_idx = st.number_input(
                "Ảnh #", 0, len(support_set['class_pil'][sample_class]) - 1, 0, key="sample_idx"
            )
            use_sample = st.button("Dùng ảnh mẫu này ↴", use_container_width=True)

        image = None
        source_label = None
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            source_label = uploaded_file.name
        elif use_sample:
            image = support_set['class_pil'][sample_class][sample_idx]
            source_label = f"mẫu: {sample_class}/{sample_idx:02d}"

        if image is not None:
            img_col, info_col = st.columns([1, 2])
            with img_col:
                st.image(image, caption=source_label, use_container_width=True)

            col1, col2 = st.columns(2)

            # --- ProtoNet ---
            with col1:
                st.subheader("Prototypical Network")
                ranked_pn = predict_protonet(protonet, image)
                top_class, top_prob = ranked_pn[0]

                info = CLASS_INFO.get(
                    top_class,
                    {'emoji': '', 'label': top_class, 'color': '#4CAF50'}
                )
                st.markdown(f"""
                <div class="result-card ok">
                <b>{info['emoji']} {top_class}</b> — {info['label']}
                <span class="badge" style="background:{info['color']};">{top_prob*100:.1f}%</span>
                <p style="font-size:13px; opacity:0.8;">
                Dự đoán có xác suất cao nhất trong 6 lớp của bài toán.
                </p>
                </div>
                """, unsafe_allow_html=True)

                df_pn = pd.DataFrame(
                    {"Xác suất": [p for _, p in ranked_pn]},
                    index=[c for c, _ in ranked_pn],
                )
                st.bar_chart(df_pn, height=220)
                with st.expander("Xem chi tiết từng lớp"):
                    for cls, prob in ranked_pn:
                        st.write(f"{class_display(cls)} — **{prob*100:.1f}%**")

            # --- Fine-tuning ---
            with col2:
                st.subheader("Fine-tuning (live)")
                with st.spinner(f"Đang fine-tune {FT_STEPS} bước trên "
                                 f"{support_set['n_support_per_class']} ảnh support/lớp..."):
                    backbone = load_finetune_backbone()
                    ranked_ft = run_live_finetune(backbone, support_set, image)

                top_class_ft, top_prob_ft = ranked_ft[0]
                info_ft = CLASS_INFO.get(top_class_ft, {'emoji': '', 'label': top_class_ft, 'color': '#1E88E5'})
                st.markdown(f"""
                <div class="result-card warn">
                <b>{info_ft['emoji']} {top_class_ft}</b> — {info_ft['label']}
                <span class="badge" style="background:{info_ft['color']};">{top_prob_ft*100:.1f}%</span>
                <p style="font-size:13px; opacity:0.85;">
                Kết quả dùng để đối chiếu trực tiếp với nhánh Prototypical Network.
                </p>
                </div>
                """, unsafe_allow_html=True)

                df_ft = pd.DataFrame(
                    {"Xác suất": [p for _, p in ranked_ft]},
                    index=[c for c, _ in ranked_ft],
                )
                st.bar_chart(df_ft, height=220)
                with st.expander("Xem chi tiết từng lớp"):
                    for cls, prob in ranked_ft:
                        st.write(f"{class_display(cls)} — **{prob*100:.1f}%**")

            # --- Lưu lịch sử + báo cáo ---
            st.session_state.history.append({
                "Thời gian": datetime.now().strftime("%H:%M:%S"),
                "Ảnh": source_label,
                "ProtoNet": top_class,
                "ProtoNet %": round(top_prob * 100, 1),
                "Fine-tune": top_class_ft,
                "Fine-tune %": round(top_prob_ft * 100, 1),
            })

            report_lines = [
                f"Ảnh: {source_label}",
                f"Thời gian: {datetime.now().isoformat(timespec='seconds')}",
                "",
                "== Prototypical Network ==",
                f"Dự đoán: {top_class} ({top_prob*100:.1f}%)",
                "",
                "== Fine-tuning (live) ==",
                f"Dự đoán: {top_class_ft} ({top_prob_ft*100:.1f}%)",
            ]
            st.download_button(
                "⬇️ Tải báo cáo kết quả (.txt)",
                data="\n".join(report_lines),
                file_name=f"ket_qua_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            )

# ====================== TAB: THƯ VIỆN MẪU ======================
with tab_gallery:
    st.subheader("Ảnh support set dùng để tính prototype / fine-tune")
    if support_set is None:
        st.info("Chưa có support_set/ để hiển thị.")
    else:
        gallery_class = st.selectbox(
            "Chọn lớp bệnh để xem ảnh mẫu", support_set['novel_classes'], key="gallery_class"
        )
        info = CLASS_INFO.get(gallery_class, {'emoji': '', 'label': gallery_class})
        st.markdown(f"### {info['emoji']} {gallery_class} — {info['label']}")
        imgs = support_set['class_pil'][gallery_class]
        cols = st.columns(5)
        for i, im in enumerate(imgs):
            with cols[i % 5]:
                st.image(im, caption=f"{i:02d}", use_container_width=True)

# ====================== TAB: LỊCH SỬ ======================
with tab_history:
    st.subheader("Lịch sử các lần nhận diện trong phiên này")
    if not st.session_state.history:
        st.info("Chưa có lần nhận diện nào. Hãy sang tab 🔍 Nhận diện để bắt đầu.")
    else:
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist, use_container_width=True, hide_index=True)
        csv = df_hist.to_csv(index=False).encode('utf-8-sig')
        st.download_button("⬇️ Tải lịch sử (.csv)", data=csv, file_name="lich_su_du_doan.csv")
        if st.button("🗑️ Xóa lịch sử"):
            st.session_state.history = []
            st.rerun()

# ====================== TAB: GIỚI THIỆU ======================
with tab_about:
    st.subheader("Về ứng dụng này")
    st.markdown("""
Ứng dụng minh họa và so sánh trực tiếp hai cách tiếp cận trên cùng bài toán
nhận diện 6 lớp lá/bệnh lá sầu riêng, sử dụng cùng ảnh đầu vào:

- **Prototypical Network** — không huấn luyện lại khi dự đoán. Mô hình so sánh
  embedding của ảnh test với prototype đại diện cho từng lớp.
- **Fine-tuning (live)** — sử dụng ảnh support để huấn luyện một lớp Linear
  mới trong lúc chạy demo, sau đó dự đoán ảnh test.

Hai nhánh được đặt cạnh nhau để quan sát sự khác biệt về kết quả dự đoán
trên cùng một ảnh.
    """)

    st.subheader("Phạm vi của demo")
    st.markdown("""
Demo tập trung vào việc so sánh **Prototypical Network** và **Fine-tuning**
trên 6 lớp đã được chuẩn bị trong tập đánh giá. Ứng dụng luôn hiển thị lớp
có xác suất cao nhất trong 6 lớp này và **không tích hợp cơ chế phát hiện
hoặc từ chối ảnh ngoài phân phối (OOD)**.

Các kết quả cần được diễn giải trong phạm vi dữ liệu và 6 lớp của bài toán.
    """)

    if protonet is not None:
        st.subheader("Thông tin mô hình")
        st.json({
            "n_classes": len(protonet['novel_classes']),
            "classes": protonet['novel_classes'],
        })
