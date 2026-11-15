# -*- coding: utf-8 -*-
"""
data_prep.py
Giai đoạn 2 — Chuẩn bị dữ liệu (dùng chung cho cả nhóm)

Gom danh sách ảnh của 3 tập lớp (base / validation / novel), lọc ảnh hỏng,
lưu thành class_to_images.json, và khai báo cấu hình + phép tiền xử lý ảnh
mà mọi file khác trong dự án import lại.

Yêu cầu trước khi chạy:
- Đã tải PlantVillage và Durian Leaf Dataset vào Google Drive
  (chỉ 1 lần, không tải lại mỗi lần chạy)
- Đã thống nhất với nhau: IMG_SIZE, N, K, Q, danh sách lớp base/val/novel

Mặc định các đường dẫn bên dưới là của Google Colab + Drive (nơi nhóm đã
chạy thực nghiệm). Muốn chạy ở máy khác thì KHÔNG cần sửa code, chỉ cần
đặt 2 biến môi trường trước khi chạy:
    DOAN_DATA_DIR   thư mục lưu trữ lâu dài (chứa plantvillage/, durian/,
                    và là nơi ghi class_to_images.json; models/ và results/
                    được tạo ngang cấp với thư mục này)
    DOAN_LOCAL_DIR  thư mục làm việc chứa dữ liệu đã giải nén
"""

import os
import json
import random

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T

# ============================================================
# 0. CẤU HÌNH CHUNG — đây là phần BẮT BUỘC 2 người phải giống nhau
#    (theo checklist Giai đoạn 0). Sửa 1 chỗ này, không sửa rải rác.
# ============================================================

SEED = 42
IMG_SIZE = 84          # 84x84 theo chuẩn phổ biến của few-shot learning
N_WAY = 5
K_SHOT = 5
Q_QUERY = 15

# Vị trí GỐC trong Google Drive (nơi lưu trữ lâu dài, chỉ tải 1 lần duy nhất
# từ Kaggle/Mendeley, không tải lại nữa).
DRIVE_BASE_DIR = os.environ.get('DOAN_DATA_DIR', '/content/drive/MyDrive/DoAn/data')

# PlantVillage: chỉ có file ZIP trong Drive (chưa giải nén sẵn).
PLANTVILLAGE_ZIP = os.path.join(DRIVE_BASE_DIR, 'plantvillage', 'plantvillage-dataset.zip')

# Durian: KHÔNG có file zip — đã được giải nén sẵn thành thư mục ngay
# trong Drive (xác nhận bằng os.walk), nằm tại:
DURIAN_DRIVE_DIR = os.path.join(
    DRIVE_BASE_DIR, 'durian',
    'A Durian Leaf Image Dataset of Common Diseases in Vietnam for Agricultural Diagnosis',
    'Durian_Leaf_Diseases'
)

# PlantVillage rất nặng (~2000+ MB, hàng chục nghìn ảnh) — đọc trực tiếp
# qua Drive (FUSE mount) rất chậm và có thể đọc thiếu lớp. Vì vậy mỗi khi
# mở phiên Colab mới, ta GIẢI NÉN LẠI (không tải lại) ra ổ đĩa local
# /content để đọc cho nhanh và đủ. /content mất khi đóng Colab, nhưng
# không sao vì file zip gốc vẫn còn nguyên trong Drive.
LOCAL_BASE_DIR = os.environ.get('DOAN_LOCAL_DIR', '/content/data')
PLANTVILLAGE_DIR = os.path.join(LOCAL_BASE_DIR, 'plantvillage dataset', 'color')

# Durian nhẹ hơn nhiều (vài trăm MB) — không cần giải nén (vì không có
# zip), chỉ cần COPY thư mục đã có sẵn từ Drive sang /content để đọc
# nhanh hơn, tránh lỗi đọc thiếu file qua FUSE giống PlantVillage.
DURIAN_DIR = os.path.join(LOCAL_BASE_DIR, 'durian')


def extract_datasets_to_local():
    """Đưa PlantVillage + Durian ra ổ local /content để đọc nhanh và đủ.
    - PlantVillage: giải nén từ file zip trong Drive.
    - Durian: copy nguyên thư mục đã giải nén sẵn từ Drive.
    CHỈ làm nếu chưa tồn tại — tránh lặp lại tốn thời gian nếu chạy
    nhiều lần trong cùng 1 phiên Colab."""
    import zipfile
    import shutil

    if not os.path.exists(PLANTVILLAGE_DIR):
        print("Đang giải nén PlantVillage ra local (chỉ chạy 1 lần/phiên)...")
        os.makedirs(LOCAL_BASE_DIR, exist_ok=True)
        with zipfile.ZipFile(PLANTVILLAGE_ZIP, 'r') as zip_ref:
            zip_ref.extractall(LOCAL_BASE_DIR)
    else:
        print("PlantVillage đã có sẵn ở local, bỏ qua giải nén.")

    if not os.path.exists(DURIAN_DIR):
        print("Đang copy Durian ra local (chỉ chạy 1 lần/phiên)...")
        shutil.copytree(DURIAN_DRIVE_DIR, DURIAN_DIR)
    else:
        print("Durian đã có sẵn ở local, bỏ qua copy.")

# Nơi lưu kết quả trung gian (danh sách ảnh mỗi lớp) — lưu vào Drive
# (không phải /content) để giữ lại giữa các phiên Colab khác nhau
CACHE_PATH = os.path.join(DRIVE_BASE_DIR, 'class_to_images.json')

# Ảnh kiểm tra bằng mắt sau tiền xử lý (xem sanity_check_images)
SANITY_CHECK_PATH = os.path.join(os.path.dirname(LOCAL_BASE_DIR), 'sanity_check.png')


def set_seed(seed=SEED):
    """Cố định seed để tái lập kết quả — bắt buộc phải gọi hàm này
    ở đầu MỌI notebook trong dự án (chuẩn bị dữ liệu, train, eval)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 1. GẮN DRIVE (chỉ cần 1 lần, không lặp lại nhiều lần trong file)
# ============================================================

def mount_drive():
    """Gắn Google Drive khi chạy trên Colab. Ở máy cá nhân (không có
    google.colab) thì bỏ qua — khi đó dữ liệu lấy theo DOAN_DATA_DIR."""
    try:
        from google.colab import drive
    except ImportError:
        print("Không chạy trên Colab, bỏ qua bước gắn Drive.")
        return
    drive.mount('/content/drive')


# ============================================================
# 2. XÁC ĐỊNH BASE / VALIDATION / NOVEL CLASSES
# ============================================================

def get_base_and_validation_classes():
    """Base = Cà chua + Khoai tây (train feature extractor)
    Validation = Ớt chuông (tune hyperparameter, KHÔNG train)"""
    all_pv_classes = sorted(os.listdir(PLANTVILLAGE_DIR))
    tomato_classes = [c for c in all_pv_classes if c.startswith('Tomato')]
    potato_classes = [c for c in all_pv_classes if c.startswith('Potato')]
    pepper_classes = [c for c in all_pv_classes if c.startswith('Pepper')]

    base_classes = tomato_classes + potato_classes
    validation_classes = pepper_classes
    return base_classes, validation_classes


def get_novel_classes():
    """Novel = 6 lớp bệnh lá sầu riêng (Durian) — dùng để đánh giá few-shot"""
    return sorted(os.listdir(os.path.join(DURIAN_DIR, 'train')))


# ============================================================
# 3. GOM ẢNH MỖI LỚP — dùng 1 hàm chung cho cả PlantVillage và Durian
#    (thay vì 2 hàm gần giống nhau như bản cũ)
# ============================================================

VALID_EXT = ('.jpg', '.jpeg', '.png')


def build_class_to_images(class_name, dataset='plantvillage'):
    """Trả về danh sách đường dẫn ảnh của 1 lớp.
    dataset: 'plantvillage' (1 thư mục phẳng mỗi lớp)
             'durian' (gộp cả train/val/test vì dùng cho few-shot eval,
                       không dùng train/val/test kiểu classification thường)
    """
    images = []
    if dataset == 'plantvillage':
        folder = os.path.join(PLANTVILLAGE_DIR, class_name)
        if os.path.exists(folder):
            images = [os.path.join(folder, f) for f in os.listdir(folder)
                      if f.lower().endswith(VALID_EXT)]
    elif dataset == 'durian':
        for split in ['train', 'val', 'test']:
            folder = os.path.join(DURIAN_DIR, split, class_name)
            if os.path.exists(folder):
                images += [os.path.join(folder, f) for f in os.listdir(folder)
                           if f.lower().endswith(VALID_EXT)]
    else:
        raise ValueError(f"dataset không hợp lệ: {dataset}")
    return images


def check_no_class_overlap(base_classes, validation_classes, novel_classes):
    """Kiểm tra chủ động 3 tập lớp không được trùng nhau — đúng yêu cầu
    checklist Giai đoạn 0. Nếu sau này ai đó sửa nhầm logic lọc lớp
    (ví dụ đổi tên lớp, thêm lớp mới), lỗi sẽ bị chặn ngay tại đây thay
    vì âm thầm gây sai lệch kết quả few-shot về sau."""
    base_set = set(base_classes)
    val_set = set(validation_classes)
    novel_set = set(novel_classes)

    overlap_bv = base_set & val_set
    overlap_bn = base_set & novel_set
    overlap_vn = val_set & novel_set

    assert not overlap_bv, f"Trùng lớp giữa base và validation: {overlap_bv}"
    assert not overlap_bn, f"Trùng lớp giữa base và novel: {overlap_bn}"
    assert not overlap_vn, f"Trùng lớp giữa validation và novel: {overlap_vn}"
    print("Kiểm tra OK: base / validation / novel không trùng lớp nào.")


def build_all_class_to_images():
    """Gom toàn bộ base + validation + novel vào 1 dict, lưu cache lại
    thành file JSON để dùng lại nhanh ở các bước sau (episode sampling, v.v.)"""
    base_classes, validation_classes = get_base_and_validation_classes()
    novel_classes = get_novel_classes()

    check_no_class_overlap(base_classes, validation_classes, novel_classes)

    class_to_images = {}
    for c in base_classes:
        class_to_images[c] = build_class_to_images(c, dataset='plantvillage')
    for c in validation_classes:
        class_to_images[c] = build_class_to_images(c, dataset='plantvillage')
    for c in novel_classes:
        class_to_images[c] = build_class_to_images(c, dataset='durian')

    meta = {
        'base_classes': base_classes,
        'validation_classes': validation_classes,
        'novel_classes': novel_classes,
        'class_to_images': class_to_images,
    }

    with open(CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta


# ============================================================
# 3b. LỌC ẢNH HỎNG — quét 1 lần để tránh pipeline crash giữa chừng
#     (ví dụ đang train dở nhiều giờ mà gặp 1 file ảnh lỗi)
# ============================================================

def filter_corrupt_images(meta, verbose=True):
    """Thử mở từng ảnh 1 lần (Image.verify()), loại khỏi danh sách những
    ảnh không mở được. Chỉ cần chạy 1 lần ở Giai đoạn 2 — kết quả đã lọc
    sẽ được lưu lại vào cache JSON, các bước train/eval sau này không cần
    quét lại. Lưu ý: bước này khá chậm với PlantVillage (~vài chục nghìn
    ảnh), có thể mất vài phút — bù lại giúp tránh crash lúc train."""
    removed = []
    for c, paths in meta['class_to_images'].items():
        valid_paths = []
        for p in paths:
            try:
                with Image.open(p) as img:
                    img.verify()
                valid_paths.append(p)
            except Exception as e:
                removed.append((c, p, str(e)))
        meta['class_to_images'][c] = valid_paths

    if verbose:
        print(f"Đã quét xong toàn bộ ảnh. Số ảnh hỏng bị loại: {len(removed)}")
        for c, p, err in removed[:20]:
            print(f"  - [{c}] {p} ({err})")
        if len(removed) > 20:
            print(f"  ... còn {len(removed) - 20} ảnh hỏng khác, không in hết.")

    # Lưu lại cache đã lọc sạch, để lần chạy sau không cần quét lại
    with open(CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta


# ============================================================
# 4. KIỂM TRA SỐ LƯỢNG ẢNH / MẤT CÂN BẰNG LỚP
#    -> quan trọng để chọn K hợp lý (K phải nhỏ hơn lớp ít ảnh nhất)
# ============================================================

def report_class_counts(meta):
    print(f"{'Lớp':40s} | {'Nhóm':12s} | Số ảnh")
    print("-" * 65)

    group_of = {}
    for c in meta['base_classes']:
        group_of[c] = 'base'
    for c in meta['validation_classes']:
        group_of[c] = 'validation'
    for c in meta['novel_classes']:
        group_of[c] = 'novel'

    counts = {c: len(imgs) for c, imgs in meta['class_to_images'].items()}

    for c, n in sorted(counts.items(), key=lambda x: x[1]):
        print(f"{c:40s} | {group_of[c]:12s} | {n}")

    min_count = min(counts.values())
    print("-" * 65)
    print(f"Số ảnh ít nhất trong 1 lớp: {min_count}")
    if min_count < (K_SHOT + Q_QUERY):
        print(
            f"CẢNH BÁO: có lớp chỉ có {min_count} ảnh, nhỏ hơn "
            f"K+Q={K_SHOT + Q_QUERY}. Lớp này sẽ không đủ ảnh để tạo "
            f"episode {K_SHOT}-shot / {Q_QUERY}-query. Cần giảm K hoặc Q, "
            f"hoặc loại lớp này khỏi tập novel/base."
        )


# ============================================================
# 5. TIỀN XỬ LÝ ẢNH — phần còn thiếu ở bản cũ
# ============================================================

# Chuẩn hoá theo thống kê ImageNet: MobileNetV2 (nhánh Fine-tuning) được
# pretrained với đúng thống kê này; Conv-4 (nhánh ProtoNet) train từ đầu
# nên dùng chung luôn để hai nhánh nhận cùng một đầu vào.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=10),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Tập eval/test KHÔNG augment — chỉ resize + normalize, để kết quả ổn định
eval_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def load_and_preprocess(image_path, transform=eval_transform):
    """Đọc 1 ảnh từ đường dẫn, trả về tensor đã tiền xử lý.
    Báo lỗi rõ ràng kèm đường dẫn nếu ảnh không đọc được, thay vì để
    traceback gốc khó hiểu của PIL/torchvision hiện ra giữa lúc train."""
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        raise RuntimeError(
            f"Không đọc được ảnh: {image_path} ({e}). "
            f"Chạy lại filter_corrupt_images(meta) ở data_prep.py để loại "
            f"ảnh hỏng này khỏi danh sách trước khi train/eval tiếp."
        ) from e
    return transform(img)


def denormalize_for_display(tensor):
    """Đảo ngược normalize để hiển thị ảnh đúng màu khi kiểm tra bằng mắt."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


# ============================================================
# 6. KIỂM TRA BẰNG MẮT — bắt buộc theo checklist Giai đoạn 2
#    (ảnh không bị méo, không bị đen)
# ============================================================

def sanity_check_images(meta, n_classes=3, n_images=3):
    """Chọn ngẫu nhiên vài lớp, vài ảnh mỗi lớp, vẽ ra để kiểm tra bằng mắt
    cả ảnh gốc và ảnh sau khi qua train_transform."""
    sample_classes = random.sample(list(meta['class_to_images'].keys()), n_classes)

    fig, axes = plt.subplots(n_classes, n_images, figsize=(3 * n_images, 3 * n_classes))
    if n_classes == 1:
        axes = [axes]

    for i, c in enumerate(sample_classes):
        img_paths = meta['class_to_images'][c]
        chosen = random.sample(img_paths, min(n_images, len(img_paths)))
        for j, path in enumerate(chosen):
            tensor = load_and_preprocess(path, transform=train_transform)
            img_show = denormalize_for_display(tensor).permute(1, 2, 0).numpy()
            ax = axes[i][j] if n_classes > 1 else axes[j]
            ax.imshow(img_show)
            ax.set_title(c[:20], fontsize=8)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(SANITY_CHECK_PATH)
    plt.show()
    print(f"Đã lưu ảnh kiểm tra tại {SANITY_CHECK_PATH} — xem thử có bị "
          "méo, bị đen, hay lệch màu không trước khi coi bước này là xong.")


# ============================================================
# 7. CHẠY TOÀN BỘ (main)
# ============================================================

if __name__ == '__main__':
    set_seed()
    mount_drive()
    extract_datasets_to_local()

    print("Đang gom danh sách ảnh mỗi lớp...")
    meta = build_all_class_to_images()

    print("\n=== Lọc ảnh hỏng (chạy 1 lần, có thể mất vài phút) ===")
    meta = filter_corrupt_images(meta)

    print("\n=== Số lượng ảnh mỗi lớp ===")
    report_class_counts(meta)

    print("\n=== Kiểm tra ảnh sau tiền xử lý ===")
    sanity_check_images(meta)

    print(f"\nĐã lưu file cache tại: {CACHE_PATH}")
    print("Giai đoạn 2 hoàn thành nếu: không có lỗi, không có cảnh báo "
          "thiếu ảnh nghiêm trọng, và ảnh trong sanity_check.png bình thường.")
