# -*- coding: utf-8 -*-
"""
calibrate_threshold.py
Đo và hiệu chỉnh NGƯỠNG TIN CẬY bằng chỉ số Youden's J cho cả hai phương pháp
(Mục 6.4, Bảng 6.4 của báo cáo).

Với mỗi phương pháp, độ tin cậy = xác suất cao nhất sau softmax. Hai nhóm ảnh:
  - known:   ảnh lá sầu riêng thật, KHÔNG trùng ảnh dùng làm support / reference,
             tối đa N_KNOWN_PER_CLASS ảnh mỗi lớp;
  - unknown: ảnh không phải lá sầu riêng, lấy từ các lớp base + validation của
             PlantVillage (cà chua, khoai tây, ớt chuông), N_UNKNOWN_PER_CLASS ảnh mỗi lớp.
Quét mọi giá trị ngưỡng, tính TPR (tỉ lệ known >= ngưỡng) và FPR (tỉ lệ unknown >=
ngưỡng), chọn ngưỡng cho J = TPR - FPR lớn nhất.

  - ProtoNet: softmax của khoảng cách Euclidean ÂM tới 6 prototype (đúng như đoạn
    mã ở Mục 6.4.2, dùng torch.cdist, tức khoảng cách KHÔNG bình phương).
  - Fine-tuning: fine-tune lớp Linear MỘT lần trên bộ support set (10 ảnh/lớp),
    20 bước, lr 1e-3, rồi lấy softmax của logits.

LƯU Ý
  * Đây là phép đo NGOẠI TUYẾN: ngưỡng chỉ lưu để tham khảo, ứng dụng
    demo/streamlit_app.py không đọc ngưỡng này và không từ chối ảnh ngoài phân phối.
  * ProtoNet trong ứng dụng dùng khoảng cách BÌNH PHƯƠNG, còn bước đo này dùng
    khoảng cách không bình phương như báo cáo; hai thang xác suất khác nhau nên ngưỡng
    đo ở đây không áp trực tiếp được lên xác suất hiển thị trong ứng dụng.
  * Số lượng ảnh mỗi nhóm không được ghi trong báo cáo (mặc định bên dưới là giả định
    hợp lý), và ảnh được rút ngẫu nhiên, nên J và ngưỡng thu được có thể lệch so với
    Bảng 6.4 (ProtoNet 0,2502 / J 0,4489; Fine-tuning 0,5359 / J 0,1644). Lớp Linear
    khởi tạo ngẫu nhiên nên J của Fine-tuning dao động giữa các lần chạy.

Chạy trong Colab, sau build_prototypes.py và build_support_set.py:
    python calibrate_threshold.py
Mặc định KHÔNG ghi đè prototypes.json: kết quả ghi vào
results/confidence_calibration_rerun.json (cùng cấu trúc khối "confidence_threshold").
Phiên bản dùng temperature scaling cũ được giữ ở calibrate_threshold_temperature.py.
"""

import os
import json
import random

import numpy as np
import torch
import torch.nn.functional as F

from data_prep import set_seed, SEED, eval_transform, load_and_preprocess
from episode_utils import load_meta, DEVICE, MODELS_DIR, RESULTS_DIR
from train_protonet import Conv4Backbone
from train_baseline_finetune import build_pretrained_backbone, finetune_on_episode
from build_prototypes import pick_reference_images, N_REFERENCE_PER_CLASS
from build_support_set import pick_support_images, N_SUPPORT_PER_CLASS

N_KNOWN_PER_CLASS = 15
N_UNKNOWN_PER_CLASS = 15
FT_STEPS = 20
FT_LR = 1e-3

PROTONET_CKPT = os.path.join(MODELS_DIR, 'protonet_conv4_best.pt')
PROTO_PATH = os.path.join(RESULTS_DIR, 'prototypes.json')
OUT_PATH = os.path.join(RESULTS_DIR, 'confidence_calibration_rerun.json')


def best_threshold(known_conf, unknown_conf):
    """Ngưỡng tối đa hoá Youden's J = TPR - FPR."""
    known, unknown = np.array(known_conf), np.array(unknown_conf)
    best_t, best_j = 0.5, -1.0
    for t in sorted(set(known_conf) | set(unknown_conf)):
        j = float(np.mean(known >= t) - np.mean(unknown >= t))
        if j > best_j:
            best_j, best_t = j, t
    return float(best_t), float(best_j)


def get_protonet_confidences(model, prototypes, image_paths):
    confs = []
    with torch.no_grad():
        for p in image_paths:
            img = load_and_preprocess(p, transform=eval_transform).unsqueeze(0).to(DEVICE)
            emb = model(img)
            dists = torch.cdist(emb, prototypes)
            probs = F.softmax(-dists, dim=1).squeeze(0)
            confs.append(float(probs.max()))
    return confs


def get_finetune_confidences(base_model, support_paths, novel_classes, image_paths):
    """Fine-tune lớp Linear một lần trên bộ support set, rồi đo độ tin cậy."""
    sx, sy = [], []
    for label, c in enumerate(novel_classes):
        for p in support_paths[c]:
            sx.append(load_and_preprocess(p, transform=eval_transform))
            sy.append(label)
    sx = torch.stack(sx).to(DEVICE)
    sy = torch.tensor(sy).to(DEVICE)
    clf = finetune_on_episode(base_model, sx, sy, len(novel_classes),
                              ft_steps=FT_STEPS, ft_lr=FT_LR)
    confs = []
    with torch.no_grad():
        for p in image_paths:
            img = load_and_preprocess(p, transform=eval_transform).unsqueeze(0).to(DEVICE)
            probs = F.softmax(clf(base_model(img)), dim=1).squeeze(0)
            confs.append(float(probs.max()))
    return confs


def main(out_path=OUT_PATH):
    set_seed(SEED)
    meta = load_meta()
    cti = meta['class_to_images']
    novel = meta['novel_classes']

    # Ảnh dùng làm reference (prototype) và support (fine-tune) -> loại khỏi nhóm known
    reference = pick_reference_images(cti, novel, N_REFERENCE_PER_CLASS)
    support = pick_support_images(cti, novel, N_SUPPORT_PER_CLASS)
    used = {p for c in novel for p in reference[c]} | {p for c in novel for p in support[c]}

    rng = random.Random(SEED)
    known = []
    for c in novel:
        pool = [p for p in cti[c] if p not in used]
        known += rng.sample(pool, min(N_KNOWN_PER_CLASS, len(pool)))
    unknown = []
    for c in meta['base_classes'] + meta['validation_classes']:
        unknown += rng.sample(cti[c], min(N_UNKNOWN_PER_CLASS, len(cti[c])))
    print(f"known: {len(known)} ảnh Durian (loại {len(used)} ảnh support/reference); "
          f"unknown: {len(unknown)} ảnh PlantVillage")

    # --- ProtoNet ---
    model = Conv4Backbone().to(DEVICE)
    model.load_state_dict(torch.load(PROTONET_CKPT, map_location=DEVICE))
    model.eval()
    with open(PROTO_PATH, 'r', encoding='utf-8') as f:
        proto_data = json.load(f)
    prototypes = torch.tensor(
        [proto_data['protonet_prototypes'][c] for c in novel]).to(DEVICE)
    pn_t, pn_j = best_threshold(get_protonet_confidences(model, prototypes, known),
                                get_protonet_confidences(model, prototypes, unknown))

    # --- Fine-tuning ---
    base = build_pretrained_backbone()
    ft_t, ft_j = best_threshold(get_finetune_confidences(base, support, novel, known),
                                get_finetune_confidences(base, support, novel, unknown))

    print(f"ProtoNet   : ngưỡng {pn_t:.4f}, Youden's J = {pn_j:.4f}   (báo cáo: 0,2502 / 0,4489)")
    print(f"Fine-tuning: ngưỡng {ft_t:.4f}, Youden's J = {ft_j:.4f}   (báo cáo: 0,5359 / 0,1644)")

    result = {
        'confidence_threshold': {
            'protonet': round(pn_t, 4), 'protonet_youden_j': round(pn_j, 4),
            'finetune': round(ft_t, 4), 'finetune_youden_j': round(ft_j, 4),
            'finetune_note': (
                "Ngưỡng đo 1 lần trên 1 lần fine-tune cụ thể -- vì finetune_on_episode() "
                "khởi tạo trọng số ngẫu nhiên, độ tin cậy thực tế lúc demo có thể lệch so "
                "với lúc calibrate. J thấp nghĩa là ngưỡng này không đáng tin cậy."),
        },
        'n_known': len(known), 'n_unknown': len(unknown), 'seed': SEED,
        'ft_steps': FT_STEPS, 'ft_lr': FT_LR,
        'protonet_checkpoint': os.path.basename(PROTONET_CKPT),
        'protonet_confidence': 'softmax(-torch.cdist(embedding, prototype))',
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu: {out_path}  (không ghi đè prototypes.json)")
    return result


if __name__ == '__main__':
    main()
