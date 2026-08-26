# AI Development Log

## Phiên 1 — Chuẩn bị dữ liệu (data_prep.py)
- Công cụ: Claude (Anthropic)
- Ngày: [điền ngày thực tế]
- Mục tiêu: Xây pipeline đọc PlantVillage + Durian, chia base/validation/novel classes
- File liên quan: data_prep.py

## Phiên 2 — Hạ tầng episode & Prototypical Network
- Công cụ: Claude (Anthropic)
- Ngày: [điền ngày thực tế]
- Mục tiêu: Viết episode_utils.py (hạ tầng dùng chung), train_protonet.py
  (Conv-4 backbone, episodic training), train_baseline_finetune.py
  (baseline đối chứng)
- File liên quan: episode_utils.py, train_protonet.py, train_baseline_finetune.py
- Lỗi phát sinh & tự sửa: quick_validate() dùng N_WAY=5 mặc định trong
  khi validation_classes (Pepper) chỉ có 2 lớp -> lỗi "Sample larger
  than population". Tự sửa bằng val_n_way = min(N_WAY, len(validation_classes))

## Phiên 3 — Debug môi trường Colab/Drive
- Công cụ: Claude (Anthropic)
- Ngày: [điền ngày thực tế]
- Mục tiêu: Xử lý lỗi giải nén PlantVillage thiếu lớp, lỗi Drive
  shortcut, lỗi tham số check_valid không tồn tại
