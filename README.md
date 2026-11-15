# Nhận dạng bệnh trên lá sầu riêng bằng học ít mẫu và học chuyển giao

Mã nguồn đồ án thực tập tốt nghiệp: so sánh **Prototypical Network** (few-shot,
backbone Conv-4 huấn luyện trên PlantVillage) với **Fine-tuning** (MobileNetV2
pretrained ImageNet, chỉ huấn luyện lại lớp Linear) trên 6 lớp lá/bệnh lá sầu
riêng của Durian Leaf Dataset, kèm ứng dụng minh hoạ Streamlit.

Kho mã nguồn: https://github.com/vugialinh905-wq/-n-TTTN

## Cấu trúc thư mục

```
data_prep.py                  Cấu hình chung, gom danh sách ảnh, tiền xử lý
episode_utils.py              Hạ tầng episode + đánh giá DÙNG CHUNG cho 2 nhánh
train_protonet.py             Huấn luyện và đánh giá Prototypical Network
train_baseline_finetune.py    Tune và đánh giá nhánh Fine-tuning đối chứng
compare_results.py            Gộp kết quả 2 nhánh thành bảng so sánh

multiseed_protonet.py         Huấn luyện ProtoNet với nhiều seed
confusion_analysis.py         Ma trận nhầm lẫn 5-way của cả 2 phương pháp
confusion_matrix_protonet.py  Ma trận nhầm lẫn 6-way của ProtoNet (chẩn đoán)
confusion_analysis_report_config.py         Ma trận nhầm lẫn theo cấu hình báo cáo
confusion_analysis_rngfix_report_config.py  (Fine-tuning 20 bước, lr 1e-3), checkpoint RNG-fix
feature_analysis.py           Few-shot trên đặc trưng đã lưu, xem ảnh đoán sai,
                              kiểm tra ảnh train/test gần trùng, so 3 giao thức
                              (gộp / train->test / tránh ảnh gần trùng: robust_grid)
domain_gap_diagnostics.py     Trần supervised và so sánh 2-way Pepper/Durian
audit_duplicates.py           Audit ảnh trùng chính xác SHA-256 -> exact_duplicate_audit*.csv
evaluate_clean_split.py       Đánh giá đã lọc cho cả 2 phương pháp (Bảng 5.1, Phụ lục G)
                              + ma trận nhầm lẫn 5-shot (PNG, CSV) + macro F1

build_prototypes.py           Tính sẵn prototype 6 lớp cho ứng dụng
build_support_set.py          Chép ảnh support thật cho nhánh Fine-tuning live
calibrate_threshold.py        Hiệu chỉnh ngưỡng tin cậy bằng Youden's J, cả 2 phương pháp
                              (ngoại tuyến; bản viết lại theo Mục 6.4, xem docstring)
calibrate_threshold_temperature.py   Bản cũ dùng temperature scaling (chỉ ProtoNet)
pick_demo_images.py           Chọn ảnh thử nhanh cho ứng dụng

demo/                         Ứng dụng minh hoạ, chạy độc lập trên máy cá nhân
  streamlit_app.py
  protonet_conv4_best.pt      Checkpoint Conv-4
  prototypes.json             6 vector prototype (+ ngưỡng đã hiệu chỉnh)
  support_set/                10 ảnh support mỗi lớp + manifest.json
results_snapshot/             File JSON kết quả của benchmark ban đầu
AI_LOG.md                     Nhật ký sử dụng công cụ AI
requirements.txt
```

Các file `.py` nằm phẳng ở thư mục gốc vì chúng import lẫn nhau
(`from data_prep import ...`); cần chạy từ đúng thư mục này.

## Chạy ứng dụng minh hoạ (máy cá nhân)

```
pip install -r requirements.txt
cd demo
streamlit run streamlit_app.py
```

Ứng dụng không cần dữ liệu huấn luyện. Lần chạy đầu cần mạng để torchvision
tải trọng số MobileNetV2 (khoảng 14 MB). Ứng dụng luôn trả về lớp có xác suất
cao nhất trong 6 lớp và không từ chối ảnh ngoài phân phối (OOD).

## Chạy lại thực nghiệm (Google Colab)

Dữ liệu đặt trong Drive tại `MyDrive/DoAn/data/`:

- `plantvillage/plantvillage-dataset.zip`
- `durian/A Durian Leaf Image Dataset of Common Diseases in Vietnam for Agricultural Diagnosis/Durian_Leaf_Diseases/{train,val,test}/<lớp>/`

Thứ tự chạy, mỗi bước cần bước trước đã xong:

| Bước | Lệnh | Kết quả |
|---|---|---|
| 1 | `python data_prep.py` | `data/class_to_images.json` |
| 2 | `python train_protonet.py` | `models/protonet_conv4_best.pt`, `results/protonet_{1,5,10}shot.json` |
| 3 | `python train_baseline_finetune.py` | `results/baseline_finetune_{1,5,10}shot.json`, `results/baseline_tuning_summary.json` |
| 4 | `python compare_results.py` | `results/comparison_table.json` |
| 5 | `python build_prototypes.py` rồi `python build_support_set.py` | `results/prototypes.json`, `support_set/` |
| 6 | `python calibrate_threshold.py` | thêm ngưỡng vào `results/prototypes.json` |
| 7 | `python audit_duplicates.py` | `results/exact_duplicate_audit_rerun.csv` |
| 8 | `python evaluate_clean_split.py` | `results/fewshot_clean_split_audit_rerun.json`, `confusion_*_clean_split_5shot_rerun.{png,csv}` |

Bước 7–8 là giao thức đánh giá đã lọc của báo cáo (Mục 5.1, Phụ lục G). Hai
script ghi file có hậu tố `_rerun`, không ghi đè kết quả đã có. Kết quả đúng
phải ra 11 cặp trùng (9 khác tập, 2 cùng tập, 5 khác nhãn) và accuracy trong
khoảng tin cậy của Bảng 5.1.

`calibrate_threshold.py` (bản viết lại) chạy trên dữ liệu thật với 5 mẫu ảnh khác
nhau cho ProtoNet ngưỡng 0,23-0,26 và J 0,31-0,42; Fine-tuning ngưỡng 0,60-0,74 và
J 0,05-0,10 (Bảng 6.4: 0,2502 / 0,4489 và 0,5359 / 0,1644). Kết luận định tính
(J của ProtoNet cao hơn Fine-tuning rõ rệt) giữ nguyên, nhưng giá trị J thay đổi theo
mẫu ảnh được rút nên không nên coi là hằng số.

Đã chạy lại trên dữ liệu Durian thật (Colab): audit ra đúng 11 cặp; ProtoNet
trùng Bảng 5.1 đến hai chữ số thập phân (29,95 / 39,87 / 44,02%, macro F1
5-shot 0,3973); Fine-tuning lệch dưới 0,3 điểm (29,52 / 47,24 / 54,02% so với
29,28 / 47,15 / 53,94%) do lớp Linear khởi tạo ngẫu nhiên mỗi episode.

`models/`, `results/`, `support_set/` được tạo ngang cấp với `data/`.

Các script phân tích dùng dưới dạng module trong notebook, cách gọi ghi ở đầu
mỗi file, ví dụ:

```python
import multiseed_protonet as ms
ms.run_all()            # 5 seed, bỏ qua seed đã có kết quả
```

`train_protonet.train()` từ chối ghi đè checkpoint đã có; dùng `tag='...'` để
lưu bản mới riêng hoặc `overwrite=True` nếu chắc chắn muốn ghi đè.

### Chạy ngoài Colab

Không cần sửa code, đặt 2 biến môi trường trước khi chạy:

| Biến | Ý nghĩa | Mặc định (Colab) |
|---|---|---|
| `DOAN_DATA_DIR` | Thư mục lưu trữ lâu dài, có cấu trúc như `MyDrive/DoAn/data/` ở trên | `/content/drive/MyDrive/DoAn/data` |
| `DOAN_LOCAL_DIR` | Thư mục làm việc chứa dữ liệu đã giải nén | `/content/data` |

`class_to_images.json` lưu đường dẫn tuyệt đối của ảnh, nên đổi máy thì phải
chạy lại `data_prep.py`.

## Cấu hình thực nghiệm

| Tham số | Giá trị |
|---|---|
| Kích thước ảnh | 84×84 |
| Episode | 5-way, K = 1 / 5 / 10 shot, 15 query mỗi lớp |
| Số episode đánh giá | 600, seed 42 |
| Base / validation / novel | Cà chua + Khoai tây / Ớt chuông (2 lớp) / 6 lớp Durian |
| ProtoNet | Conv-4, 60 epoch × 100 episode, Adam lr 1e-3, chọn checkpoint theo validation |
| Fine-tuning | MobileNetV2 đóng băng + Linear, tune `ft_steps` × `ft_lr` trên validation |

## Về `results_snapshot/`

Thư mục này là kết quả của **benchmark ban đầu**: support và
query rút từ toàn bộ ảnh mỗi lớp, ProtoNet huấn luyện trước khi sửa lỗi đặt lại
`random.seed` lúc validate.

| | 1-shot | 5-shot | 10-shot |
|---|---|---|---|
| Prototypical Network | 31,38% | 42,82% | 46,81% |
| Fine-tuning (MobileNetV2) | 28,94% | 46,47% | 53,61% |

Các số liệu về sau trong báo cáo (checkpoint RNG-fix, giao thức đã lọc theo
audit SHA-256, nhiều seed, ma trận nhầm lẫn) được lưu trên Google Drive của
nhóm, không nằm trong thư mục này.
