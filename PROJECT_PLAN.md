# Mini Project cá nhân: E-ELAN Classification và Object Detection

> **Quyết định kiến trúc:** Mô hình chính là mô hình tự xây dựng gồm
> `EELANLite backbone + Neck + Detection Head`. YOLOv7 chỉ dùng làm tài liệu
> tham khảo để giải thích E-ELAN; YOLOv7/YOLOv8 chỉ là baseline tùy chọn nếu
> còn thời gian, không thay thế phần tự xây dựng.

## 0. Phạm vi và mục tiêu

Project được thực hiện một mình, gồm hai sản phẩm chính:

1. **Classification**
   - Phân tích `EELANLite`.
   - Huấn luyện mô hình phân loại `cat/dog/panda`.
   - Save/load weights.
   - Inference ảnh đơn và thư mục.
   - Tính accuracy, precision, recall, F1-score.
   - Vẽ loss curve, accuracy curve và confusion matrix.

2. **Object detection**
   - Dùng E-ELAN làm backbone.
   - Tự xây dựng một Neck đơn giản/FPN.
   - Tự xây dựng Detection Head một-stage.
   - Huấn luyện trên dataset có bounding box.
   - Inference, NMS và đánh giá IoU, precision, recall, mAP@0.5.

Nguyên tắc thực hiện:

```text
Chạy được bản gốc
    -> hoàn thiện classification
    -> kiểm tra dataset detection
    -> làm detector một-scale với Neck và Head
    -> overfit tập nhỏ
    -> hoàn thiện inference và metrics
    -> đánh giá và viết báo cáo
```

Không cố sao chép toàn bộ YOLOv7. Mục tiêu là chứng minh đầy đủ pipeline:

```text
Backbone + Neck + Head + Loss + Inference + Evaluation
```

---

## 1. Đầu ra cuối cùng cần nộp

### 1.1. Mã nguồn/notebook

- [ ] Notebook classification hoàn chỉnh.
- [ ] Notebook detection hoàn chỉnh.
- [ ] Code chạy được từ đầu đến cuối hoặc có hướng dẫn chạy rõ ràng.
- [ ] Các đường dẫn dữ liệu được cấu hình tập trung, không rải rác trong code.
- [ ] File text hướng dẫn nguồn dữ liệu, cấu hình và cách chạy.

### 1.2. Kết quả classification

- [ ] File weights tốt nhất.
- [ ] Kết quả inference ảnh đơn.
- [ ] Kết quả inference thư mục.
- [ ] Classification report.
- [ ] Confusion matrix.
- [ ] Loss/accuracy curves.
- [ ] File history dạng JSON hoặc CSV.

Baseline classification đã được giữ nguyên tại
`classification/outputs/cls_eelan`. Bản huấn luyện cải thiện được cấu hình
riêng trong `CNN_ELAN_Class_split.ipynb` và sẽ ghi vào
`classification/outputs/cls_eelan_improved`, dùng 40 epochs, learning rate
`1e-3`, weight decay `0.01`, label smoothing `0.05`, và EELANLite lớn hơn
(`C_stem=48`, `C_stage=128`). Không ghi đè checkpoint baseline khi so sánh.

### 1.3. Kết quả object detection

- [ ] Dataset có bounding box.
- [ ] Ảnh kiểm tra annotation.
- [ ] File detection weights.
- [ ] Một số ảnh inference có bounding box.
- [ ] Bảng IoU, precision, recall, mAP@0.5.
- [ ] Phân tích ảnh dự đoán đúng/sai.

### 1.4. Báo cáo

- [ ] Mô tả bài toán.
- [ ] Phân tích lý thuyết E-ELAN.
- [ ] So sánh lý thuyết với code `EELANLite`.
- [ ] Mô tả dataset và preprocessing.
- [ ] Mô tả classification.
- [ ] Mô tả detection backbone, neck, head và loss.
- [ ] Bảng kết quả và biểu đồ.
- [ ] Hạn chế và hướng phát triển.
- [ ] Ghi rõ: `Project được thực hiện cá nhân`.
- [ ] Không đưa code vào báo cáo PDF; code nằm trong notebook nộp riêng.

---

## 2. Cấu trúc thư mục làm việc và cách chuẩn bị dữ liệu

### 2.1. Tạo thư mục project

Trong `D:\SPCV\miniproject`, tạo cấu trúc sau. Có thể tạo thủ công bằng File Explorer hoặc PowerShell:

```powershell
New-Item -ItemType Directory -Force `
  classification\weights, classification\outputs, `
  detection\weights, detection\outputs\annotation_samples, detection\outputs\predictions, `
  datasets\classification, datasets\detection, report\figures
```

Sau khi tạo, cấu trúc cần có:

```text
D:\SPCV\miniproject\
├── PROJECT_PLAN.md
├── classification\
│   ├── classification.ipynb
│   ├── weights\
│   └── outputs\
├── detection\
│   ├── detection.ipynb
│   ├── weights\
│   └── outputs\
│       ├── annotation_samples\
│       └── predictions\
├── datasets\
│   ├── classification\
│   └── detection\
└── report\
    └── figures\
```

Sao chép notebook classification mẫu vào:

```text
D:\SPCV\miniproject\classification\classification.ipynb
```

Không sửa trực tiếp file đính kèm read-only.

### 2.2. Cấu trúc dữ liệu classification

Giữ bộ dữ liệu đã thực hành theo cấu trúc:

```text
datasets\classification\
├── train\
│   ├── cats\
│   ├── dogs\
│   └── panda\
└── val\
    ├── cats\
    ├── dogs\
    └── panda\
```

Đây là dữ liệu dùng cho phần classification và transfer learning backbone.

### 2.3. Cấu trúc dữ liệu object detection

Sau khi tải và giải nén dataset detection, phải chuẩn hóa thành:

```text
datasets\detection\
├── images\
│   ├── train\
│   ├── val\
│   └── test\
├── labels\
│   ├── train\
│   ├── val\
│   └── test\
└── data.yaml
```

Mỗi ảnh phải có file label cùng tên:

```text
images\train\image_001.jpg
labels\train\image_001.txt
```

Mỗi dòng trong file `.txt` có dạng YOLO:

```text
class_id x_center y_center width height
```

Tọa độ phải được chuẩn hóa về `[0, 1]`.

### 2.4. Tải dataset detection đúng yêu cầu đề bài

Đề bài yêu cầu dữ liệu object detection có bounding box cho các đối tượng cat, dog và panda. Không dùng bộ classification hiện tại cho detection nếu bộ đó chỉ có tên class mà không có bounding box.

#### Phương án dữ liệu đúng với đề bài

Đề bài gợi ý tách dữ liệu từ nhiều nguồn:

```text
Cat, dog: COCO hoặc Pascal VOC
Panda: Roboflow Universe hoặc V3Det
```

Bạn có thể chọn một dataset Roboflow đã có đủ ba class để giảm công việc.
Nếu không tìm được dataset đủ tốt, dùng phương án ghép nguồn: lọc cat/dog từ
COCO hoặc Pascal VOC, lấy panda từ Roboflow Universe hoặc V3Det, rồi chuyển
tất cả annotation về YOLO TXT trước khi gộp.

Mô hình tự xây dựng không phụ thuộc YOLOv7/YOLOv8. YOLO TXT chỉ được chọn làm
format trung gian vì đơn giản và dễ đọc bằng `DetectionDataset`.

#### Tải dữ liệu panda từ Roboflow Universe

1. Mở [Roboflow Universe](https://universe.roboflow.com/).
2. Tìm:

   ```text
   cat dog panda object detection
   panda object detection
   cat dog object detection
   ```

3. Chọn dataset có **bounding box**, không chọn dataset chỉ có classification.
4. Kiểm tra danh sách class. Dataset panda chỉ cần có class panda; nếu dataset
   đã có đủ cat/dog/panda thì có thể dùng làm nguồn duy nhất.

   ```text
   cat, dog, panda
   ```

5. Kiểm tra số lượng ảnh của từng class, chất lượng box và license.
6. Chọn **Download Dataset**.
7. Chọn format YOLO có label dạng:

   ```text
   class_id x_center y_center width height
   ```

8. Chọn download ZIP, giải nén tạm vào:

   ```text
   D:\SPCV\miniproject\datasets\detection_download
   ```

9. Kiểm tra dataset trước khi di chuyển vào `datasets\detection`.

Hướng dẫn chính thức:

- [Roboflow Universe](https://universe.roboflow.com/)
- [Roboflow: Download a Universe Dataset](https://docs.roboflow.com/datasets/universe/universe/download-a-universe-dataset)
- [Roboflow: Annotation Formats](https://roboflow.com/formats)

#### Tải cat/dog từ COCO hoặc Pascal VOC khi cần ghép nguồn

Nếu dùng COCO:

1. Tải ảnh và annotation COCO theo hướng dẫn chính thức.
2. Lọc category `cat` và `dog`.
3. Chuyển bbox COCO từ:

   ```text
   x_min, y_min, width, height
   ```

   sang YOLO:

   ```text
   class_id, x_center, y_center, width, height
   ```

4. Gán `0=cat`, `1=dog`.

Nếu dùng Pascal VOC:

1. Tải ảnh và XML annotation.
2. Đọc `object/name` và `bndbox`.
3. Giữ `cat` và `dog`.
4. Chuyển XML sang YOLO TXT.
5. Gán `0=cat`, `1=dog`.

Không trộn trực tiếp COCO JSON hoặc Pascal VOC XML với YOLO TXT. Phải chuyển
tất cả về YOLO TXT trước khi gộp với dữ liệu panda.

#### Nếu không có một dataset đủ cả ba class

Có thể kết hợp hai hoặc ba nguồn:

```text
Nguồn A: cat và dog detection
Nguồn B: panda detection
```

Khi gộp, bắt buộc:

- [ ] Đổi tên class về đúng `cat`, `dog`, `panda`.
- [ ] Đổi class ID về đúng mapping:

  ```text
  0 = cat
  1 = dog
  2 = panda
  ```

- [ ] Chuyển toàn bộ annotation về YOLO format.
- [ ] Không trộn nhầm train/val/test.
- [ ] Kiểm tra license của từng nguồn.
- [ ] Loại ảnh trùng hoặc gần như trùng.

Không tự ghép các file label trước khi kiểm tra mapping class của từng nguồn.

### 2.5. Kiểm tra dataset sau khi tải

Trước khi train, kiểm tra đủ các điều sau:

- [ ] Có thư mục `images/train`, `images/val`, `images/test`.
- [ ] Có thư mục `labels/train`, `labels/val`, `labels/test`.
- [ ] Mỗi ảnh có label tương ứng, trừ ảnh thực sự không có object.
- [ ] Các class ID chỉ thuộc `{0, 1, 2}`.
- [ ] Tọa độ label nằm trong `[0, 1]`.
- [ ] `width` và `height` lớn hơn 0.
- [ ] Vẽ thử tối thiểu 20 ảnh và box đúng vật thể.
- [ ] Số lượng ảnh giữa các class không quá mất cân bằng.
- [ ] Không có data leakage giữa train, val và test.

Nếu dataset tải về có tên thư mục khác, hãy đổi/copy dữ liệu về đúng cấu trúc ở mục 2.3; không sửa mapping bằng phỏng đoán.

### 2.6. Tạo `data.yaml`

Tạo file:

```text
D:\SPCV\miniproject\datasets\detection\data.yaml
```

Nội dung mẫu:

```yaml
path: D:/SPCV/miniproject/datasets/detection
train: images/train
val: images/val
test: images/test
names:
  0: cat
  1: dog
  2: panda
```

Trong YAML có thể dùng dấu `/` cho đường dẫn Windows. Trong Python, dùng raw string hoặc `os.path.join`.

### 2.7. Tạo file ghi nguồn dữ liệu

Tạo `datasets\detection\DATA_SOURCE.txt` và ghi:

```text
Tên dataset:
URL:
Ngày tải:
Phiên bản:
Định dạng:
Số ảnh train/val/test:
Class mapping:
License:
Các bước tiền xử lý:
```

File này giúp chứng minh nguồn dữ liệu và quá trình chuẩn hóa trong báo cáo.

---

## 3. Giai đoạn 1 - Chuẩn bị môi trường và baseline

### 3.1. Kiểm tra và cài đặt môi trường

Thực hiện trong PowerShell tại thư mục project:

```powershell
Set-Location D:\SPCV\miniproject
```

#### Bước 1: Kiểm tra Python

```powershell
py --version
```

Nếu lệnh `py` không tồn tại, thử:

```powershell
python --version
```

Ghi lại phiên bản Python. Nên dùng Python 3.10 hoặc 3.11 nếu phiên bản PyTorch hiện tại chưa hỗ trợ tốt phiên bản mới hơn.

#### Bước 2: Tạo môi trường ảo

Nếu dùng Python Launcher:

```powershell
py -3.11 -m venv .venv
```

Kích hoạt:

```powershell
.\.venv\Scripts\Activate.ps1
```

Nếu PowerShell chặn script, chỉ trong cửa sổ hiện tại chạy:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Sau khi kích hoạt, kiểm tra:

```powershell
python --version
python -c "import sys; print(sys.executable)"
```

#### Bước 3: Cài PyTorch/torchvision

Không nên đoán lệnh cài CUDA. Truy cập [PyTorch Start Locally](https://pytorch.org/get-started/locally/) và chọn:

```text
OS: Windows
Package: Pip
Language: Python
Compute Platform: CUDA phù hợp với GPU, hoặc CPU
```

Chạy đúng lệnh trang PyTorch cung cấp. Ví dụ CPU:

```powershell
python -m pip install torch torchvision
```

Nếu dùng NVIDIA GPU, chọn đúng lệnh CUDA từ trang chính thức, sau đó kiểm tra:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

#### Bước 4: Cài thư viện project

```powershell
python -m pip install numpy pillow matplotlib scikit-learn tqdm pyyaml
```

Nếu dùng notebook:

```powershell
python -m pip install jupyter ipykernel
python -m ipykernel install --user --name miniproject --display-name "Python (miniproject)"
```

Trong VS Code, chọn kernel:

```text
Python (miniproject)
```

#### Bước 5: Kiểm tra import

Chạy:

```powershell
python -c "import torch, torchvision, numpy, PIL, matplotlib, sklearn, yaml; print('Environment OK')"
```

Nếu lệnh lỗi, dừng tại đây và sửa môi trường trước khi chạy notebook.

#### Bước 6: Ghi thông tin môi trường

Chạy:

```powershell
python -c "import sys, torch; print('Python:', sys.version); print('PyTorch:', torch.__version__); print('Torchvision:', __import__('torchvision').__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
```

Ghi kết quả vào báo cáo, mục `Môi trường thực nghiệm`.

#### Bước 7: Kiểm tra GPU và bộ nhớ

Nếu dùng NVIDIA GPU:

```powershell
nvidia-smi
```

Nếu không có `nvidia-smi` hoặc không có GPU, dùng CPU và giảm:

```text
batch_size
num_workers
image_size
```

#### Bước 8: Test một forward pass

Trong notebook hoặc Python:

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(2, 3, 224, 224, device=device)
print(device, x.shape)
```

Sau khi định nghĩa model, chạy thêm:

```python
model = EELANClassifier(num_classes=3).to(device)
y = model(x)
print(y.shape)  # kỳ vọng: torch.Size([2, 3])
```

### 3.1.1. Tiêu chí hoàn thành mục 3.1

- [ ] Có môi trường `.venv`.
- [ ] Notebook dùng đúng kernel `.venv`/`miniproject`.
- [ ] Import tất cả thư viện thành công.
- [ ] Biết đang chạy CPU hay GPU.
- [ ] Model classification forward không lỗi.
- [ ] Ghi lại phiên bản môi trường vào báo cáo.

Chỉ chuyển sang mục 3.2 và chạy baseline sau khi tất cả checklist trên đạt.

### 3.2. Kiểm tra dữ liệu classification

Dataset hiện tại nên có dạng:

```text
datasets/classification/
├── train/
│   ├── cats/
│   ├── dogs/
│   └── panda/
└── val/
    ├── cats/
    ├── dogs/
    └── panda/
```

Kiểm tra:

- [ ] Số ảnh trong mỗi class.
- [ ] Ảnh bị lỗi hoặc không đọc được.
- [ ] Tên class giữa train và validation.
- [ ] Mapping class cố định.

Mapping nên thống nhất:

```text
0 = cats
1 = dogs
2 = panda
```

Nếu tự động split validation, dùng `cfg["seed"]`, không dùng biến global `CFG["seed"]`.

### 3.3. Chạy code gốc

Trước khi sửa, chạy notebook mẫu và lưu:

- [ ] Train loss từng epoch.
- [ ] Validation loss từng epoch.
- [ ] Train accuracy.
- [ ] Validation accuracy.
- [ ] Epoch có kết quả tốt nhất.
- [ ] Thời gian train.

Kết quả tham khảo của notebook:

```text
Best validation accuracy: khoảng 80.33%
```

Đây là baseline để so sánh sau khi bổ sung chức năng.

---

## 4. Giai đoạn 2 - Phân tích E-ELAN trước khi chỉnh sửa

Đọc và hiểu các class theo thứ tự:

```text
ConvBNAct
channel_shuffle
EELANLite
EELANClassifier
```

### 4.1. Phân tích `ConvBNAct`

Giải thích chuỗi:

```text
Conv2d -> BatchNorm2d -> SiLU
```

Nêu vai trò của:

- Kernel.
- Stride.
- Padding.
- Group convolution.
- BatchNorm.
- Activation SiLU.

### 4.2. Phân tích channel expansion

Đoạn code:

```python
C_exp = int(round(self.m * self.C_in))
C_exp = max(4*self.g, (C_exp // (4*self.g)) * (4*self.g))
```

Với cấu hình hiện tại:

```text
C_in = 96
m = 2
g = 2
C_exp = 192
split_ch = 192 / 4 = 48
```

Giải thích `C_exp` phải chia hết cho 4 để chia branch và chia hết cho `g` để group convolution hợp lệ.

### 4.3. Phân tích branch

```python
branch_depths = [0, 1, 2, 3]
```

Ý nghĩa:

- Branch 1 không có convolution bổ sung.
- Branch 2 có một block 3x3.
- Branch 3 có hai block 3x3.
- Branch 4 có ba block 3x3.

Các branch có độ sâu khác nhau giúp tạo feature ở nhiều mức xử lý.

### 4.4. Phân tích shuffle, concat, fuse và residual

Trình bày:

```text
Group conv
    -> channel shuffle
    -> concat các branch
    -> 1x1 fuse
    -> residual nếu shape phù hợp
```

### 4.5. So sánh lý thuyết với YOLOv7

Phải nêu rõ:

- `EELANLite` là bản rút gọn minh họa ý tưởng.
- YOLOv7 có backbone nhiều stage và feature map đa tỉ lệ.
- Code hiện tại chỉ có một stage E-ELAN chính.
- Classification dùng `GAP -> Linear`.
- Detection phải bỏ `GAP` và `Linear` để giữ thông tin vị trí `(x, y)`.
- YOLOv7 chỉ là cơ sở tham khảo lý thuyết, không phải mô hình chính bắt buộc.

Đầu ra cần lưu:

```text
report/figures/eelan_diagram.png
```

---

## 5. Giai đoạn 3 - Hoàn thiện classification

Làm xong toàn bộ giai đoạn này trước khi bắt đầu detection.

### 5.1. Tạo training history

Khởi tạo:

```python
history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": [],
}
```

Sau mỗi epoch:

- [ ] Append loss training.
- [ ] Append loss validation.
- [ ] Append accuracy training.
- [ ] Append accuracy validation.

Lưu:

```text
classification/outputs/history.json
```

### 5.2. Tạo checkpoint đầy đủ

Checkpoint nên chứa:

```python
{
    "epoch": epoch,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "scheduler_state": scheduler.state_dict() if scheduler else None,
    "scaler_state": scaler.state_dict() if scaler else None,
    "best_acc": best_acc,
    "classes": class_names,
    "cfg": cfg,
}
```

Lưu checkpoint tốt nhất khi `val_acc` cải thiện:

```text
classification/weights/classification_best.pt
```

### 5.3. Load checkpoint

Viết hàm load theo quy trình:

1. Nhận `weight_path`.
2. `torch.load(..., map_location=device)`.
3. Đọc `classes` và cấu hình model.
4. Tạo model với đúng số class.
5. Load `model_state`.
6. Gọi `model.eval()` khi inference.

Nếu resume training, load thêm:

- [ ] Optimizer state.
- [ ] Scheduler state.
- [ ] AMP scaler state.
- [ ] Epoch bắt đầu.
- [ ] Best accuracy cũ.

### 5.4. Kiểm tra save/load

- [ ] Dự đoán một ảnh trước khi giải phóng model.
- [ ] Tạo model mới.
- [ ] Load checkpoint.
- [ ] Dự đoán lại ảnh.
- [ ] So sánh class và probability.

Kết quả phải giống hoặc sai khác không đáng kể.

### 5.5. Inference ảnh đơn

Pipeline:

```text
Đọc ảnh
    -> convert RGB
    -> validation transform
    -> model
    -> softmax
    -> top-k
```

Output:

```text
Image: dog_001.jpg
1. dogs   probability=0.9342
2. cats   probability=0.0501
3. panda  probability=0.0157
```

Không in logits trực tiếp như probability; phải dùng `softmax`.

### 5.6. Inference thư mục không nhãn

- [ ] Duyệt `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`.
- [ ] Bỏ qua file không đọc được và ghi cảnh báo.
- [ ] In class top-1 và confidence.
- [ ] Lưu toàn bộ kết quả vào:

```text
classification/outputs/predictions.csv
```

Các cột:

```text
image,predicted_class,confidence
```

### 5.7. Inference dataset có nhãn

Dùng `ImageFolder` trên test set để thu thập:

```python
y_true
y_pred
probabilities
```

### 5.8. Metrics classification

Tính và lưu:

- [ ] Accuracy.
- [ ] Precision từng class.
- [ ] Recall từng class.
- [ ] F1-score từng class.
- [ ] Macro average.
- [ ] Weighted average.
- [ ] Confusion matrix.

Output:

```text
classification/outputs/classification_report.txt
classification/outputs/confusion_matrix.png
```

### 5.9. Vẽ biểu đồ

Vẽ hai biểu đồ:

1. Train loss và validation loss.
2. Train accuracy và validation accuracy.

Lưu:

```text
classification/outputs/training_curves.png
```

Phân tích:

- Loss có giảm đều không?
- Validation có dao động không?
- Train và validation có tách xa không?
- Có overfitting không?
- Class nào yếu nhất?

### 5.10. Mốc hoàn thành classification

Chỉ chuyển sang detection khi tất cả đúng:

- [ ] Notebook chạy được từ đầu đến cuối.
- [ ] Có checkpoint.
- [ ] Load checkpoint thành công.
- [ ] Inference ảnh đơn hoạt động.
- [ ] Inference thư mục hoạt động.
- [ ] Có probability.
- [ ] Có precision, recall, F1.
- [ ] Có confusion matrix.
- [ ] Có loss/accuracy curves.

---

## 6. Giai đoạn 4 - Chuẩn bị dataset object detection

### 6.1. Chọn dataset

Khuyến nghị:

```text
3 class: cat, dog, panda
150-300 ảnh mỗi class cho phiên bản đầu
```

Ưu tiên dataset có sẵn bounding box và export được YOLO format, ví dụ từ Roboflow Universe. Có thể kết hợp nhiều nguồn nhưng phải chuẩn hóa class và annotation.

Không dùng trực tiếp dataset classification nếu ảnh không có bounding box.

### 6.2. Cấu trúc dataset

```text
datasets/detection/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

Chia:

```text
70% train
20% validation
10% test
```

### 6.3. Class mapping

Giữ cố định:

```text
0 = cat
1 = dog
2 = panda
```

`data.yaml`:

```yaml
path: D:/SPCV/miniproject/datasets/detection
train: images/train
val: images/val
test: images/test
names:
  0: cat
  1: dog
  2: panda
```

### 6.4. YOLO label

Mỗi dòng:

```text
class_id x_center y_center width height
```

Các tọa độ nằm trong `[0, 1]`.

Ví dụ:

```text
1 0.512 0.476 0.340 0.590
```

### 6.5. Kiểm tra annotation

Viết code đọc ảnh và label, chuyển tọa độ YOLO sang pixel rồi vẽ box.

- [ ] Kiểm tra ngẫu nhiên tối thiểu 20-30 ảnh.
- [ ] Class hiển thị đúng.
- [ ] Box bao quanh đúng vật thể.
- [ ] Box không vượt biên ảnh.
- [ ] Không có ảnh thiếu label.
- [ ] Không có label thiếu ảnh.
- [ ] Không có tọa độ âm hoặc lớn hơn 1.
- [ ] Không có ảnh trùng giữa train/val/test.

Lưu một số ảnh mẫu:

```text
detection/outputs/annotation_samples/
```

Kết quả kiểm tra ngày 2026-10-10: đã sinh 30 ảnh mẫu (10 train, 10 val,
10 test) bằng `detection/annotation_check.py`. Kiểm tra trực quan phát hiện
một số annotation có dấu hiệu sai class, ví dụ ảnh panda bị gắn nhãn `cat`
và ảnh mèo bị gắn nhãn `panda`. Vì vậy bước này **chưa đạt**; cần rà soát
và làm sạch các annotation sai trước khi train detector. Không train detector
trên toàn bộ dataset trước khi hoàn tất việc này.

---

## 7. Giai đoạn 5 - Thiết kế detector tối thiểu

### 7.1. Tách E-ELAN thành backbone

Classification model hiện tại:

```text
Stem -> Stage1 -> EELAN -> Neck -> GAP -> Linear
```

Detection backbone cần:

```text
Stem -> Stage1 -> EELAN -> feature maps
```

Bỏ:

```text
GAP -> Linear
```

Mục tiêu là giữ tensor dạng:

```text
B x C x H x W
```

### 7.2. Làm bản detector một-scale trước

Kiến trúc phiên bản đầu:

```text
E-ELAN backbone
    -> 1x1/3x3 neck
    -> detection head
```

Với 3 class, mỗi grid cell dự đoán:

```text
4 box values + 1 objectness + 3 class scores = 8 channels
```

Kiểm tra forward:

- [ ] Input batch chạy được.
- [ ] Output có đúng số channel.
- [ ] Không có NaN.
- [ ] Kích thước feature map đúng dự kiến.

### 7.3. Viết `DetectionDataset`

Mỗi sample:

```python
image, target
```

Target:

```python
target = {
    "boxes": Tensor[N, 4],
    "labels": Tensor[N],
    "image_id": image_id,
}
```

Dùng box dạng:

```text
x_min, y_min, x_max, y_max
```

Viết `collate_fn` riêng vì mỗi ảnh có số object khác nhau.

### 7.4. Gán ground truth vào grid

Với feature map `H x W`:

```python
grid_x = int(x_center * W)
grid_y = int(y_center * H)
```

Ô chứa tâm object:

```text
objectness = 1
class = class_id
box target = ground truth
```

Ô không có object:

```text
objectness = 0
```

### 7.5. Loss

```text
total_loss =
    lambda_box * box_loss
  + lambda_obj * objectness_loss
  + lambda_cls * classification_loss
```

Bản đầu:

- [ ] SmoothL1 cho box.
- [ ] BCEWithLogits cho objectness.
- [ ] CrossEntropy hoặc BCEWithLogits cho class.

Chỉ thay bằng GIoU/CIoU sau khi bản cơ bản chạy đúng.

---

## 8. Giai đoạn 6 - Debug detector trước khi train thật

### 8.1. Overfit 5-10 ảnh

Đây là bước bắt buộc:

- [ ] Chọn 5-10 ảnh.
- [ ] Train nhiều epoch.
- [ ] Loss giảm rõ rệt.
- [ ] Box bám ground truth.
- [ ] Confidence tăng.

Nếu không overfit được, kiểm tra theo thứ tự:

1. Label có đúng không?
2. Box chuyển tọa độ có đúng không?
3. `grid_x/grid_y` có nằm trong feature map không?
4. Shape output có đúng không?
5. Decode box có đúng không?
6. Các thành phần loss có giá trị hợp lý không?
7. Có nhầm class mapping không?

### 8.2. Test từng module

- [ ] Test `Dataset` bằng cách hiển thị một batch.
- [ ] Test backbone bằng cách in shape.
- [ ] Test head bằng một tensor giả.
- [ ] Test loss bằng output và target giả.
- [ ] Test decode với một box biết trước.
- [ ] Test NMS với hai box chồng nhau.

Chỉ train toàn bộ dataset sau khi các test này đạt.

---

## 9. Giai đoạn 7 - Hoàn thiện Neck và huấn luyện detector

### 9.1. Neck của mô hình chính

Neck bắt buộc của phiên bản nộp là một Neck một-scale gồm các convolution
`1x1` và `3x3` đặt giữa backbone và head:

```text
Backbone feature map
    -> Conv 1x1 điều chỉnh channel
    -> Conv 3x3 tinh chỉnh feature
    -> Detection Head
```

Nếu backbone trả về nhiều feature map, có thể dùng FPN/PAN-lite:

```text
P5 -> upsample -> merge P4
P4 -> upsample -> merge P3
```

FPN/PAN-lite kết hợp:

- Semantic mạnh ở tầng sâu.
- Thông tin vị trí tốt ở tầng nông.
- Khả năng phát hiện object nhiều kích thước.

Detector một-scale với Neck convolution là mô hình chính được nộp. Không bắt
buộc triển khai đầy đủ P3/P4/P5; điều bắt buộc là phải có Neck riêng, mô tả
được đầu vào/đầu ra và giải thích vai trò biến đổi/tinh chỉnh feature map.

### 9.2. Transfer learning từ classification

Nếu classification weights đã ổn:

1. Tạo detection backbone có layer tương ứng.
2. Load các layer matching từ classification checkpoint.
3. Bỏ qua classifier head.
4. Freeze backbone vài epoch đầu.
5. Train neck và head.
6. Unfreeze backbone.
7. Fine-tune toàn bộ với learning rate nhỏ hơn.

Kiểm tra số layer load được và ghi lại trong báo cáo.

### 9.3. Cấu hình huấn luyện ban đầu

```text
Image size: 320 hoặc 416
Batch size: tùy GPU
Epochs: 30-50
Backbone learning rate: 1e-4
Neck/head learning rate: 1e-3
Confidence threshold: 0.25
NMS IoU threshold: 0.5
```

### 9.4. Lưu best detection checkpoint

Checkpoint:

```python
{
    "model_state": model.state_dict(),
    "class_names": class_names,
    "model_config": model_config,
    "epoch": epoch,
    "val_map50": val_map50,
}
```

Lưu theo validation `mAP@0.5`:

```text
detection/weights/detection_best.pt
```

---

## 10. Giai đoạn 8 - Inference object detection

Pipeline:

```text
Ảnh
 -> model
 -> decode output
 -> sigmoid/softmax
 -> confidence filtering
 -> NMS
 -> vẽ box
 -> lưu ảnh
```

Confidence:

```python
confidence = objectness_probability * class_probability
```

Kết quả in:

```text
Image: image_001.jpg
Class: dog
Confidence: 0.87
Box: [xmin, ymin, xmax, ymax]
```

Lưu ảnh vào:

```text
detection/outputs/predictions/
```

Kiểm tra:

- [ ] Box nằm trong ảnh.
- [ ] Class đúng.
- [ ] Confidence nằm trong `[0, 1]`.
- [ ] NMS loại được box trùng.
- [ ] Ảnh không có prediction vẫn được xử lý.

---

## 11. Giai đoạn 9 - Đánh giá object detection

### 11.1. IoU

```text
IoU = diện tích giao / diện tích hợp
```

Một prediction được xem là đúng khi:

```text
Đúng class và IoU >= 0.5
```

### 11.2. TP, FP, FN

- `TP`: prediction khớp một ground truth chưa được match.
- `FP`: prediction sai class hoặc IoU dưới ngưỡng.
- `FN`: ground truth không có prediction phù hợp.

### 11.3. Metrics

Tính:

- [ ] Precision.
- [ ] Recall.
- [ ] AP từng class.
- [ ] mAP@0.5.
- [ ] IoU trung bình.

Lưu:

```text
detection/outputs/detection_metrics.json
```

Trong báo cáo cần nêu rõ ngưỡng confidence và IoU dùng để đánh giá.

### 11.4. Phân tích lỗi

Chọn một số ảnh:

- [ ] Dự đoán đúng.
- [ ] Bỏ sót object.
- [ ] Nhầm class.
- [ ] Box lệch.
- [ ] Nhiều box trùng.

Viết nguyên nhân có thể:

- Dataset ít.
- Class mất cân bằng.
- Object nhỏ hoặc bị che.
- Backbone chưa đủ sâu.
- Detector một-scale.
- Annotation chưa chính xác.

---

## 12. Timeline cá nhân đề xuất

### Ngày 1: Baseline

- [ ] Tạo thư mục project.
- [ ] Sao chép notebook mẫu.
- [ ] Kiểm tra môi trường.
- [ ] Kiểm tra dataset classification.
- [ ] Chạy code gốc.
- [ ] Lưu baseline.

### Ngày 2: Phân tích kiến trúc

- [ ] Đọc `ConvBNAct`.
- [ ] Đọc `channel_shuffle`.
- [ ] Đọc `EELANLite`.
- [ ] Vẽ sơ đồ E-ELAN.
- [ ] Viết phần so sánh lý thuyết/code.

### Ngày 3: Checkpoint

- [ ] Thêm history.
- [ ] Thêm save checkpoint.
- [ ] Thêm load checkpoint.
- [ ] Kiểm tra dự đoán trước/sau load.

### Ngày 4: Classification evaluation

- [ ] Inference ảnh đơn.
- [ ] Inference thư mục.
- [ ] Tính precision, recall, F1.
- [ ] Vẽ confusion matrix.
- [ ] Vẽ loss/accuracy.

### Ngày 5: Detection data

- [ ] Chọn dataset.
- [ ] Chuẩn hóa class mapping.
- [ ] Chia train/val/test.
- [ ] Kiểm tra và vẽ annotation.

### Ngày 6: Detector tối thiểu

- [ ] Tách backbone.
- [ ] Viết `DetectionDataset`.
- [ ] Viết collate function.
- [ ] Viết detection head một-scale.
- [ ] Kiểm tra shape.

### Ngày 7: Loss và debug

- [ ] Viết target assignment.
- [ ] Viết box/objectness/class loss.
- [ ] Overfit 5-10 ảnh.
- [ ] Sửa lỗi đến khi overfit được.

### Ngày 8: Train detector

- [ ] Train trên dataset đầy đủ.
- [ ] Load classification weights nếu phù hợp.
- [ ] Freeze rồi unfreeze backbone.
- [ ] Lưu best checkpoint.

### Ngày 9: Inference và metrics

- [ ] Decode prediction.
- [ ] Thêm confidence filtering.
- [ ] Thêm NMS.
- [ ] Vẽ bounding box.
- [ ] Tính IoU, precision, recall, mAP.

### Ngày 10: Báo cáo

- [ ] Chọn hình và bảng kết quả.
- [ ] Viết phân tích.
- [ ] Ghi hạn chế.
- [ ] Kiểm tra code từ đầu đến cuối.
- [ ] Kiểm tra toàn bộ file output.

Nếu có nhiều thời gian hơn, dùng ngày bổ sung để thêm FPN/multi-scale hoặc thí nghiệm so sánh.

---

## 13. Ưu tiên khi thiếu thời gian

### Bắt buộc

1. Classification baseline.
2. Save/load weights.
3. Inference classification.
4. Precision/recall/F1.
5. Loss/accuracy curves.
6. Dataset detection có annotation đúng.
7. E-ELAN backbone.
8. Detection head và loss.
9. Overfit được tập nhỏ.
10. Inference detection và ít nhất mAP@0.5.

### Nên có nếu còn thời gian

- [ ] FPN Neck.
- [ ] Transfer learning từ classification.
- [ ] Confusion matrix detection theo class.
- [ ] So sánh detector có/không pretrained.
- [ ] Chạy YOLOv7-tiny hoặc YOLOv8n làm baseline tham khảo.

### Có thể bỏ nếu quá tải

- [ ] Multi-scale đầy đủ P3/P4/P5.
- [ ] CIoU/GIoU phức tạp.
- [ ] So sánh nhiều backbone.
- [ ] Dataset quá lớn.

Không được bỏ bước kiểm tra annotation và overfit 5-10 ảnh.

---

## 14. File text hướng dẫn chạy và nguồn dữ liệu

Tạo file:

```text
README_EXPERIMENT.txt
```

File này là một phần bắt buộc khi nộp. Nội dung tối thiểu:

```text
1. Thông tin
- Họ tên:
- Mã sinh viên:
- Project thực hiện cá nhân

2. Môi trường
- Python:
- PyTorch:
- torchvision:
- CPU/GPU:
- CUDA:

3. Nguồn dữ liệu
- Classification dataset:
- Detection dataset:
- URL từng nguồn:
- Ngày tải:
- License:
- Số ảnh train/val/test:
- Số bounding box:

4. Class mapping
- 0 = cat
- 1 = dog
- 2 = panda

5. Kiến trúc
- Backbone: EELANLite từ code mẫu
- Neck: mô tả module đã chọn
- Head: mô tả module đã chọn
- Loss: box/objectness/classification

6. Cấu hình classification
- image size:
- batch size:
- epochs:
- learning rate:
- optimizer:
- checkpoint:

7. Cấu hình detection
- image size:
- batch size:
- epochs:
- confidence threshold:
- NMS IoU threshold:
- checkpoint:

8. Cách chạy
- Cách chạy notebook classification
- Cách chạy notebook detection
- Cách load weights
- Cách chạy inference ảnh đơn/thư mục

9. File kết quả
- Classification report:
- Classification curves:
- Detection metrics:
- Detection prediction folder:
- Prediction CSV/TXT:
```

File kết quả detection dạng CSV/TXT phải có tối thiểu:

```text
image, xmin, ymin, xmax, ymax, class, confidence
```

Người chấm phải có thể dựa vào file này để biết nguồn dữ liệu, cấu hình,
đường dẫn weights và cách chạy lại thực nghiệm.

---

## 15. Checklist trước khi nộp

### Classification

- [ ] Chạy notebook không lỗi từ đầu đến cuối.
- [ ] Có `classification_best.pt`.
- [ ] Load weights từ đường dẫn chỉ định.
- [ ] Inference ảnh đơn.
- [ ] Inference thư mục.
- [ ] In top-k class và probability.
- [ ] Có accuracy, precision, recall, F1.
- [ ] Có `classification_report.txt`.
- [ ] Có `confusion_matrix.png`.
- [ ] Có `training_curves.png`.

### Detection

- [ ] Dataset có bbox.
- [ ] Class mapping cố định.
- [ ] Annotation đã kiểm tra bằng hình ảnh.
- [ ] Có backbone E-ELAN.
- [ ] Có Neck.
- [ ] Có Detection Head.
- [ ] Có box/objectness/classification loss.
- [ ] Detector overfit được 5-10 ảnh.
- [ ] Có `detection_best.pt`.
- [ ] Có decode và NMS.
- [ ] Có ảnh prediction.
- [ ] Có IoU, precision, recall, mAP@0.5.
- [ ] Có CSV/TXT chứa image, BBox, class và confidence.

### Báo cáo

- [ ] Nêu rõ project làm cá nhân.
- [ ] Mô tả dữ liệu và cách chia tập.
- [ ] Mô tả E-ELAN.
- [ ] So sánh code với lý thuyết.
- [ ] Mô tả backbone, neck, head.
- [ ] Giải thích loss.
- [ ] Có bảng kết quả.
- [ ] Có biểu đồ.
- [ ] Có ảnh prediction.
- [ ] Phân tích lỗi và hạn chế.
- [ ] Không chèn code vào PDF báo cáo.

### File nộp bổ sung

- [ ] `README_EXPERIMENT.txt`.
- [ ] Classification weights.
- [ ] Detection weights.
- [ ] CSV/TXT kết quả detection.
- [ ] Các ảnh prediction.
- [ ] File source/license của dataset.

---

## 16. Các mốc dừng và kiểm tra

Không chuyển sang bước tiếp theo nếu mốc hiện tại chưa đạt:

1. **Mốc A:** Classification gốc chạy được.
2. **Mốc B:** Save/load classification cho kết quả nhất quán.
3. **Mốc C:** Classification có đầy đủ metrics và biểu đồ.
4. **Mốc D:** Annotation detection vẽ đúng.
5. **Mốc E:** Detector forward không lỗi.
6. **Mốc F:** Detector overfit được 5-10 ảnh.
7. **Mốc G:** Detector chạy trên validation/test.
8. **Mốc H:** Có ảnh prediction và mAP.
9. **Mốc I:** Báo cáo giải thích được toàn bộ pipeline.

Khi bị lỗi, quay về mốc gần nhất đã kiểm tra được; không tiếp tục thêm kiến trúc mới khi pipeline cơ bản chưa đúng.
