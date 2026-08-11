# E4 — ghi chú về `epoch_seconds` trong `training_history.csv`

## Vấn đề

`training_history.csv` của E4 ghi `epoch_seconds ≈ 4,120` giây/epoch, trong khi
E2 và E3 chỉ ≈ 493 giây/epoch — chênh **8.4 lần**.

## Đây KHÔNG phải đặc tính của sqrt-inverse weighting

E2, E3 và E4 có khối lượng tính toán **giống hệt nhau**. Kiểm chứng từ chính
các file config đã ghi:

| | E2 | E3 | E4 |
|---|---|---|---|
| `model` | phobert-base-v2 @ `fb76b7e1` | giống | giống |
| `max_length` | 192 | 192 | 192 |
| `train_batch_size` / `grad_accum` | 4 / 4 | 4 / 4 | 4 / 4 |
| `epochs` | 5 | 5 | 5 |
| `fp16_enabled` | true | true | true |
| số window train | 38,111 | 38,111 | 38,111 |
| **khác biệt duy nhất** | `weight_mode: none` | `weight_mode: inverse` | `weight_mode: sqrt_inverse` |

Class weight chỉ là một vector 4 phần tử nhân vào loss. Nó không thể làm
training chậm đi 8 lần.

## Nguyên nhân thật: thiếu VRAM do kernel cũ chưa tắt

E4 được chạy khi ba kernel Jupyter của E1, E2 và E3 **vẫn còn sống** và vẫn giữ
model + optimizer state + cache allocator của PyTorch trong VRAM. Trạng thái GPU
đo được lúc E4 đang chạy:

```
GPU     : NVIDIA GeForce RTX 4060 Laptop (8 GB)
VRAM    : 7,925 / 8,188 MiB   (96.8% — gần như đầy)
util    : 98–100%
power   : 30–41 W / 105 W     ← dấu hiệu quyết định
```

Utilization 100% nhưng công suất chỉ ~1/3 mức tối đa nghĩa là GPU đang **chờ dữ
liệu**, không phải đang tính. Khi VRAM đầy, driver Windows WDDM tự động tràn
sang shared system memory và chuyển qua lại qua PCIe — chậm hơn VRAM rất nhiều
lần.

## Ảnh hưởng tới kết quả

**Không có.** Thiếu VRAM chỉ làm chậm, không làm sai:

* thứ tự batch do seed 42 quyết định, không phụ thuộc tốc độ;
* phép toán vẫn là fp16/fp32 như E2 và E3;
* checkpoint E4 đã được nạp lại và **tái tạo đúng** điểm validation đã báo cáo
  (0.745676, sai lệch 1.7e-05 — xem
  `outputs/evaluation/validation_unweighted_loss.json`).

Vì vậy điểm số của E4 vẫn so sánh được bình thường với E2 và E3.

## Kết luận cho báo cáo

Khi trình bày thời gian huấn luyện, **không** được viết "E4 chậm hơn E2/E3 8
lần". Câu đúng là:

> Ba thí nghiệm PhoBERT có cùng chi phí tính toán (~493 giây/epoch trên RTX
> 4060 8GB). Cột `epoch_seconds` của E4 cao bất thường vì lần chạy đó bị thiếu
> VRAM do các kernel notebook trước chưa được tắt; đây là hiện tượng môi
> trường, không ảnh hưởng tới kết quả.

**Bài học vận hành:** tắt kernel của notebook trước khi mở notebook tiếp theo.
GPU 8 GB không chứa được hai PhoBERT cùng lúc.
