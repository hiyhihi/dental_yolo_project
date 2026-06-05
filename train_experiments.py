import torch
from ultralytics import YOLO

# Kiểm tra GPU
if torch.cuda.is_available():
    print(f"Đang sử dụng GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CẢNH BÁO: Không tìm thấy GPU, đang chạy bằng CPU!")

# Đường dẫn tới file cấu hình (đảm bảo file yaml đã setup đúng)
DATA_YAML = 'datasets/dental-decay-dp2024/decay_server.yaml'
PROJECT_DIR = 'runs/Dental_Detection'

# Cấu hình chung
EPOCHS = 300
PATIENCE = 50
IMGSZ = 960

def train_baseline(model_version, batch_size):
    """Huấn luyện mô hình Baseline (Không có Augmentation)"""
    print(f"\n========== BẮT ĐẦU TRAIN YOLO{model_version} BASELINE ==========")
    model = YOLO(f'yolo{model_version}.pt')
    
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        patience=PATIENCE,
        imgsz=IMGSZ,
        batch=batch_size,
        device=0,
        project=PROJECT_DIR,
        name=f'YOLO{model_version}_Baseline'
    )

def train_augmented(model_version, batch_size):
    """Huấn luyện mô hình với Dental Augmentation"""
    print(f"\n========== BẮT ĐẦU TRAIN YOLO{model_version} AUGMENTED ==========")
    model = YOLO(f'yolo{model_version}.pt')
    
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        patience=PATIENCE,
        imgsz=IMGSZ,
        batch=batch_size,
        device=0,
        # Các thông số Augmentation
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, 
        degrees=10.0, scale=0.5, mosaic=1.0, copy_paste=0.1,
        project=PROJECT_DIR,
        name=f'YOLO{model_version}_Augmented'
    )

if __name__ == '__main__':
    # THIẾT LẬP BATCH SIZE CHO RTX 4090 (24GB VRAM)
    # 4090 dư sức gánh batch lớn hơn Kaggle, giúp hội tụ nhanh hơn
    BATCH_S = 32
    BATCH_M = 16

    # COMMENT HOẶC UNCOMMENT CÁC DÒNG DƯỚI ĐÂY ĐỂ CHỌN MÔ HÌNH MUỐN CHẠY
    
    # 1. Thực nghiệm Baseline
    train_baseline(model_version='v8s', batch_size=BATCH_S)
    train_baseline(model_version='v8m', batch_size=BATCH_M)
    
    # 2. Thực nghiệm Augmentation
    train_augmented(model_version='v8s', batch_size=BATCH_S)
    train_augmented(model_version='v8m', batch_size=BATCH_M)
    
    print("\n🎉 ĐÃ HOÀN THÀNH TẤT CẢ CÁC THỰC NGHIỆM!")