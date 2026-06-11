import torch
from ultralytics import YOLO
import gc

# Kiểm tra GPU
if torch.cuda.is_available():
    print(f"Đang sử dụng GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CẢNH BÁO: Không tìm thấy GPU, đang chạy bằng CPU!")

# Đường dẫn tới file cấu hình
DATA_YAML = '/mnt/data/users/quynhptit/huyptit/dental_yolo_project/datasets/dental.yaml'
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
    """Huấn luyện mô hình với Dental Augmentation tinh chỉnh"""
    print(f"\n========== BẮT ĐẦU TRAIN YOLO{model_version} AUGMENTED ==========")
    model = YOLO(f'yolo{model_version}.pt')
    
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        patience=PATIENCE,
        imgsz=IMGSZ,
        batch=batch_size,
        device=0,
        # --- BỘ AUGMENTATION NHA KHOA ĐÃ TINH CHỈNH ---
        degrees=10.0,           
        translate=0.1,          
        fliplr=0.5,             
        scale=0.2,              
        hsv_h=0.010,            
        hsv_s=0.3,              
        hsv_v=0.3,              
        mosaic=0.4,             
        close_mosaic=20,        
        copy_paste=0.05,        
        # ---------------------------------------------
        project=PROJECT_DIR,
        name=f'YOLO{model_version}_Augmented'
    )

# if __name__ == '__main__':
#     # 4090 có 24GB VRAM nên chạy batch size lớn thoải mái
#     BATCH_S = 32
#     BATCH_M = 16

#     print("🚀 BẮT ĐẦU CHUỖI THỰC NGHIỆM 4 MÔ HÌNH...")
    
#     # 1. Chạy 2 mô hình Baseline trước
#     train_baseline(model_version='v8s', batch_size=BATCH_S)
#     train_baseline(model_version='v8m', batch_size=BATCH_M)
    
#     # 2. Chạy 2 mô hình Augmented kế tiếp
#     train_augmented(model_version='v8s', batch_size=BATCH_S)
#     train_augmented(model_version='v8m', batch_size=BATCH_M)
    
#     print("\n🎉 ĐÃ HOÀN THÀNH TẤT CẢ CÁC THỰC NGHIỆM!")
if __name__ == '__main__':
    BATCH_S = 16
    BATCH_M = 8

    print("🚀 BẮT ĐẦU CHẠY TIẾP CÁC MÔ HÌNH CÒN LẠI...")
    
    # 1. Đã chạy xong, comment lại để bỏ qua
    # train_baseline(model_version='v8s', batch_size=BATCH_S)
    
    # --- MODEL 2 ---
    # train_baseline(model_version='v8m', batch_size=BATCH_M)
    
    # # Ép hệ thống xả RAM GPU trước khi chạy model tiếp theo
    # torch.cuda.empty_cache()
    # gc.collect()
    
    # --- MODEL 3 ---
    train_augmented(model_version='v8s', batch_size=BATCH_S)
    
    # Xả RAM tiếp
    torch.cuda.empty_cache()
    gc.collect()
    
    # --- MODEL 4 ---
    train_augmented(model_version='v8m', batch_size=BATCH_M)
    
    print("\n🎉 ĐÃ HOÀN THÀNH TẤT CẢ CÁC THỰC NGHIỆM!")