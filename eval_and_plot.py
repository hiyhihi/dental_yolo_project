import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO

# Cấu hình đường dẫn (khớp với train_experiments.py)
RUNS_DIR = 'runs/detect/runs/Dental_Detection'
DATA_YAML = '/mnt/data/users/quynhptit/huyptit/dental_yolo_project/datasets/dental.yaml'
MODELS = [
    'YOLOv8s_Baseline', 
    'YOLOv8m_Baseline', 
    'YOLOv8s_Augmented', 
    'YOLOv8m_Augmented'
]

def evaluate_all_models(split='val'):
    """
    Đánh giá lại toàn bộ 4 mô hình trên tập validation (hoặc test)
    và in kết quả mAP ra màn hình.
    """
    print(f"\n{'='*50}")
    print(f"BẮT ĐẦU ĐÁNH GIÁ CÁC MÔ HÌNH TRÊN TẬP '{split.upper()}'")
    print(f"{'='*50}\n")
    
    for model_name in MODELS:
        weights_path = os.path.join(RUNS_DIR, model_name, 'weights', 'best.pt')
        
        if not os.path.exists(weights_path):
            print(f"⚠️ Bỏ qua {model_name}: Không tìm thấy file trọng số tại {weights_path}")
            continue
            
        print(f"\n--- Đang đánh giá: {model_name} ---")
        try:
            model = YOLO(weights_path)
            # Chạy validation
            metrics = model.val(
                data=DATA_YAML,
                split=split,
                imgsz=960,
                conf=0.25,
                iou=0.6,
                verbose=False # Tắt log chi tiết để terminal gọn gàng hơn
            )
            
            map50 = metrics.box.map50
            map50_95 = metrics.box.map
            print(f"✅ {model_name} -> mAP@0.5: {map50:.4f} | mAP@0.5:0.95: {map50_95:.4f}")
        except Exception as e:
            print(f"❌ Lỗi khi đánh giá {model_name}: {e}")

def plot_loss_curves(save_path='loss_comparison.png'):
    """
    Vẽ biểu đồ so sánh Loss của các mô hình và lưu thành file ảnh.
    """
    print(f"\nĐang vẽ biểu đồ so sánh Loss...")
    plt.figure(figsize=(16, 6))
    
    # 1. Vẽ Training Box Loss
    plt.subplot(1, 2, 1)
    for run in MODELS:
        csv_file = os.path.join(RUNS_DIR, run, 'results.csv')
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            df.columns = df.columns.str.strip() # Dọn dẹp khoảng trắng ở tên cột
            if 'train/box_loss' in df.columns:
                plt.plot(df['epoch'], df['train/box_loss'], label=run)
    plt.title('Training Box Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Vẽ Validation mAP@0.5 (So sánh hiệu suất thực tế)
    plt.subplot(1, 2, 2)
    for run in MODELS:
        csv_file = os.path.join(RUNS_DIR, run, 'results.csv')
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            df.columns = df.columns.str.strip()
            # Cột mAP trong YOLOv8 thường có tên là 'metrics/mAP50(B)'
            map_col = [col for col in df.columns if 'mAP50' in col]
            if map_col:
                plt.plot(df['epoch'], df[map_col[0]], label=run)
    plt.title('Validation mAP@0.5')
    plt.xlabel('Epochs')
    plt.ylabel('mAP@0.5')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ Đã lưu biểu đồ so sánh tại: {save_path}")

def predict_and_save(image_path, model_name='YOLOv8s_Augmented', save_path='prediction_result.jpg'):
    """
    Dự đoán một ảnh và lưu ảnh kết quả (đã vẽ bounding box) về máy.
    """
    print(f"\nĐang chạy dự đoán ảnh bằng mô hình {model_name}...")
    weights_path = os.path.join(RUNS_DIR, model_name, 'weights', 'best.pt')
    
    if not os.path.exists(weights_path):
        print(f"❌ Không tìm thấy model tại {weights_path}")
        return
        
    if not os.path.exists(image_path):
        print(f"❌ Không tìm thấy ảnh tại {image_path}")
        return

    try:
        model = YOLO(weights_path)
        # Chạy predict
        results = model.predict(source=image_path, imgsz=960, conf=0.25)
        
        # Lấy ảnh kết quả (numpy array dạng BGR)
        res_img = results[0].plot()
        
        # Lưu ảnh bằng OpenCV
        cv2.imwrite(save_path, res_img)
        print(f"✅ Đã lưu ảnh dự đoán thành công tại: {save_path}")
        
    except Exception as e:
        print(f"❌ Lỗi khi dự đoán: {e}")

def plot_paper_style_curve(model_name):
    """
    Vẽ biểu đồ chuẩn Paper: Đường Loss và mAP trên cùng 1 hình (Trục Y kép).
    """
    print(f"\nĐang vẽ biểu đồ chuẩn Paper cho mô hình: {model_name}...")
    csv_file = os.path.join(RUNS_DIR, model_name, 'results.csv')
    
    if not os.path.exists(csv_file):
        print(f"⚠️ Không tìm thấy file kết quả của {model_name} tại {csv_file}")
        return
        
    # Đọc dữ liệu
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip() # Dọn dẹp khoảng trắng
    
    epochs = df['epoch']
    # Chọn cột Box Loss của tập Train (Bạn có thể đổi thành val/box_loss nếu muốn)
    if 'train/box_loss' not in df.columns:
        print("⚠️ Không tìm thấy cột 'train/box_loss' trong file kết quả.")
        return
    loss = df['train/box_loss']
    
    # Tìm cột mAP@0.5
    map_cols = [c for c in df.columns if 'mAP50' in c and '95' not in c]
    if not map_cols:
        print("⚠️ Không tìm thấy cột mAP trong file kết quả.")
        return
    map50 = df[map_cols[0]]
    
    # Khởi tạo đồ thị
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # --- Trục Y bên TRÁI (Dành cho Loss) ---
    color1 = '#ff7f0e' # Màu cam đặc trưng
    ax1.set_xlabel('Epochs', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Training Box Loss', color=color1, fontweight='bold', fontsize=12)
    line1 = ax1.plot(epochs, loss, color=color1, linewidth=2.5, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # --- Trục Y bên PHẢI (Dành cho mAP) ---
    ax2 = ax1.twinx()  # Lệnh tạo trục Y thứ 2 đối xứng
    color2 = '#1f77b4' # Màu xanh dương
    ax2.set_ylabel('mAP@0.5', color=color2, fontweight='bold', fontsize=12)
    line2 = ax2.plot(epochs, map50, color=color2, linewidth=2.5, label='mAP@0.5')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # --- Ghép chú thích (Legend) ---
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', fontsize=11)
    
    # Tiêu đề và căn lề
    plt.title(f'Training Progress - {model_name}', fontweight='bold', fontsize=14)
    fig.tight_layout()
    
    # Lưu file ảnh chất lượng cao
    save_path = f'{model_name}_paper_plot.png'
    plt.savefig(save_path, dpi=300)
    print(f"✅ Đã lưu biểu đồ thành công tại: {save_path}")

if __name__ == '__main__':

    # 1. Đánh giá tất cả models
    evaluate_all_models(split='test')
    
    # # 2. Vẽ và lưu biểu đồ Loss & mAP
    # plot_loss_curves(save_path='loss_comparison_plot.png')
    
    # for model_name in MODELS:
    #     plot_paper_style_curve(model_name)

    # 3. Test thử 1 ảnh 
    # test_image_path = 'datasets/val/images/image_2874.jpg' 
    # predict_and_save(
    #     image_path=test_image_path, 
    #     model_name='YOLOv8s_Baseline', # Chọn model tốt nhất để test
    #     save_path='sample_prediction.jpg'
    # )