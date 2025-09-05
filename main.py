import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import LSTM
from data_loader import load_data
from utils import get_device, save_model
from trainer import LSTMTrainer
from visualizer import visualize_results


def main():
    # 初始化
    device = get_device()
    print(f"Using device: {device}")

    # 加载数据
    x_data, y_data = load_data('origin_data_new.txt')
    print(f"Loaded {len(x_data)} samples")

    # 初始化模型和训练器
    model = LSTM()
    trainer = LSTMTrainer(model, device)

    # 训练参数
    Epoch = 1
    nums = 0

    # 训练循环
    for k in range(Epoch):
        for i in range(len(x_data)):
            x_np = np.array(x_data[i], dtype='float32')
            y_np = np.array(y_data[i], dtype='float32')

            result = trainer.train_step(x_np, y_np, nums)

            print(f"|Epoch: {k} |step: {nums} |loss: {result['loss']:.4f} "
                  f"|MAE: {result['mae']:.4f} |R^2: {result['r2']:.4f}")

            # 可视化
            trainer.plot_progress()
            key = visualize_results(x_np[:32], result['prediction'])
            if key == 27:  # ESC键退出
                break

            nums += 1

    # 保存模型
    save_path = r"D:\openpose_master\openpose\photo_recognition_Demo\project_project1\bin\openpose_models\lstm_model.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_model(model, save_path)

    plt.show()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()