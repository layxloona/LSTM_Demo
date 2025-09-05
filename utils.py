import torch

def r_squared(y_true, y_pred):
    y_true_mean = torch.mean(y_true)
    ss_total = torch.sum((y_true - y_true_mean) ** 2)
    ss_residual = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    return r2.item()

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"模型已保存至：{path}")