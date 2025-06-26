import torch
print(torch.cuda.is_available())  # 应返回True
print(torch.cuda.get_device_name(0))  # 应显示"NVIDIA GeForce RTX 3050 Laptop GPU"