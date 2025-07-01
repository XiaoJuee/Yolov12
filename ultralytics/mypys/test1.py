from ultralytics import YOLO
model = YOLO(r'D:\homewrok\GITHUB\Yolov12\ultralytics\cfg\models\Attentions\add_attention.yaml').load("yolov12n.pt")
for i, layer in enumerate(model.model):
    if hasattr(layer, 'c1'):
        print(f"Layer {i}: Input channels={layer.c1}")  # 检查LEGM的c1值