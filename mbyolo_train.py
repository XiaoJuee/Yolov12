from ultralytics import YOLO
import argparse
import os
import torch


def parse_opt():
    parser = argparse.ArgumentParser()
    # 数据集配置
    parser.add_argument('--data', type=str, default='D:\homewrok\GITHUB\Yolov12\Car\data.yaml', help='dataset.yaml path')
    # parser.add_argument('--data', type=str, default='/root/autodl-tmp/Mamba-YOLO-main/ultralytics/cfg/datasets/car.yaml', help='dataset.yaml path')
    # 模型配置（使用此参数替代硬编码）
    # parser.add_argument('--model', type=str, default='/root/autodl-tmp/Mamba-YOLO-main/ultralytics/cfg/models/mamba-yolo/yolo-mamba-seg.yaml', help='model config path')
    # parser.add_argument('--model', type=str, default='/root/autodl-tmp/Mamba-YOLO-main/ultralytics/cfg/models/v8/yolov8-seg.yaml', help='model config path')
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='image size')
    parser.add_argument('--task', default='train', choices=['train', 'val', 'test'], help='task type')
    parser.add_argument('--device', default=0, help='cuda device')
    parser.add_argument('--workers', type=int, default=32, help='number of workers')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--optimizer', default='SGD', choices=['SGD', 'Adam', 'AdamW'], help='optimizer')
    # 结果保存配置（关键修改）
    parser.add_argument('--project', default='runs/output_dir', help='output directory')
    parser.add_argument('--name', default='mambayolo_run', help='experiment name')  # 修改默认名称
    parser.add_argument('--exist_ok', action='store_true', help='overwrite existing dir')  # 新增参数

    # 可选参数
    parser.add_argument('--amp', action='store_true', help='mixed precision')
    parser.add_argument('--half', action='store_true', help='FP16 inference')
    parser.add_argument('--dnn', action='store_true', help='OpenCV DNN')

    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_opt()

    # 统一参数配置（关键修改）
    train_args = {
        "data": opt.data,
        "epochs": opt.epochs,
        "workers": opt.workers,
        "batch": opt.batch_size,
        "imgsz": opt.imgsz,
        "optimizer": opt.optimizer,
        "device": opt.device,
        "amp": opt.amp,
        "project": opt.project,
        "name": opt.name,
        "exist_ok": opt.exist_ok,  # 防止目录重复创建
        "val": True,
    }

    # 根据任务类型执行
    # model = YOLO(opt.model)  # 使用命令行参数，并确保模型在 GPU 上
    # model = model.to(device)
    model = YOLO('yolov12n.pt')

    if opt.task == 'train':
        # 训练模式：关闭训练过程中的验证
        results = model.train(**train_args)

        # 训练完成后单独执行最终验证
        val_args = train_args.copy()
        val_args.update({
            "task": "val",
            "model": f"{opt.project}/{opt.name}/weights/best.pt"
        })
        model.val(**val_args)
    else:
        # 直接执行验证/测试
        eval_args = train_args.copy()
        eval_args.update({"task": opt.task})
        if opt.task == 'val':
            model.val(**eval_args)
        elif opt.task == 'test':
            model.test(**eval_args)