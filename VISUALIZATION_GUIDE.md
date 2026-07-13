# ActiveSGM 可视化运行指南

## 快速开始

### 方法 1：使用默认参数运行（推荐）
```bash
cd /home/chen/Desktop/ActiveSGM
bash run_visualization.sh
```

### 方法 2：自定义参数运行
```bash
bash run_visualization.sh [scene] [exp] [gpus] [enable_vis]
```

**参数说明：**
- `scene`: 场景名称，默认 `office0`
  - 可选值: `office0`, `office1`, `office2`, `office3`, `office4`, `room_0`, `apartment_0` 等
- `exp`: 实验配置，默认 `ActiveSem`
  - 可选值: `ActiveSem`, `ActiveGS`, `ActiveLang`, `predefine`, `sgsslam` 等
- `gpus`: GPU 设备 ID，默认 `0,1`
  - 示例: `0` (单 GPU), `0,1` (双 GPU)
- `enable_vis`: 启用可视化，默认 `1`
  - `1` = 启用可视化窗口
  - `0` = 禁用可视化窗口（仅保存文件）

### 常用命令示例

**运行 office0 场景的 ActiveSem 配置（启用可视化）：**
```bash
bash run_visualization.sh office0 ActiveSem 0,1 1
```

**运行 office1 场景的 ActiveGS 配置：**
```bash
bash run_visualization.sh office1 ActiveGS 0,1 1
```

**仅使用 GPU 0：**
```bash
bash run_visualization.sh office0 ActiveSem 0 1
```

**禁用可视化窗口（仅保存结果）：**
```bash
bash run_visualization.sh office0 ActiveSem 0,1 0
```

## 数据集信息

你的系统使用 **Replica 数据集**，包含以下场景：
- office_0, office_1, office_2, office_3, office_4
- apartment_0, apartment_1, apartment_2
- frl_apartment_0 到 frl_apartment_5
- hotel_0
- room_0

数据位置：`/home/chen/Desktop/ActiveSGM/data/replica_v1_local/`

## 输出文件位置

运行完成后，结果保存在：
```
results/Replica/{scene}/{exp}/run_0/
```

包含内容：
- `visualization/rgbd/` - RGB-D 可视化图像
- `visualization/rendered_rgbd/` - 渲染的 RGB-D 图像
- `visualization/pose/` - 相机位姿
- `visualization/state/` - 规划器状态
- 最终的 3D 重建网格和检查点

## 故障排除

### 问题 1：找不到配置文件
确保场景和实验配置存在：
```bash
ls configs/Replica/office0/
```

### 问题 2：GPU 内存不足
尝试使用单个 GPU：
```bash
bash run_visualization.sh office0 ActiveSem 0 1
```

### 问题 3：可视化窗口不显示
- 如果在远程服务器，需要 X11 转发
- 或者禁用可视化窗口，改为保存文件：
```bash
bash run_visualization.sh office0 ActiveSem 0,1 0
```

## 直接在 VS Code 终端运行

1. 在 VS Code 中打开终端（Ctrl+`）
2. 输入命令：
```bash
cd /home/chen/Desktop/ActiveSGM
bash run_visualization.sh office0 ActiveSem 0,1 1
```
3. 按 Enter 运行

或者直接运行 Python 命令：
```bash
python src/main/sgm_launcher.py --dataset Replica --scene office0 --exp ActiveSem --gpus 0,1 --enable_vis 1
```
