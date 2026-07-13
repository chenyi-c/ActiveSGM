# ActiveSGM 快速启动指南 (RTX 4060 本地运行)

## ✅ 系统状态

系统已基本修复完毕，能够成功初始化并进入 mapping iteration。还有一个梯度形状问题需要解决（非阻塞性）。

---

## 运行命令

### 方法 1: 直接运行（推荐）


source /home/chen/miniconda3/etc/profile.d/conda.sh
conda activate activegamer
cd /home/chen/Desktop/ActiveSGM
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64,expandable_segments:True
python src/main/sgm_launcher.py --dataset Replica --scene office0 --exp ActiveSem4060Vis --gpus 0 --enable_vis 1








```bash
source /home/chen/miniconda3/etc/profile.d/conda.sh
conda activate activegamer
cd /home/chen/Desktop/ActiveSGM

# 基础运行（无可视化）
python src/main/activesgm.py \
  --cfg configs/Replica/office0/ActiveSem.py \
  --seed 0 \
  --result_dir results/Replica/office0/ActiveSem/run_0 \
  --enable_vis 0
```

### 方法 2: 使用启动脚本

```bash
cd /home/chen/Desktop/ActiveSGM
bash run_visualization.sh office0 ActiveSem 0 0
```

### 方法 3: 使用 tmux（后台运行，推荐用于长时间任务）

```bash
tmux new-session -d -s activesgm
tmux send-keys -t activesgm "source /home/chen/miniconda3/etc/profile.d/conda.sh && conda activate activegamer && python src/main/activesgm.py --cfg configs/Replica/office0/ActiveSem.py --seed 0 --result_dir results/Replica/office0/ActiveSem/run_0 --enable_vis 0" Enter

# 查看运行状态
tmux attach -t activesgm

# 退出 (Ctrl+B, D)
```

---

## 关键配置参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `desired_image_height` | 340 | 输入图像高度 |
| `desired_image_width` | 600 | 输入图像宽度 |
| `mapping_window_size` | 4 | 映射窗口（帧数） |
| `tracking_iters` | 5 | 跟踪迭代次数 |
| `mapping_iters` | 5 | 映射迭代次数 |
| `num_topk_logits` | 16 | 语义 channel 数量 |
| `map_every` | 5 | 每 N 帧进行一次映射 |

---

## 输出目录结构

```
results/Replica/office0/ActiveSem/run_0/
├── splatam/
│   ├── config.py                 # SLAM 配置复制
│   ├── params0.npz               # 初始高斯参数（可选）
│   └── params100.npz             # 第 100 步检查点
├── metrics.json                  # 评估指标
└── splatam_recon.ply             # 最终 3D 重建点云
```

---

## 预期运行时间

| 阶段 | 时间 | 说明 |
|------|------|------|
| 初始化 | ~30 秒 | 加载模型、数据集、编译 CUDA |
| Habitat 模拟 | ~2 秒/帧 | 单帧渲染 |
| SLAM（tracking） | ~5 秒/帧 | 相机跟踪（5 iter） |
| SLAM（mapping） | ~30 秒 | 每 5 帧 1 次，mapping_window_size=4 |
| **总计** | **~5-10 分钟/100 帧** | 取决于系统负载 |

---

## 性能监控

### 实时查看 GPU 使用

```bash
watch -n 1 nvidia-smi
```

预期显存占用: **6-7.5 GB** （RTX 4060 8GB 总量）

### 查看运行日志

```bash
# 如果在 tmux 中运行，直接查看即可
# 如果后台运行，使用：
tail -f results/Replica/office0/ActiveSem/run_0/log.txt
```

---

## 常见问题

### Q1: "CUDA out of memory"
**解决方案**:
```python
# 进一步降低参数：
mapping_window_size = 2
desired_image_height = 256
desired_image_width = 512
mapping_iters = 3
```

### Q2: "tensor dimension mismatch"
**解决方案**: 确保 habitat.py 和 ActiveSem.py 的分辨率一致
```bash
# 检查
grep "resolution_hw" configs/Replica/office0/habitat.py
grep "desired_image" configs/Replica/office0/ActiveSem.py
```

### Q3: 模型加载缓慢
**解决方案**: 这是正常的，首次加载 OneFormer 模型需要 2-3 分钟。模型会被缓存到：
```
~/.cache/huggingface/hub/
```

### Q4: 进程卡在 "Mapping Time Step"
**解决方案**: 
- 让其继续运行（可能在做 GPU 计算）
- 或按 `Ctrl+C` 中断并检查错误

---

## 实验评估

运行完成后，可以生成评估指标：

```bash
# 3D 重建评估
bash scripts/evaluation/eval_replica_3d.sh office0 1 ActiveSem 0 0

# 语义分割评估
bash scripts/evaluation/eval_replica_semantic.sh office0 1 ActiveSem 0 0 0 final

# 新视角合成评估
bash scripts/evaluation/eval_replica_nvs_result.sh office0 1 ActiveSem 0 0
```

---

## 下一步

### 如果想在服务器上运行
1. 参考 `SERVER_DEPLOYMENT_PLAN.md`
2. 建议租赁 RTX 4090 × 2 以加快训练
3. 预计成本：¥20-30 / 4-5 小时

### 如果想修改参数
1. 编辑 `configs/Replica/office0/ActiveSem.py`
2. 或新建场景配置文件
3. 重新运行命令

### 如果想进行实验
1. 尝试不同场景：office1, office2, room0, room1, room2
2. 调整 planner 策略（active_gsv2 vs predefined_traj）
3. 对比不同的 SLAM 参数

---

**最后更新**: 2026-04-14
**系统版本**: v0.1-rtc-4060-optimized
**维护者**: Your Name
