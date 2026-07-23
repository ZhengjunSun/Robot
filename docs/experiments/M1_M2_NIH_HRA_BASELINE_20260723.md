# M1/M2 NIH/HRA 粗对准基线（2026-07-23）

## 资产与场景

- NIH 条目：3DPX-020963；
- 模型：HRA Visible Human female right eye v1.2；
- 许可证：CC-BY；
- 署名：Kristen Browne; Heidi Schlehlein. 2023. “3D Reference Organ for Eye,
  Female, Right v1.2.” DOI: 10.48539/HBM745.RCGM.944；
- 使用范围：解剖外观和视觉背景，不作为软组织物性、碰撞精度或临床验证模型；
- 戳卡：外径 2.0 mm、内径 1.0 mm、壁长 2.5 mm、法兰外径 2.64 mm。

场景入口：
`3d_modeling/mujoco/single_arm_trocar_visual_alignment_nih_hra.xml`。

## 控制与真值隔离

在线控制链为：

`eye-in-hand RGB → 传统/YOLO 检测 → CoarseObservation → 有界 IBVS 动作`

MuJoCo 目标位姿和毫米误差只写入评估报告。几何 segmentation 只用于离线生成 YOLO
标签，不作为控制器输入。M1/M2 为了隔离目标视觉，在 eye-in-hand 感知视图中隐藏器械
轴杆的视觉组；轴杆仍保留于世界视图和碰撞模型。后续完整相机外参和器械共视问题属于
M3/M4。

## M1 传统视觉结果

标准回合：

| 指标 | 结果 |
| --- | ---: |
| 初始横向误差 | 8.602 mm |
| 控制步数 | 7 |
| 最终像素误差 | 6.262 px |
| 最终横向误差（评价真值） | 0.466 mm |
| 停止原因 | 进入 FINE，等待精对准 provider |

随机 100 回合，初始相机偏置在每轴 ±7 mm 内，seed=20260723：

| 指标 | 结果 |
| --- | ---: |
| 成功率 | 100/100 |
| 平均步数 | 4.630 |
| p95 步数 | 7.050 |
| 平均最终像素误差 | 4.942 px |
| p95 最终像素误差 | 7.466 px |
| 平均最终横向误差 | 0.856 mm |
| p95 最终横向误差 | 1.303 mm |
| 平均路径 | 5.233 mm |
| 平均折返 | 0 |
| 目标丢失帧 | 0 |

## M2 YOLO 结果

旧项目权重在 NIH/HRA 新场景的零样本推理中没有检测到戳卡，因此没有沿用旧权重
成绩。当前模型使用 MuJoCo 特权 segmentation 生成的 240/60/60 合成 split；测试集
不参与训练或验证。

YOLO11n 在 CPU 上训练 5 轮后验证指标已收敛，保留当时的最佳 checkpoint：

| 独立合成测试指标 | 结果 |
| --- | ---: |
| Precision | 0.9981 |
| Recall | 1.0000 |
| mAP50 | 0.9950 |
| mAP50-95 | 0.8981 |
| CPU 推理时间（当次环境） | 约 212 ms/image |

标准在线闭环：

| 指标 | 结果 |
| --- | ---: |
| 控制步数 | 7 |
| 最终像素误差 | 5.629 px |
| 最终横向误差（评价真值） | 0.448 mm |
| 停止原因 | 进入 FINE，等待精对准 provider |

同 seed、同 10 个初始偏置的小样本在线对照：

| 方法 | 成功率 | 平均步数 | 最终像素误差 | 最终横向误差 | 路径 | 折返 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 传统 | 100% | 4.500 | 4.608 px | 0.821 mm | 4.976 mm | 0 |
| YOLO | 100% | 4.400 | 4.654 px | 0.800 mm | 4.778 mm | 0 |

该 10 回合对照只能证明 YOLO 适配器已经进入在线 RGB 闭环，不能据此宣称 YOLO
优于传统方法。

## 复现

```powershell
python generate_nih_yolo_dataset.py
python train_nih_yolo.py --epochs 30
python run_mujoco_coarse_batch.py --episodes 100 --detector traditional
python run_mujoco_coarse_alignment.py `
  --detector yolo `
  --yolo-target-classes 0 `
  --yolo-model output/yolo_nih_hra_m2_training/yolo11n_nih_hra_trocar/weights/best.pt
```

权重和实验输出不提交到核心代码仓库。已验证 checkpoint 的大小、SHA-256、训练轮次和
测试指标记录在 `config/yolo_nih_hra_m2_model_manifest.json`。

## 结论与后续

M1 与 M2 粗对准第一版已经在 NIH/HRA 合成场景中实现。下一步是 M3：内外环亚像素
拟合、PnP/椭圆姿态、2.5D 视觉伺服和连续五帧三线合一。完成 M3 后才能把既有插入
控制器接入并生成可称为“视觉粗对准—细对准—安全插入”的完整视频。
