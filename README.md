# TwoArmRobot Core

眼科手术机械臂“插入前视觉对准 + 对准后安全插入”核心代码。

## 研究主线

```text
目标获取
  → 传统方法 / YOLO 粗对准
  → 内外环 + PnP 细对准
  → 三线合一连续多帧验收
  → 传统插入控制器交接
```

“三线合一”不是单纯让三个二维点重合，而是同时检查摄像头光轴、戳卡外环中心和
实际内通道轴线，并约束横向误差、轴角、悬停距离、重投影误差和连续稳定帧数。
强化学习只在传统闭环冻结后研究有界残差，不替代硬质量门和最终插入授权。

## 当前仿真基准

- 眼球：NIH 3D / HRA Visible Human female right eye v1.2；
- 戳卡冻结合同：外径 2.0 mm、内径 1.0 mm、壁长 2.5 mm、法兰外径 2.64 mm；
- M0：统一 RGB 感知—控制—评价入口已完成第一版；
- M1：传统颜色/轮廓检测 + IBVS 粗对准已完成；
- M2：YOLO 共用粗观测接口、合成数据生成、训练和在线闭环已完成第一版；
- M3：NIH 场景内外环椭圆几何、2.5D 平移精对准和连续 5 帧验收已完成第一版；
- M4：既有插入控制器交接、插入中视觉监测和接触指标待完成。

控制动作只读取 eye-in-hand RGB 检测结果。MuJoCo 目标真值只用于评价，特权几何分割
只用于生成训练标签，不进入在线控制器。

## 目录

| 路径 | 内容 |
| --- | --- |
| `real_3d_alignment/` | 视觉对准、阶段状态机、质量门和执行桥 |
| `yolo_perception/` | YOLO 粗检测适配器和审计工具 |
| `3d_modeling/mujoco/` | MuJoCo 机械臂、eye-in-hand、NIH 眼球和戳卡场景 |
| `3d_modeling/external_assets/` | NIH/HRA 可追溯视觉资产及审计记录 |
| `single_arm_precision_rl/` | 保留的插入环境、传统基线和残差研究资产 |
| `config/` | 对准、插入、感知合同及模型复现清单 |
| `tests/` | 状态机、视觉检测、MuJoCo 闭环和 YOLO 适配测试 |
| `docs/` | 项目方向、里程碑计划、状态和实验记录 |

## 快速验证

```powershell
python -m pip install -r requirements-simulation.txt
python -m pytest -q
```

运行 NIH 场景传统粗对准：

```powershell
python run_mujoco_coarse_alignment.py --detector traditional
python run_mujoco_coarse_batch.py --episodes 100 --detector traditional
```

运行 M3 粗—细对准和随机测试：

```powershell
python run_mujoco_fine_alignment.py
python run_mujoco_fine_batch.py --episodes 50
```

生成数据、训练 YOLO 并运行 M2 闭环：

```powershell
python generate_nih_yolo_dataset.py
python train_nih_yolo.py --epochs 30
python run_mujoco_coarse_alignment.py `
  --detector yolo `
  --yolo-target-classes 0 `
  --yolo-model output/yolo_nih_hra_m2_training/yolo11n_nih_hra_trocar/weights/best.pt
```

数据集、权重、视频和实验输出位于 `output/`，不会提交到核心代码仓库。已验证模型的
哈希和证据边界记录在 `config/yolo_nih_hra_m2_model_manifest.json`。

详见
[实施计划](docs/plans/VISUAL_ALIGNMENT_AND_INSERTION_PLAN_20260723.md)、
[当前状态](docs/handoffs/CURRENT_PROJECT_STATUS_20260723.md)和
[M1/M2 NIH 基线记录](docs/experiments/M1_M2_NIH_HRA_BASELINE_20260723.md)、
[M3 NIH 精对准记录](docs/experiments/M3_NIH_HRA_FINE_ALIGNMENT_20260723.md)。

## 安全边界

本仓库不构成临床软件。未经真实相机标定、TCP/跟踪误差测量、碰撞检查、人工批准和
独立硬件验证，不得启用自主真实运动，也不得把仿真无碰壁结果表述为真机安全证明。
