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

“三线合一”不是单纯让三个二维点重合，而是同时检查：

- 摄像头光轴与戳卡外环中心的像素误差；
- 内外环同心度；
- 横向位置误差；
- 摄像头光轴与戳卡通道轴夹角；
- 悬停距离及重投影误差；
- 连续多帧稳定性。

强化学习用于优化对准路径、六轴耦合、阶段切换和收敛效率；硬质量门、工作空间约束和
最终插入授权不交给无约束策略。

## 目录

| 路径 | 内容 |
| --- | --- |
| `real_3d_alignment/` | 核心视觉对准、阶段状态机、质量门和执行桥 |
| `yolo_perception/` | YOLO 数据与检测审计 |
| `3d_modeling/scripts/` | 圆环/PnP、手眼标定、姿态过滤和仿真工具 |
| `3d_modeling/mujoco/` | Meca500、eye-in-hand、眼球和戳卡场景 |
| `single_arm_precision_rl/` | 精简保留的传统插入与残差环境基线 |
| `config/` | 对准、插入和视觉合同 |
| `tests/` | 核心状态机与接口测试 |
| `docs/` | 项目方向、实施计划和当前状态 |

## 快速验证

```powershell
python -m pip install -r requirements-core.txt
$env:PYTHONPATH=(Get-Location).Path
python -m pytest -q
```

离线图像 dry-run：

```powershell
python run_real_3d_alignment.py --help
```

所有统一入口默认在真实运动前停止。模型权重、数据集、训练输出、论文构建物和原项目
历史报告不存入本仓库。

## 当前状态

- 分级对准状态机：已实现；
- 连续多帧三线合一验收：已实现；
- 现有圆环/PnP/手眼链：已保留；
- YOLO 与粗对准统一运行接口：待接入；
- 传统两级对准正式基线：待冻结；
- 对准残差强化学习：待基线通过后启动；
- 双臂协同：等待单臂全流程通过。

详见 [项目方向](docs/project/PROJECT_DIRECTION_20260723.md)和
[实施计划](docs/plans/VISUAL_ALIGNMENT_AND_INSERTION_PLAN_20260723.md)。

## 安全边界

本仓库不构成临床软件。未经现场标定、TCP/跟踪误差测量、碰撞检查、人工批准和独立
硬件验证，不得启用自主真实运动或把仿真零碰壁结果表述为真机安全证明。
