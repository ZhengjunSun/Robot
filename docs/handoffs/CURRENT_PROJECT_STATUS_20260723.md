# 当前项目状态（2026-07-23）

## 新主线

导师沟通后，项目核心调整为插入前 eye-in-hand 视觉对准：

`目标获取 → 粗对准 → 细对准 → 三线合一稳定验收 → 传统插入`

既有安全插入代码和实验保留，但作为对准后的执行与最终安全验证模块。原 G98/H 系列
插入风险学习结果转为历史资产，不再决定项目首页主线。

## 可复用基础

- `real_3d_alignment/`：真实观测、内外环估计、PnP、质量门、控制建议和执行桥；
- `3d_modeling/scripts/`：手眼标定、圆环检测、姿态过滤、扫描与仿真控制；
- `yolo_perception/`：YOLO 数据、检测和 ROI 审计；
- `single_arm_precision_rl/`：插入环境、几何基线、残差学习和安全实验资产；
- `3d_modeling/mujoco/`：六轴机械臂、eye-in-hand、眼球和戳卡仿真场景。

## 本轮新增

新增 `real_3d_alignment/staged_alignment.py`：

- 明确 SEARCH、COARSE、FINE、ALIGNED 四阶段；
- 将三线合一转成光轴—外环、内外环同心度、横向误差、轴角、悬停距离、重投影误差；
- 需要连续多帧稳定通过才允许插入交接；
- 观测缺失或质量失败时 fail-closed，不包含机器人运动命令。

专项测试 `6 passed`。

## 下一步

1. 把现有传统/YOLO 检测统一为粗对准观测接口；
2. 把现有 PnP 报告接入细对准状态机；
3. 建立传统两级对准闭环基线；
4. 再训练用于路径、耦合和收敛优化的有界残差 RL；
5. 单臂全流程通过后再扩展双臂。

详细计划见 `docs/plans/VISUAL_ALIGNMENT_AND_INSERTION_PLAN_20260723.md`。
