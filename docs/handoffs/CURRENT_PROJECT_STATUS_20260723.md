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

### M0/M1 启动结果

已开始执行正式里程碑计划，并完成第一版：

- `real_3d_alignment/visual_loop.py`：统一 RGB 检测、阶段判定、动作和逐步记录入口；
- `real_3d_alignment/coarse_vision.py`：传统红色外环检测和有界 IBVS 粗控制；
- `real_3d_alignment/mujoco_visual_env.py`：控制侧 RGB 与评估侧真值隔离的 MuJoCo 适配器；
- `3d_modeling/mujoco/single_arm_trocar_visual_alignment.xml`：末端相机、眼球和戳卡粗对准场景；
- `run_mujoco_coarse_alignment.py`：视频与机器可读报告入口；
- 四象限偏置 MuJoCo 集成测试和部分遮挡检测回归测试。

当前测试为 `16 passed`。标准演示从约 `169.0 px / 8.60 mm` 初始误差，经 7 个 RGB
控制步进入细对准区域，最终约 `3.45 px / 0.185 mm`。毫米真值只供评估记录，没有进入
检测器或控制器。

证据边界：这是 M1 轻量笛卡尔 MuJoCo 传统粗对准基线，不是完整六轴闭环，不包含
YOLO、内外环/PnP 细对准、连续五帧最终验收或插入。

## 下一步

1. 将 M1 扩展为随机初始位姿批量评估、搜索与重新捕获；
2. 以同一粗观测接口接入现有 YOLO 权重并做冻结测试集对照；
3. 把现有内外环/PnP 报告接入 M3 细对准状态机与视觉伺服；
4. 接入传统插入控制器，建立 M4 单臂全流程；
5. 冻结传统基线后再训练有界残差 RL；
6. 单臂全流程通过后再扩展双臂。

详细计划见 `docs/plans/VISUAL_ALIGNMENT_AND_INSERTION_PLAN_20260723.md`。
