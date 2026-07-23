# 分级视觉对准

该包是当前项目核心，负责：

1. 采集 eye-in-hand 图像和机器人状态；
2. 获取戳卡目标并估计内外环/通道位姿；
3. 生成有界粗对准或细对准建议；
4. 连续多帧验证三线合一；
5. 仅在验收通过后向现有插入控制器发出交接授权。

## 现有模块

- `observation.py`：相机与机器人状态采集；
- `pipeline.py`：圆环检测、PnP、质量门和 dry-run 控制建议；
- `staged_alignment.py`：粗细对准阶段与插入交接门；
- `execution_bridge.py`：真实运动前静态门和执行桥；
- `step_evaluation.py`：执行前后误差评价；
- `preflight.py`：环境、模型、配置和安全预检。

视觉对准模块不得直接执行插入。`insertion_handoff_ready` 只是授权信号，仍需插入模块
独立检查当前观测、标定、TCP、速度和安全合同。
