# 分级视觉对准

该包是当前项目核心，负责：

1. 采集 eye-in-hand 图像和机器人状态；
2. 获取戳卡目标并估计内外环/通道位姿；
3. 生成有界粗对准或细对准建议；
4. 在安全悬停区执行非共线三视点主动观察，联合估计戳卡中心、法向和协方差；
5. 连续多帧验证三线合一；
6. 仅在单帧质量门和主动多视点门全部通过后向现有插入控制器发出交接授权。

## 现有模块

- `observation.py`：相机与机器人状态采集；
- `pipeline.py`：圆环检测、PnP、质量门和 dry-run 控制建议；
- `staged_alignment.py`：SEARCH、COARSE、FINE、OBSERVE_ACTIVE、ALIGNED
  阶段与插入交接门；
- `multiview_circle_pose.py`：标定相机位姿和多帧椭圆联合圆平面估计；
- `execution_bridge.py`：真实运动前静态门和执行桥；
- `step_evaluation.py`：执行前后误差评价；
- `preflight.py`：环境、模型、配置和安全预检。

视觉对准模块不得直接执行插入。`insertion_handoff_ready` 只是授权信号，仍需插入模块
独立检查当前观测、标定、TCP、速度和安全合同。

当前 `OBSERVE_ACTIVE` 已完成状态机合同、病态解拒绝和近场实验接入；尚未接入正式
M4 视频全流程的真实微动作执行。该项必须在视点可达性、回位轨迹和协方差门完成
网格冻结后再接入，不能通过降低阈值绕过。
