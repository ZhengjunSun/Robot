# 保留的单臂精密插入代码

2026-07-23 起，项目核心转为 `real_3d_alignment/` 中的插入前分级视觉对准。本目录继续
保留成熟插入控制、MuJoCo 环境、历史残差 RL 和安全审计资产，作为对准通过后的执行模块
及全流程验证平台。

## 稳定入口

- `config.py`：配置加载与继承。
- `controllers.py`：冻结几何基线控制器。
- `environment.py`：任务空间残差环境。
- `gym_env.py`：Gymnasium 适配层。
- `mujoco_in_loop_env.py`：MuJoCo、接触、延迟和风险门闭环。
- `train_residual_sac.py`：残差 SAC 训练入口。

其余 `audit_*`、`collect_*`、`generate_*`、`train_*` 和 `evaluate_*` 文件主要对应历史
实验阶段。除修复插入交接和全流程验证所需问题外，不再继续扩展历史 G98/H 阶段。

## 设计边界

几何控制器生成名义动作，学习策略只生成有界微残差；质量门、间隙门和拒绝机制拥有
最终执行权限。MuJoCo 真值、真实接触和特权状态只能用于离线标签或明确声明的训练期
critic，不得进入部署 actor。

实验报告统一写入 `docs/experiments/`，原始数据写入
`3d_modeling/outputs/single_arm_precision_rl/`。
