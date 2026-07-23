# 开发工作流

## 工作副本

本仓库不得依赖任意机器绝对路径。代码、配置和文档统一使用仓库相对路径。

## 文件放置

| 内容 | 位置 |
| --- | --- |
| 当前核心：粗细视觉对准和插入交接 | `real_3d_alignment/` |
| 精简保留的插入环境和传统基线 | `single_arm_precision_rl/` |
| 视觉审计代码 | `yolo_perception/` |
| 实验配置 | `config/` |
| 标定、仿真和维护脚本 | `3d_modeling/scripts/` |
| 自动测试 | `tests/` |
| 项目决策 | `docs/project/` |
| 后续计划 | `docs/plans/` |
| 实验结论 | `docs/experiments/<方向>/` |
| 可再生成的原始实验输出 | `3d_modeling/outputs/` |
| 论文图表与交付物 | `output/` |

不要在根目录新增阶段报告、临时脚本、模型、数据集或论文构建目录。

## 新实验约定

1. 配置放入 `config/`，文件名包含阶段、目的和条件。
2. 实现放入现有 Python 包，避免继续向根目录增加入口脚本。
3. 测试与实现同步添加，数据按完整 episode/物理条件划分。
4. 原始输出写入 `3d_modeling/outputs/<模块>/<实验名>/`。
5. 人工结论写入 `docs/experiments/<方向>/`。
6. 只在晋级条件改变时更新当前状态、主线和交接文档。

## 最小验证

```powershell
$env:PYTHONPATH=(Get-Location).Path
python -m pytest -q
git diff --check
```

大型训练和论文编译不属于默认回归。运行前应记录配置、随机种子、数据清单哈希和代码
提交；生成结果不得覆盖已冻结实验。

## 当前安全边界

真实硬件自主运动、对准强化学习长训和双臂阶段仍受项目方向与实施计划中的晋级门控制。
软件接口通过不等于真实运动授权。
