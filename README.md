# Change Detection Experiments

基于 Open-CD 的变化检测复现与实验记录：SNUNet、STANet、TinyCD、BIT、ChangeFormer-B0。

本项目记录学习阶段的模型复现、STA 实现排查、扰动与输入顺序分析、跨数据集迁移及短程适配。当前数据是遥感变化检测；工业机器人视觉是后续研究方向，当前实验尚不能证明工业现场性能。

## 已完成结果

LEVIR-CD 完整测试集 2048 对，原五模型按验证集选择检查点，单次推理，F1 单位为 %：

| 模型 | Test F1 | IoU |
|---|---:|---:|
| SNUNet-C16 | 90.82 | 83.18 |
| STANet-PAM 原实验 | 83.81 | 72.13 |
| TinyCD | 90.41 | 82.50 |
| BIT | 90.65 | 82.90 |
| ChangeFormer-B0（Open-CD 实现） | 90.66 | 82.91 |
| STA 调整后，单次推理 | 89.7405 | 81.3903 |
| STA 调整后，4 次翻转 TTA | 90.1781 | 82.1130 |

原 STA 训练使用 batch 2、累积 4、160000 个微步；其他基线 batch 8、40000 步。样本预算相当不意味着 BatchNorm、精度或更新行为完全相同。STA 调整包含额外训练及多项同时变更，不是控制变量消融，也未超过最强基线。

其他实验：

- 固定 256 对验证子集的亮度、噪声、模糊、平移扰动；三轮实验统一排除 4 像素边界。
- A/B 单独模糊及输入交换复核：暴露部分实现的方向敏感性，不能推广为架构的普遍结论。
- CDD Real/subset 的固定 256 对验证样本零样本迁移（全图评价，与上述边界规则不同）。
- SNUNet 在 CDD 上训练 4000 步：同一验证子集 F1 从 28.6872 到最佳 78.5556（3500 步）。验证集参与选优，且抽样溯源发现训练/验证共享原始场景；不是独立场景泛化结果，没有从零训练对照。

## 仓库内容

- `framework/`：本地 Open-CD 的 Python 源码、配置、工具及安装支持快照，保留上游许可证。
- `baseline_configs/`、`baseline_results/`：五模型实际运行配置和冻结测试记录。
- `outputs/sta_final/`：STA 残差头、训练和评估脚本、最终结果。
- `outputs/robustness/`：三轮扰动/模糊/交换实验脚本、固定样本清单与汇总。
- `outputs/cross_dataset/`、`outputs/cdd_adaptation/`：CDD 核查、迁移和适配材料。
- `docs/experiment_ledger.md`：30 组历史实验记录，包含失败尝试。
- `docs/source_manifest.json`：源文件与发布文件摘要；路径脱敏后文件哈希会改变。
- `environment/reference_versions.json`：原实验环境的关键版本，不是跨平台锁文件。

未上传数据集图像、模型权重、虚拟环境、个人文档或训练日志。逐图结果压缩归档在 `.json.gz` 中；同名 `.json` 保留汇总字段。历史记录中的原始哈希用于追溯，不能当作脱敏后文件的哈希。

## 使用与复现范围

这是研究材料归档，不宣称一键复现全部历史实验。路径已替换为占位符，先阅读 [运行说明](docs/RUNNING.md)，通过 `prepare_runtime.py` 生成单独运行副本。权重、数据以及必要的预训练模型需自行提供；归档结果可直接阅读，不需要 GPU。

原 STANet 基线发生在 BCL 修正之前；`framework/` 是当前修正后的代码快照，不能用它声称精确重现原有缺陷基线。ChangeFormer-B0 是 Open-CD 中基于 MiT-B0 的实现，不等于作者原始 V6。遥感建筑变化标注并不涵盖植被、车辆、光照等所有视觉差异。

## 来源

框架来自 [Open-CD](https://github.com/likyoo/open-cd)。本地目录没有 Git 元数据，因此无法可靠提供精确上游提交号；不虚构版本来源。参见 [第三方来源与许可](THIRD_PARTY.md)。
