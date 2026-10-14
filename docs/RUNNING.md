# 运行与复现边界

先阅读 README 的结果范围。本次没有启动训练。

1. 参照 environment/reference_versions.json 建立兼容环境；CUDA/PyTorch/MMCV 需匹配，不能只盲目安装最新包。
2. 运行 `python prepare_runtime.py --destination ../cd-runtime --levir-root /path/to/LEVIR-CD256 --cdd-root /path/to/ChangeDetectionDataset`。生成副本后进入该目录。
3. 在兼容环境安装 `python -m pip install -e framework --no-deps`。
4. 根据 baseline_results/results.json 与各 manifest 中的 checkpoint 路径放置权重。原始 STA 的 128000 步权重放入 framework/work_dirs/stanet_pam_base_s42；调整后权重放入 work/sta_final_train。文件摘要可用于核验。
5. 运行新实验前，将已有 results、blur_results、swap_results 目录备份到运行副本之外。脚本有断点恢复与 manifest 校验；不要混用历史结果与新运行。历史 manifest 哈希不会因路径修改自动有效。

常用入口（在运行副本内）：

- `python framework/tools/train.py <配置路径>`：Open-CD 基线训练。
- `python framework/tools/test.py <配置路径> <权重路径>`：基线评估。
- `python outputs/sta_final/train.py --mode train --steps 40000`：STA 调整训练；脚本默认步数不是本轮实际 40000，必须显式指定。
- `python outputs/robustness/run.py --smoke`：2 对样本的基础扰动检查。完整实验去掉 --smoke。
- 模糊与交换实验使用同目录 blur_followup.py、swap_followup.py，先查看 --help。
- CDD 脚本见 outputs/cross_dataset/run.py 与 outputs/cdd_adaptation/train.py，先查看 --help，再运行 smoke。

说明：部分历史分析入口仍依赖未打包的旧工作区文件（__LEGACY_ROOT__）；这些入口作为归档参考，不承诺直接运行。字体默认使用 Windows Arial，其他平台需 --font-root 指向提供 arial.ttf 的目录，或自行调整绘图字体。数据必须保持原实验划分及 A/B/标签配对。程序性路径替换不等于跨平台验证。
