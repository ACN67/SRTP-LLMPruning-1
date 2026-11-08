# SRTP v3.1 2026-11-08 剪枝Direct实验归档

遵循仓库results/README.md约定：Git存储报告和小型聚合元数据；原始模型、批量生成与逐题测试文件不写入Git历史。

- [11.08进度汇总](11.08进度汇总.md)：按10.19报告六节结构记录本次进度和数据。
- [48组评分](scores_48_runs.csv)：两模型、四种方法、两保留率、三基准，共12144条评分的聚合。
- [90%/80%对比](comparison_80_vs_90.csv)：同模型同方法的分数变化。
- [LCB难度分解](livecodebench_difficulty.csv)：Easy43、Medium52，未包含Hard。
- [生成与评分SHA-256索引](sha256_manifest_48_runs.csv)：生成、评分、任务集、数据集与checkpoint身份。
- [16组Checkpoint元数据](checkpoint_metadata_16.json)：摘要、结构化删层等，不包含safetensors。
- [TaBP异常分析](TaBP_异常分析.md)：证据、局限性及建议消融验证。
- [生成质量统计](generation_quality_metrics.csv) 和 [失败类型](failure_modes_80_vs_90.csv)。
- [SHA256SUMS.txt](SHA256SUMS.txt)：本目录发布文件的完整性校验。

## 限制

- 48/48 Generate和48/48 Evaluate任务经本机备份与SHA索引核实，两个保留率各6072条评分。
- 本次LCB是95题子集且单题测试超时3秒；10.19 Dense基线是完整175题且6秒，不应直接当成同一协议。
- TaBP有显著退化且根因尚需GPU权重复核。
- 本次GitHub发布**不含**模型权重、完整批量Generate文件、完整逐题outcomes或第三方数据集；SHA索引不代表那些大文件已托管到GitHub。
