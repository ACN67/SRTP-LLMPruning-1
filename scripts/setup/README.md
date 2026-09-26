# 环境与资产准备工具

持久化目录约定为 `/data/{models,cache,datasets,checkpoints,results}`。

- `preflight.py`：检查软件版本、GPU capability 和可写 mount。
- `download_models.py`：只负责下载；默认通过 ModelScope 获取 canonical runtime snapshot，按 manifest 校验 size/hash 并写 provenance sidecar。
- `verify_model_snapshot.py`：只读验证，不加载权重、不改 provenance；复用同一 manifest 校验全部 runtime 文件、config、tokenizer 与 weight index。
- `prefetch_assets.py`：下载固定 benchmark 文件以及 C4/WikiText raw calibration 文件。
- `verify_assets.py`：只读校验数据 hash、任务身份和轻量结构。
- `build_image.sh`、`smoke_image.sh`、`export_image.sh`：build、smoke 与本地 tar/hash/metadata 导出；build 时把 source Git commit 与 dirty 状态注入 image provenance。

工具不会把 token 写入 provenance。模型和数据下载都是显式步骤，不属于 Docker build。
