# 本地模型目录

除本文件外，本目录内容不进入 Git。服务器应把持久存储挂载到 `/data/models`。每个完整 snapshot 必须含 schema v3 `.srtp_model_source.json`（verified runtime file set + artifact content SHA256），并在正式入口加载权重前通过 `scripts/setup/verify_model_snapshot.py`。

已有 schema v2 snapshot 不需要重新下载：对原目录运行 verifier 会核验文件并原地升级 sidecar；升级前正式入口会 fail closed。

ModelScope 只是默认国内 transport；模型科学身份始终是 `configs/models/` 与 `configs/model_snapshots/` 固定的 canonical Hugging Face repo 和 exact revision。
