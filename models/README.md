# 本地模型目录

除本文件外，本目录内容不进入 Git。服务器应把持久存储挂载到 `/data/models`。每个完整 snapshot 必须含 `.srtp_model_source.json`，并在加载权重前通过 `scripts/setup/verify_model_snapshot.py`。

ModelScope 只是默认国内 transport；模型科学身份始终是 `configs/models/` 与 `configs/model_snapshots/` 固定的 canonical Hugging Face repo 和 exact revision。
