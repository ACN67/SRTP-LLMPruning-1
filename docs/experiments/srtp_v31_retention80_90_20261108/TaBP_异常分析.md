# TaBP 90%与80%独立异常分析

结论：数据校验完成，但TaBP生成阶段已严重退化；最优先怀疑整层删除对末尾层和连续中间层的影响。此为待GPU/实际权重验证的假设，而非已证明算法错误。

## 关键事实

- 90% SWE-Lego删除原始32–35层；80%删除21、25、30–35层；90% Granite删除21、23、24、26层；80% Granite删除21–29连续9层。
- SWE-Lego TaBP的MBPP两个保留率均500/500用满2048 tokens；LCB两个保留率均95/95用满4096 tokens，代码生成质量明显异常。
- 评分输入/源manifest哈希、题目、Prompt、配置与Checkpoint身份核查通过；配置层数与索引连续性检查通过，但无真实权重数值审计。
- Granite 90% TaBP MBPP 76条原始文本包含可解析的围栏代码，但后处理失效，值得做独立提取A/B。

## 待验证工作

1. Dense/Wanda/TaBP同题小样本固定配置GPU生成，并核对EOS、logits、解码结果和层映射。
2. 保护末层的90% SWE-Lego消融，以及避免长段连续删层的Granite 80%消融，需明确与原TaBP方法不同。
3. 非测试集代码校准与ARC-Easy QA校准对照，评估恢复微调影响。
4. 另开独立提取流程验证，不覆盖原始已冻结Evaluate结果。

状态：ABNORMAL / REQUIRES_AUDIT。数据及证据索引见本目录的CSV与JSON。
