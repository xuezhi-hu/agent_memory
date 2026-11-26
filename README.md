# Agent Memory

本仓库整理长期对话记忆研究的最新 SparseMem 代码，以及运行所需的公共组件和少量参考源码。当前为研究开发版本；实体扩展、发言级索引和集合选证据保留为独立实验，并非声称已合并为正式主方法。

## 目录

- `DPC/SparseMem/sparsemem/`：SparseMem 的对话分块、L0/L1/L2 记忆构建、检索、来源恢复、预算与 Token 记录、评估入口及提示词。
- `DPC/SparseMem/experiments/`：发言级检索入口、实体补检与集合选证据、记忆构建改进的开发实验代码。这些脚本用于历史开发实验，依赖相应缓存和数据；不能将其结果视为完整测试集结果。
- `DPC/SparseMem/tests/`：合成样本和离线单元测试；`vendor/LongMemEval/` 包含评分脚本及原许可证。
- `DPC/SparseMem/shared/`：SparseMem 自带的底层组件、配置和第三方许可，不依赖另一个 TFM 项目。
- `references/LightMem/`：上游 LightMem 的三个参考文件和 MIT 许可证，见该目录的来源说明。

实验日志、数据集、模型权重、密钥和本机环境不在仓库中。使用正式数据时请从数据集官方来源下载并遵循其许可证；不要把测试答案输入检索器或回答模型。使用 API 时通过环境变量提供密钥。

## 安装与离线验证

建议 Python 3.10 或更新版本，安装 PyTorch 等依赖请结合本机环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r DPC/SparseMem/requirements.txt
cd DPC/SparseMem
python -m unittest discover -s tests -v
python -m sparsemem.run --mock --data tests/fixture.json --out runs/fixture --limit 1
```

`--mock` 使用合成历史与模拟模型响应，不需要网络服务，也不能用于报告准确率。实际运行需要配置模型服务和独立下载本地模型；具体模型配置见 `DPC/SparseMem/shared/configs/compact_v3.json`。不提交任何真实密钥。

## 第三方来源

- LightMem: https://github.com/zjunlp/LightMem ，提交 `09b57f75d82cc90148b8c14ab258f2a49a3fae21`，MIT。
- SimpleMem: https://github.com/aiming-lab/SimpleMem ，提交 `db80b6a7c591e0ea730a058e9f5fc4eb06572299`；许可与本项目适配边界见 `DPC/SparseMem/shared/THIRD_PARTY_NOTICES.md`。
- LongMemEval 官方评分脚本及许可位于 `DPC/SparseMem/vendor/LongMemEval/`。

具体实现和开发思路以代码与设计文档为准；不同实验配置不得混合比较。
