---
license: apache-2.0
base_model: Qwen/Qwen2.5-3B-Instruct
library_name: deepinfer-sm86
tags:
  - qwen2.5
  - deepinfer
  - sm86
  - rtx-3050
  - w4a16
  - int4
  - gqa
  - cuda-graph
---

# Qwen2.5-3B-DeepInfer-SM86

Qwen2.5-3B-Instruct + `maibot-catgirl-v3` LoRA 的 **DeepInfer SM86 W4A16**
权重与运行代码。

目标设备：

```text
NVIDIA RTX 3050 Laptop GPU
4 GB VRAM
SM86
Windows
```

这是一个 **DeepInfer model_executor 级** 的 3B 部署包，不是完整 DeepInfer
engine。

- 父项目：<https://github.com/umouyoumou-del/DeepInfer-SM86>
- 权重 Release：<https://github.com/umouyoumou-del/Qwen2.5-3B-DeepInfer-SM86/releases/tag/v0.1.0>
- 运行格式：DeepInfer layered W4A16 checkpoint
- 权重大小：约 **1.912 GiB**

---

## 结果速览

在 RTX 3050 4 GB 上，用 **融合后的 3B W4A16 + 256 GQA + CUDA Graph** 实测：

| 指标 | 结果 |
|---|---:|
| checkpoint | 1.912 GiB |
| 服务加载 | 12.299 秒 |
| prefill 中位 | 0.4551 秒 |
| decode 中位 | **44.777 tokens/s** |
| 一次请求 wall 中位 | **0.9009 秒** |
| completion 中位 | 19 tokens |
| context | 256 tokens |

同一协议的 Ollama 3B Q4_K_M：

| 指标 | 3B DeepInfer SM86 | Ollama 3B |
|---|---:|---:|
| prefill | 0.4551 s | 0.0603 s |
| decode | 44.777 t/s | 57.285 t/s |
| wall | 0.9009 s | 0.4637 s |
| context | 256 | 4096 |

结论：3B DeepInfer SM86 已经进入个人快捷操作可用区间；Ollama 仍然是更成熟的
3B 快速基线。

---

## 权重下载

权重 **不放在普通 Git 文件里**，因为 embedding 单文件就约 622 MB，整个
checkpoint 约 1.912 GiB。使用 GitHub Release 的两个资产：

```text
Qwen3B-W4A16-SM86.tar.part-00   1000 MiB
Qwen3B-W4A16-SM86.tar.part-01    970 MiB
```

下载：

```powershell
gh release download v0.1.0 `
  --repo umouyoumou-del/Qwen2.5-3B-DeepInfer-SM86 `
  --dir weights
```

合并并解压：

```powershell
cd weights
cmd /c copy /b Qwen3B-W4A16-SM86.tar.part-00+Qwen3B-W4A16-SM86.tar.part-01 Qwen3B-W4A16-SM86.tar
tar -xf Qwen3B-W4A16-SM86.tar -C D:\Qwen3B-W4A16-SM86
```

Linux/macOS：

```bash
cat Qwen3B-W4A16-SM86.tar.part-00 Qwen3B-W4A16-SM86.tar.part-01 > Qwen3B-W4A16-SM86.tar
tar -xf Qwen3B-W4A16-SM86.tar -C /path/to/Qwen3B-W4A16-SM86
```

校验：

```powershell
sha256sum -c SHA256SUMS
```

当前资产校验值：

```text
bec12d38c1d6f620b990011181013c85c3c3f24e0c7d5ead53dc3e5a5e1a71aa  Qwen3B-W4A16-SM86.tar.part-00
9c207ce6f7af7711d69f1e392f61320866a4a393e85040e610c42d5399def6b0  Qwen3B-W4A16-SM86.tar.part-01
155a2f35baf44081a31858181df6de06a5fcc9c7a6e5afb4df74f9c04f6d8e8e  Qwen3B-W4A16-SM86.tar
```

解压后的目录应包含：

```text
config.json
generation_config.json
tokenizer.json
tokenizer_config.json
vocab.json
merges.txt
manifest.json
embed.safetensors
norm.safetensors
layer_00.safetensors
...
layer_35.safetensors
```

---

## 快速启动

### 1. 环境

已验证：

```text
Python 3.10
torch 2.5.1+cu121
CUDA 12.1
CuPy 14.2.0
safetensors
transformers
```

安装：

```powershell
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1
pip install -r requirements-sm86.txt
```

### 2. 启动本地 OpenAI 兼容网关

```powershell
cd D:\Qwen2.5-3B-DeepInfer-SM86
python apps\server.py `
  --checkpoint D:\Qwen3B-W4A16-SM86 `
  --backend w4a16 `
  --model-id deepinfer-sm86-3b-w4a16 `
  --max-cache 256 `
  --port 18001
```

接口：

```text
GET  http://127.0.0.1:18001/health
GET  http://127.0.0.1:18001/v1/models
POST http://127.0.0.1:18001/v1/chat/completions
```

OpenAI 客户端 Base URL：

```text
http://127.0.0.1:18001/v1
```

模型名：

```text
deepinfer-sm86-3b-w4a16
```

### 3. 直接用 loader

```python
from deepinfer.model_executor.models.qwen2_w4a16 import (
    Qwen2W4A16ForCausalLM,
)

model = Qwen2W4A16ForCausalLM.from_checkpoint(
    r"D:\Qwen3B-W4A16-SM86",
    device="cuda",
    group_size=128,
)
```

`from_checkpoint` 会：

1. 构造空白 W4A16 模型；
2. 加载 dense tied embedding / norm；
3. 逐层加载 q/k/v/o、gate/up/down；
4. **构造期融合 qkv 和 gate-up**；
5. 绑定 W4A16 GEMV + 256 GQA kernel。

---

## 融合内核与执行路径

### W4A16

```text
symmetric int4
group size = 128
two int4 values packed into one uint8
scale = max(abs(group)) / 7
```

- `kernels/qwen2_w4a16_gemv.cu`
- 一个 warp 处理一个输出 row；
- x 预加载到 shared memory；
- 单 token decode 使用 W4A16 GEMV；
- prefill 走分块 dequant + cuBLAS。

### 融合 qkv / gate-up

3B checkpoint 的分层文件保存的是：

```text
q_proj / k_proj / v_proj
gate_proj / up_proj
```

loader 在构造期把它们装入：

```text
Qwen2Attention(fused_qkv=...)
Qwen2MLP(fused_gate_up=...)
```

这样不会先创建普通 q/k/v/gate/up 模块再替换，避免 4 GB 边缘上的 allocator 碎片。

**说明：** 融合是模块级融合，复用同一个 `qwen2_w4a16_gemv` CUDA kernel；
不存在另一个独立的 “fused_matmul.cu”。`qwen2_w4a16.py` 和
`qwen2.py` 里的 fused factory 就是完整实现。

### 256-slot GQA

```text
一个 block 处理一个 query head
256 threads
最多 256 个静态 cache slot
支持 16 Q / 2 KV heads（3B）
```

- `kernels/qwen2_gqa_decode_256.cu`
- 静态 KV cache；
- CUDA Graph replay；
- `max_cache=256`。

---

## 模型规格

```text
base model            Qwen2.5-3B-Instruct
adapter               maibot-catgirl-v3 checkpoint-120
architecture          Qwen2ForCausalLM
hidden_size           2048
intermediate_size     11008
num_hidden_layers     36
num_attention_heads   16
num_key_value_heads   2
head_dim              128
vocab_size            151936
tie_word_embeddings   true
quantization          W4A16, group 128
```

权重内容：

```text
dense FP16 embedding      约 0.622 GB
36 层 W4A16 linear         约 1.29 GB
layernorm / manifest / tokenizer
```

`lm_head` 与 embedding tied，只保存一份。

---

## 目录

```text
README.md
LICENSE
NOTICE
requirements-sm86.txt

model/
  config.json
  generation_config.json
  tokenizer.json
  tokenizer_config.json
  vocab.json
  merges.txt
  manifest.json
  norm.safetensors

deepinfer/
  model_executor/models/qwen2.py
  model_executor/models/qwen2_w4a16.py
  model_executor/models/qwen2_mixed_sm86.py
  hugging_face/qwen2_config.py
  ...

apps/
  server.py           OpenAI 兼容网关
  runtime.py          常驻模型 + 256 static cache + Graph
  ask.py              CLI

scripts/
  export_qwen3b_w4_checkpoint.py
  run_qwen2_w4a16_gpu_harness.py
  bench_productivity_backends.py
  export_sm86_kernels.py

kernels/
  qwen2_w4a16_gemv.cu
  qwen2_w3a16_gemv.cu
  qwen2_gqa_decode.cu
  qwen2_gqa_decode_256.cu
  README.md

docs/
  3b_fused_benchmark.json
  ARTICLE_2_DEEPINFER_SM86_PRODUCTIVITY.md
  PRODUCTIVITY_BENCHMARK_3B_7B_OLLAMA.md
```

---

## 复现 3B checkpoint

原始 3B 分片和 adapter 不包含在仓库里：

```powershell
python scripts\export_qwen3b_w4_checkpoint.py `
  --model F:\QwenQLoRA\models\Qwen2.5-3B-Instruct-smallshards `
  --adapter F:\QwenQLoRA\train\maibot-catgirl-v3\checkpoint-120 `
  --output D:\Qwen3B-W4A16-SM86
```

然后打包、切分并上传 Release。

---

## 已知限制

- 当前是 **DeepInfer model_executor 级**，`full_engine=false`；
- 没有完整 CacheEngine、Sampler、scheduler/worker、continuous batching；
- context 上限 **256 tokens**；长文需要分段；
- 4 GB 上 3B 与 7B 不能同时常驻；
- 3B 是 W4A16 有损量化，不能声称与 BF16 等价；
- GPU harness 在 Python 3.10 下验证，Python 3.12 package 验证未完成；
- 融合 qkv / gate-up 是构造期模块融合，不是独立的 fused GEMM 文件；
- 当前 paging 或后台 GPU 占用会显著拉高 prefill。

---

## 许可

- 基础模型 Qwen2.5-3B-Instruct 遵循其原始许可；
- DeepInfer 派生代码遵循 Apache License 2.0；
- `maibot-catgirl-v3` adapter 由使用者训练，发布前请确认其数据与分发许可；
- 详见 `LICENSE` 与 `NOTICE`。

---

## 相关项目

- 内核与量化工程：<https://github.com/umouyoumou-del/DeepInfer-SM86>
- 上篇：`docs/ARTICLE_1...`（在父仓库）
- 下篇：`docs/ARTICLE_2_DEEPINFER_SM86_PRODUCTIVITY.md`
- 3B/7B/Ollama 基准：`docs/PRODUCTIVITY_BENCHMARK_3B_7B_OLLAMA.md`
