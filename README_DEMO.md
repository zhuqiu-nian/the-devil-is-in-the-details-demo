# Window-Based Attention Image Compression Demo

本项目是论文 **The Devil Is in the Details: Window-Based Attention for Image Compression** 的本地演示复现。它使用 Kodak 数据集、论文官方预训练权重和 CompressAI baseline，提供 FastAPI 推理服务与 React 仪表盘。

## 已就绪资产

下面是本地 demo 当前已准备好的资产。GitHub 仓库不会包含大模型、外部依赖包、缓存输出和 Kodak 图片原文件；新机器 clone 后可按 `README.md` 的脚本重新下载。

- Kodak 24 张测试图：`data/kodak/kodim01.png` 到 `kodim24.png`
- 论文官方权重：
  - `checkpoints/stf/stf_0035.pth.tar`
  - `checkpoints/stf/cnn_0035.pth.tar`
- CompressAI zoo baseline 隔离包：`external/compressai_zoo_pkg`
- 官方 STF 源码：`external/stf`

## 启动

后端：

```powershell
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

前端：

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

浏览器打开：

```text
http://127.0.0.1:5173
```

## 可用模型

- 本文方法：
  - `STF (Ours)`：当前已下载 `lambda=0.0035`
  - `CNN+WAM (Ours)`：当前已下载 `lambda=0.0035`
- 旧学习式 SOTA：
  - `Minnen2018 mean-scale`：CompressAI `mbt2018-mean`，quality 1-8
  - `Cheng2020 attention`：CompressAI `cheng2020-attn`，quality 1-6；当前作为可选项，只有检测到 Torch cache 中已有权重时才在前端启用
- 传统基线：
  - JPEG：Q=20/35/50/70/90

CompressAI baseline 首次运行会自动下载官方预训练权重到 Torch cache。

## 更多 STF/CNN 权重

前端会自动检测 `checkpoints/stf/` 中已有的官方权重。若要启用更多质量档，请按官方 README 的链接下载并放置为：

```text
checkpoints/stf/cnn_0018.pth.tar
checkpoints/stf/cnn_0067.pth.tar
checkpoints/stf/cnn_025.pth.tar
checkpoints/stf/stf_0018.pth.tar
checkpoints/stf/stf_0067.pth.tar
checkpoints/stf/stf_013.pth.tar
checkpoints/stf/stf_025.pth.tar
checkpoints/stf/stf_0483.pth.tar
```

## 复现边界

- 这是“预训练推理复现”，不重新训练 OpenImages。
- 学习式模型的 bpp 按论文/CompressAI 常用方式由 likelihood 估计：

  \[
  \mathrm{bpp}=\frac{\sum -\log_2 p(\hat{y})+\sum -\log_2 p(\hat{z})}{H\times W}
  \]

- Windows 下 CompressAI 的 C++ entropy/rANS 扩展在本机有兼容问题，因此后端 worker 做了 forward-only patch。当前 demo 计算重建图、likelihood bpp、PSNR、MS-SSIM 和 bit allocation heatmap，不执行真实 bitstream 的 `compress/decompress`。
- 所有结果缓存到 `outputs/cache/`，相同图片、模型和质量档重复运行会直接读缓存。
