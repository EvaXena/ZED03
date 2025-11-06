# ZED03 自定义 HLS 模型转换器

本项目提供了一个完整的工作流，用于将 Keras 的 `.h5` 模型通过 HLS4ML 转换为一个高层次综合（HLS）工程。项目中包含执行转换、对比原始 Keras 模型与 HLS 模型预测结果的脚本，以实现定性和定量的分析。

## 🚀 快速开始

请按照以下说明在您的本地机器上配置和运行本项目。

### 1. 环境要求

在开始之前，请确保您已安装以下软件：
*   [Git](https://git-scm.com/)
*   [Conda (Miniconda 或 Anaconda)](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
*   [Vitis_Hls(2024.1)]

### 2. 安装步骤

请遵循以下步骤克隆仓库并设置所需的 Python 环境。

1.  **克隆本仓库：**
    ```bash
    git clone https://github.com/EvaXena/ZED03.git
    cd ZED03/hls4ml
    ```

2.  **创建并激活 Conda 环境：**
    ```bash
    conda create -n zed03 python=3.10.18 -y
    conda activate zed03
    ```

3.  **安装所需的 Python 依赖包：**
    ```bash
    pip install -r requirements_final.txt
    ```

## 💻 使用说明

请确保在 `src` 目录下执行以下所有命令。

```bash
cd src
```

### 第一步：生成 HLS 工程

该脚本会读取 `.h5` 模型文件，使用 HLS4ML 进行转换，并生成一个完整的 Vitis HLS 工程。

```bash
python transform.py
```

### 第二步：对比模型预测结果

生成 HLS 工程后，运行此脚本以对比原始 Keras 浮点模型与 HLS 定点模型的输出结果。

```bash
python compare.py
```
*   脚本使用的测试图片位于 `test_img/` 文件夹内。
*   预测结果与对比图片将保存在 `result_img/` 文件夹内。

### 第三步：评估模型性能

该脚本用于计算原始图片与增强图片的 MSE, PSNR, SSIM, ED, MED, MAE。

```bash
python evaluate.py
```

该脚本用于计算原始图片与增强图片的 MSE, PSNR, SSIM, ED, MED, MAE, UIQE, UIQM。

```bash
python evaluate_v2.py
```

## 🛠️ HLS 工作流程

生成的 HLS 工程建议使用 **Vitis HLS 2024.1** 版本打开和操作。

1.  **打开工程**：启动 Vitis HLS 并打开生成的项目文件夹 (例如 `hls4mlprj_...`)。
2.  **执行 C 综合 (C Synthesis)**：将 C++/SystemC 代码综合成 RTL 设计。
3.  **执行 C/RTL 协同仿真 (Co-simulation)**：(可选但强烈推荐) 验证 RTL 设计的功能是否与 C++ 测试代码一致。
4.  **导出 RTL (Package IP)**：将综合后的设计打包成 IP 核，以便在 Vivado 等工具中使用。

## 📂 项目结构

```
.
├── src/                  # 主要源代码目录
│   ├── transform.py      # 用于将 .h5 转换 HLS 工程的脚本
│   ├── compare.py        # 用于对比 Keras 和 HLS 模型输出结果的脚本
│   ├── evaluate.py       # 用于评估模型性能的脚本(单张图片)
│   └── evaluate_v2.py    # 用于评估模型性能的脚本(批量图片)
├── test_img/             # 存放用于预测的输入图片
├── result_img/           # 存放预测结果和对比图片
├── requirements_final.txt # Python 依赖包列表
└── README.md             # 本说明文件
```


