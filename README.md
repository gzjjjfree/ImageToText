# 🚀 ImageToText: 高性能安卓离线 OCR 助手

[![Go Report Card](https://goreportcard.com/badge/github.com/gzjjjfree/ImageToText)](https://goreportcard.com/report/github.com/gzjjjfree/ImageToText)
[![Platform](https://img.shields.io/badge/Platform-Android-green.svg)](https://www.android.com)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Downloads](https://img.shields.io/github/downloads/gzjjjfree/ImageToText/total?style=flat-square&color=orange)

**ImageToText** 是一款基于 Go 语言、Fyne 界面框架以及 ONNX Runtime 构建的高性能安卓离线 OCR 工具。它集成了 PaddleOCR 模型，并针对移动端进行了深度优化，实现了在安卓设备上极致的识别速度。

## ✨ 核心特性

* **极致速度**：经过深度优化的全链路处理，首识别仅需 **1045ms**，后续稳定在 **1000ms** 以内。
* **全离线识别**：无需联网，保护隐私，随时随地可用。
* **智能并行处理**：采用 IoBinding 机制实现线程安全的多行并行识别。
* **交互顺畅**：
    * **状态分离**：独立的只读状态栏，实时显示识别耗时。
    * **一键保存**：集成原生文件保存对话框，轻松导出识别结果。
* **底层优化**：
    * 手写图像预处理（CHW 格式转换），避开昂贵的绘图接口。
    * CTC 解码性能优化，减少内存分配。
    * 全自动模型预热机制，启动即巅峰。

## 📸 运行预览



## 🛠️ 技术架构

* **UI 框架**: [Fyne](https://fyne.io/)
* **推理引擎**: [ONNX Runtime Go](https://github.com/yalue/onnxruntime_go)
* **OCR 模型**: PaddleOCR V3 (Detection + Recognition)
* **语言**: [Go](https://golang.org/) 1.2x (CGO)

## 🏗️ 编译与打包

### 环境准备
1. 安装 Android SDK & NDK (推荐 r25c)。
2. 配置 Go 移动开发环境。
3. 准备 `libonnxruntime.so` 动态库(已包含在 /lib)。

### 自动化构建脚本
可参考 `release.yml` 脚本自己生成 `build.sh`，可一键完成编译、库注入、对齐及签名：

```bash
./build.sh
```

## 📖 使用指南

1.  **加载模型**：应用启动后会自动后台预热，无需手动干预。
2.  **选择图片**：点击顶部的 **“选择图片”** 按钮。
3.  **保存结果**：点击 **“保存到本地”**，将文字导出为 `.txt` 文件。

## ⚙️ 核心性能指标 (基于 Android 真机单次并行测试, 文本行数少于CPU核心数)

| 阶段 | 耗时 | 优化技术 |
| :--- | :--- | :--- |
| **预处理(后台)** | ~1200ms | Pointer-based Pix Access |
| **文本检测 (Det)** | ~170ms | Single-thread Intra-Op |
| **文本识别 (Rec)** | ~700ms | IoBinding Parallelism |
| **CTC 解码** | < 10ms | strings.Builder & Segmented Slice |
| **总计** | **~900ms** | 全链路预热与并发调度 |

## 🤝 贡献

欢迎提交 [Issue](https://github.com/gzjjjfree/ImageToText/issues) 或 Pull Request 来完善这个项目！

## 📄 开源协议

本项目基于 [**MIT License**](LICENSE) 协议开源。