# NPU与RKNN

RK3588 / 香橙派 / RKNN 模型导出与部署。

上级：[[部署与推理索引]]　|　总索引：[[总索引]]

## 笔记（11）

- [[PC环境搭建]] — 会与YOLOv8 ultralytics冲突，务必建一个新环境
- [[RKNN模型导出]] — 编译模型转换程序，将torchscript模型（需要注意load_pytorch导入的模型后缀是.pt，模型需要重新命名
- [[RKNN模型部署]] — 这里还是用的野火，版本1.5.0
- [[pt  TorchScript ONNX RKNN 均能转出，无法推理问题]] — pt文件是权重文件，训练出的best.pt，不存在问题
- [[rk3588]] — strings /usr/lib/librknnrt.so grep "librknnrt version"
- [[rknn-toolkit2支持包导出]] （纯代码）
- [[rknn导出格式的修改]] — 导出的模型应该是导出 torchscript 模型
- [[rknn推理YOLOv8]] — 仍然使用野火历程
- [[rknn部署导出TorchScript形式]] — 整个测试型导出，注意测试的是rk_opt_v1分支， 该分支将导出torchscript模型，如果使用main分支默认导
- [[香橙派环境搭建]] — 使用3.7/3.9版本
- [[香橙派网卡驱动]] — http://www.orangepi.cn/orangepiwiki/index.php/AP6275P_PCIe_%
