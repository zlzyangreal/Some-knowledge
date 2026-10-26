```cpp
#include <torch/torch.h>
#include <iostream>

// 定义 U-Net 模型结构
struct UNet : torch::nn::Module {
    UNet() {
        encoder = register_module("encoder", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 3).padding(1)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(2, 2),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).padding(1)),
            torch::nn::BatchNorm2d(128),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(2, 2)
        ));
        decoder = register_module("decoder", torch::nn::Sequential(
            torch::nn::UpsampleBilinear2d(scale_factor(2)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 64, 3).padding(1)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU(),
            torch::nn::UpsampleBilinear2d(scale_factor(2)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 1, 1)),
            torch::nn::Sigmoid()
        ));
    }
	
	    torch::Tensor forward(torch::Tensor x) {
        x = encoder(x);
        x = decoder(x);
        return x;
    }

    torch::nn::Sequential encoder{nullptr}, decoder{nullptr};
};

int main() {
    torch::manual_seed(0);

    // 加载预训练模型
    auto model_path = "path_to_your_model.pt";
    auto model = std::make_shared<UNet>();
    torch::load(model, model_path);

    // 预处理输入图像
    auto input_image = torch::randn({1, 3, 256, 256}); // 假设输入图像尺寸为 256x256
    auto input_tensor = input_image.to(torch::kFloat32); // 转换为浮点数

    // 推理
    auto output = model->forward(input_tensor);

    // 应用阈值分割
    auto threshold = 0.5;
    auto mask = output > threshold;

    // 显示结果
    std::cout << "Output shape: " << output.sizes() << std::endl;
    std::cout << "Mask shape: " << mask.sizes() << std::endl;

    return 0;
}
```