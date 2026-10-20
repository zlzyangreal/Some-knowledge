这是一种用于前馈卷积神经网络的简单而有效的注意力模块。
# 1.简介
给定一个中间特征图，CBAM模块会沿着两个独立的维度（通道和空间）依次推断注意力图，然后将注意力图与输入特征图相乘以进行自适应特征优化。
由于CBAM是轻量级的通用模块，因此可以忽略的该模块的开销而将其无缝集成到任何CNN架构中，并且可以与基础CNN一起进行端到端训练。
# 2.模型
Convolutional Block Attention Module (CBAM) 表示卷积模块的注意力机制模块，是一种结合了空间（spatial）和通道（channel）的注意力机制模块。
![[模块框架.jpg]]
- 上图给出了添加CBAM模块之后的整体结构。可以看到的是，卷积层输出的结果，会先通过一个通道注意力模块，得到加权结果之后，会再经过一个空间注意力模块，最终进行加权得到结果。
![[模块分框架.jpg]]
- 通道注意力模块如第一个图，将输入的特征图，分别经过基于`width`和`height`的`global max pooling `和`global average pooling`，然后分别经过`MLP`。将`MLP`输出的特征进行基于`element-wise`的加和操作，再经过`sigmoid`激活操作，生成最终的`channel attention featuremap`。将该`channel attention featuremap`和`input featuremap`做`elementwise`乘法操作，生成`Spatial attention`模块需要的输入特征。
- 空间注意力模块如下面的图所示。将`Channel attention`模块输出的特征图作为本模块的输入特征图。首先做一个基于`channel`的`global max pooling` 和`global average pooling`，然后将这2个结果基于`channel `做`concat`操作。然后经过一个卷积操作，降维为1个`channel`。再经过`sigmoid`生成`spatial attention feature`。最后将该`feature`和该模块的输入`feature`做乘法，得到最终生成的特征。