用于记录需要预先设置的配置选项，这些配置项在运行时会被用到。
# 1.成员函数
## 1.1create_instance
![[Screenshot 2024-06-09 203634.png]]
```cpp
Config* config = Config::create_instance();
if(config == nullptr){
  printf("Create config failed !\n");
  return EXIT_FAILURE; 
}
```
## 1.2成员函数 Context
用于记录运行过程中所涉及的运行时上下文数据，就包括 Model 对象和 Confi 对象
![[Screenshot 2024-06-16 172427.png]]
## 1.3成员函数 get_model
![[Screenshot 2024-06-16 172521.png]]
## 1.4成员函数 get_config
![[Screenshot 2024-06-16 172609.png]]
## 2.1 accelerate_type
![[Screenshot 2024-06-09 204059.png]]
```cpp
enum struct AccelerateType : uint8_t{
    TYPE_DEFAULT = 0,
    TYPE_CPU,
    TYPE_GPU,
    TYPE_DSP,
    TYPE_NPU,
};
```
## 2.2 is_quantify_model
![[Screenshot 2024-06-09 204332.png]]
## 2.3 snpe_out_names
![[Screenshot 2024-06-09 205551.png]]
## 2.4 implement_type
![[Screenshot 2024-06-16 171806.png]]
## 2.5 framework_type
![[Screenshot 2024-06-16 171932.png]]
## 2.6 number_of_threads
![[Screenshot 2024-06-16 172017.png]]
## 2.7 fast_timeout
![[Screenshot 2024-06-16 172124.png]]
```cpp
config->implement_type = ImplementType::TYPE_FAST;
config->framework_type = FrameworkType::TYPE_SNPE;
config->accelerate_type = AccelerateType::TYPE_CPU;
config->is_quantify_model = 0;
……
std::vector<std::string> output_node_names;
output_node_names.push_back("InceptionV3/Predictions/Softmax");
config->snpe_out_names = output_node_names;
```
