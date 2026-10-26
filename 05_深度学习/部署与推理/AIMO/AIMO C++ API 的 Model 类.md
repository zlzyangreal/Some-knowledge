Model 类主要用于记录模型的文件信息、结构信息、运行过程中模型相关内容。
# 1.成员函数 set_model_properties
![[Screenshot 2024-06-09 202518.png]]
# 2.成员函数 create_instance
![[Screenshot 2024-06-09 203006.png]]
```cpp
Model* model = Model::create_instance("./inceptionv3_float32.dlc");
if(model == nullptr){
    printf("Create model failed !\n");
    return EXIT_FAILURE;
}
```
# 3.成员函数 get_model_absolute_path
![[Screenshot 2024-06-16 171418.png]]
# 4.成员函数 get_model_type
![[Screenshot 2024-06-16 171525.png]]