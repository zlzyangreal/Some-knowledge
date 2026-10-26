# 1.调用函数
```cpp
void executePythonFile(const char* filePath) {
    // 初始化Python解释器
    Py_Initialize();

    // 设置环境变量 MPLCONFIGDIR
    setenv("MPLCONFIGDIR", "/home/aidlux/Desktop/CODE/plan2/python/Efficient-ADLKlk/mplconfig", 1);
    // 设置环境变量 PYTHONPATH
    setenv("PYTHONPATH", "/home/aidlux/Desktop/CODE/plan2/python/Efficient-ADLKlk", 1);

    // 设置Python文件路径
    FILE* fp = _Py_fopen(filePath, "r");
    if (fp == nullptr) {
        std::cerr << "Failed to open the Python file: " << filePath << std::endl;
        Py_Finalize();
        return;
    }

    // 执行Python文件
    int result = PyRun_SimpleFile(fp, filePath);
    if (result != 0) {
        std::cerr << "Failed to execute the Python file: " << filePath << std::endl;
    }

    // 关闭文件
    fclose(fp);

    // 清理并关闭Python解释器
    Py_Finalize();
}
```
# 2.主函数
```cpp
    // 更改工作目录
    const char* newWorkingDir = "/home/aidlux/Desktop/CODE/plan2/python/Efficient-ADLKlk";
    if (chdir(newWorkingDir) != 0) {
        std::cerr << "Failed to change working directory to: " << newWorkingDir << std::endl;
        return 1;
    }

    // 确保工作目录设置正确
    std::cout << "Current working directory: " << get_current_dir_name() << std::endl;
    const char* pythonFilePath = "aidlux.py"; // 文件路径相对于新的工作目录
    executePythonFile(pythonFilePath);
    std::cout << "Run complete" << std::endl;
```