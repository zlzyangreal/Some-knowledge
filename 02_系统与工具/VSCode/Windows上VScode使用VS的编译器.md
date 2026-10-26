打开终端(`ctrl + Shift + p`)
输入`C/C++: Edit Configurations`
配置`JSON`文件
```json
{
    "configurations": [
        {
            "name": "Win32",
            "includePath": [
                "${workspaceFolder}/**",
                "${vcpkgRoot}/x64-windows/include",
                "${vcpkgRoot}/x64-windows-static/include",
                "${vcpkgRoot}/x86-windows-static/include"
            ],
            "defines": [
                "_DEBUG",
                "UNICODE",
                "_UNICODE"
            ],
            "compilerPath": "D:\\visual_studio\\IDE\\VC\\Tools\\MSVC\\14.29.30133\\bin\\Hostx64\\x86\\cl.exe",
            "cStandard": "c17",
            "cppStandard": "c++20",
            "intelliSenseMode": "windows-msvc-x64"
        }
    ],
    "version": 4
}
```