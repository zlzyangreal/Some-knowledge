# 1.全上传
## 1.1 建立一个仓库，复制仓库地址
![[Git1.png]]
## 1.2 在本地新建Git配置
![[Git2.png]]
## 1.3 指令操作
### 1.3.1 克隆地址
```bash
git clone https://github.com/zlzyangreal/MyNote.git
```
### 1.3.2 上传与联系
```bash
cd  boke     //根据自己的远程仓库名输入
git init
git add .
git commit -m “你的提交信息”
git remote add origin https://github.com/zlzyangreal/MyNote.git
git push -u origin master
```
# 2. 上传指定部分文件
## 2.2 含中文要先防报错
```bash
git config --global core.quotepath false
```
## 2.3 创建白名单（也就是指定上传的部分）
创建文件`.gitignore`
```txt
# =========== 1. 忽略所有文件 ===========
*

# =========== 2. 允许 Git 进入所有子目录 ===========
# (如果没有这行，Git 根本不会进入文件夹去检查白名单)
!**/

# =========== 3. 你想保留的特定文件夹 (白名单) ===========
# 注意：路径用斜杠 /，空格不需要加引号

!Deep_learning/model/目标检测/YOLO/**
!语言/python/**
!Document images/python/**
!Document images/YOLO/**

# =========== 4. 不要忽略配置文件本身 ===========
!.gitignore
```
核实
```bash
git status
%% 输出如下表示正确 %%
No commits yet

Untracked files:
  (use "git add <file>..." to include in what will be committed)
        .gitignore
        Deep_learning/
        Document images/
        语言/

nothing added to commit but untracked files present (use "git add" to track)

```
## 2.4 推送
```bash
git add .
git commit -m "Initial commit: Upload specific knowledge notes"
git remote add origin https://github.com/zlzyangreal/Some-knowledge.git
git push -u origin main
```
# 3. 错误情况
## 3.1 权限问题
```bash
fatal: detected dubious ownership in repository at 'F:/Data/Self_Project/LeetCode/LeetCode_cpp'

'F:/Data/Self_Project/LeetCode/LeetCode_cpp' is owned by:

        'S-1-5-32-544'

but the current user is:

        'S-1-5-21-2919904234-389789952-1821828111-1001'

To add an exception for this directory, call:



        git config --global --add safe.directory
```
解决方法
```bash
git config --global --add safe.directory F:/Data/Self_Project/LeetCode/LeetCode_cpp
```
## 3.2 远程有内容直接覆盖
```bash
 ! [rejected]        main -> main (fetch first)
error: failed to push some refs to 'https://github.com/zlzyangreal/Some-knowledge.git'
hint: Updates were rejected because the remote contains work that you do
hint: not have locally. This is usually caused by another repository pushing
hint: to the same ref. You may want to first integrate the remote changes
hint: (e.g., 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.
```
解决方法（强制推送）
```bash
git push -u origin main -f
```
# 4.常用其他指令
## 4.1 切换分枝
```bash
git branch -M main
```