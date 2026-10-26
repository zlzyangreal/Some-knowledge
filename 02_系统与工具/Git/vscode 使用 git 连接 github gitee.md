[方法](https://blog.csdn.net/qq_38981614/article/details/115013188)

## 1. 一次性配置（每台机器做一次）

```bash
git config --global user.name  "你的名字"
git config --global user.email "你的邮箱"

# 生成 SSH key（一路回车即可）
ssh-keygen -t ed25519 -C "你的邮箱"
# 公钥在 C:\Users\<用户名>\.ssh\id_ed25519.pub
```

把公钥内容贴到 GitHub → Settings → SSH and GPG keys；Gitee → 设置 → SSH 公钥。

验证：

```bash
ssh -T git@github.com     # 看到 "Hi xxx! You've successfully authenticated"
ssh -T git@gitee.com
```

## 2. 让 VS Code 认识仓库

1. 打开文件夹（不是打开单个文件），左侧会出现源代码管理图标。
2. `Ctrl+Shift+P` → `Git: Initialize Repository`，或在终端 `git init`。
3. 改完文件 → 暂存（+）→ 填提交信息 → 提交 → 推送。

## 3. 关联远程仓库

```bash
git remote add origin git@github.com:用户名/仓库名.git   # SSH 地址
git branch -M main
git push -u origin main
```

Gitee 同理，把地址换成 `git@gitee.com:用户名/仓库名.git`。

## 4. 常见错误

| 报错 | 原因 | 解决 |
| --- | --- | --- |
| `Permission denied (publickey)` | 公钥没配 / 用错 key | 重配 SSH key，`ssh -T` 验证 |
| `fatal: remote origin already exists` | 已加过 origin | `git remote set-url origin <新地址>` |
| `Updates were rejected` | 远程有新提交 | 先 `git pull --rebase` 再 push |
| 每次都要输密码 | 用了 HTTPS 地址 | 换成 SSH 地址，或配 credential helper |

## 5. 日常只用这 5 条

```bash
git status              # 我现在改了啥
git add -A              # 全部暂存
git commit -m "说明"    # 提交到本地
git pull --rebase       # 先同步远程
git push                # 再推上去
```

参考：[VS Code 使用 Git 连接 GitHub/Gitee](https://blog.csdn.net/qq_38981614/article/details/115013188)

## 相关

- [[git命令操作]]
- [[Git错误解决方式]]
- [[如何上传文件夹到GitHub上]]
- [[gitkraken使用端口占用解决方案]]
