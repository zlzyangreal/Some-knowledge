# DSH 工作区从 F: 搬到 D: 后会话丢失：排查与修复

> 适用：改过 workspace.json 里的工作区路径、或把工作区换过盘之后，DSH 出现「未分组」「暂无会话」「工作区选不了」。
> 数据没丢：会话都在 `%USERPROFILE%\.dsh\sessions`。这次修复用的脚本、备份、DSH 源码都留在 `D:\codex_workspace`。

## 一、问题在哪：三处路径必须一致

DSH 把工作区路径编码成会话目录名：字母、数字和 `._-` 原样保留，`:`、`\`、`/` 连续出现压成一个 `-`，其他字符转 `~XXXX`，外层再包 `--...--`。例如 `D:\deepseek_workspace\note` → `--D-deepseek_workspace-note--`。

一条会话要被正确找到，三处必须同时成立：

1. `workspace.json` 里工作区的 `path`
2. `sessions` 下的目录名（由 path 编码而来）
3. 会话文件头里的 `cwd`

这次踩到的三种状态，正好对应三处：

- 只改了 1 → 按新路径找不到目录，会话全落到「未分组」。
- 把旧目录**复制**成新目录 → 同一 session ID 出现两份，启动报 `duplicate JSONL session id "..." appears in multiple project directories`，会话服务和工作区服务都起不来，界面「暂无会话」。看到这条报错基本就能确诊。
- 复制不改文件内容 → 会话头 `cwd` 还是旧路径；即使 1、2 都改对了，DSH 仍判定会话不属于该工作区。

## 二、修复步骤

前提：**先完全退出 DSH**（托盘退出，任务管理器确认没有 `DeepSeek Harness` 进程）。运行中的 DSH 会覆盖你改的状态文件。

### 1. 备份

```powershell
$stamp = Get-Date -Format 'yyyyMMdd-HHmm'
$bak = "D:\codex_workspace\dsh-recovery-$stamp"
robocopy "$env:USERPROFILE\.dsh\sessions" "$bak\sessions" /E | Out-Null
robocopy "$env:USERPROFILE\.dsh\storages" "$bak\storages" /E | Out-Null
```

robocopy 退出码 0–7 都算成功。出问题就把 `$bak` 里的两个目录复制回去。

### 2. 看现状

用现成的校验脚本，把每个会话的目录、头 ID、cwd 列出来：

```powershell
$s = "$env:USERPROFILE\.dsh\sessions"
node D:\codex_workspace\validate_dsh_sessions.mjs $s > D:\codex_workspace\sessions-check.json
$rows = Get-Content D:\codex_workspace\sessions-check.json -Raw | ConvertFrom-Json
$rows | Select-Object project, directoryId, headerId, cwd, idMatchesDirectory | Format-Table -AutoSize

# 头 ID 重复（有输出 = 存在同一会话两份）
$rows | Group-Object headerId | Where-Object Count -gt 1 | Select-Object Count, Name
```

再对一下配置路径与目录名：

```powershell
(Get-Content "$env:USERPROFILE\.dsh\storages\workspace.json" -Raw | ConvertFrom-Json).tables.workspaces.PSObject.Properties.Value |
  Select-Object title, path
Get-ChildItem $s -Directory | Select-Object -ExpandProperty Name
```

判断：`cwd` 还是 F: → 做第 6 步；有重复 headerId → 先做第 3 步；配置 path 与目录名对不上 → 做第 4、5 步。

后面的命令都在同一个 PowerShell 窗口里执行（`$s` 已在上面定义）；另开窗口就先重跑第 2 步的第一行。

### 3. 去重：只留一份

```powershell
$s = "$env:USERPROFILE\.dsh\sessions"
$q = "$env:USERPROFILE\.dsh\recovery-quarantine-$(Get-Date -Format 'yyyyMMdd-HHmm')"
New-Item -ItemType Directory -Force $q | Out-Null
Move-Item "$s\--F-*" $q        # 旧的整组挪进隔离目录，不删
```

原则：**move，不 delete，不 copy**。保留与新路径一致的那份（一般是 `--D-*`）。如果重复只出现在个别会话上，就移动单个 `session-<id>` 目录。

### 4. 目录改名

```powershell
Get-ChildItem $s -Directory -Filter '--F-*' | ForEach-Object {
  $new = $_.Name -replace '^--F-', '--D-'
  if (Test-Path (Join-Path $s $new)) { "跳过（已存在）: $new" }
  else { Rename-Item $_.FullName $new }
}
```

改名、改配置、改会话头全部做完之前，不要启动 DSH。

### 5. 改 workspace.json 的 path

```powershell
$p = "$env:USERPROFILE\.dsh\storages\workspace.json"
Copy-Item $p "$p.bak_migrate" -Force
$old = 'F:\\deepseek_workspace'      # JSON 文件里路径是双反斜杠
$new = 'D:\\deepseek_workspace'
$utf8 = New-Object System.Text.UTF8Encoding($false)   # 无 BOM
[System.IO.File]::WriteAllText($p, (Get-Content $p -Raw).Replace($old, $new), $utf8)
```

验证：能解析，路径是 `D:\\...` 而不是 `D:\\\\...`：

```powershell
(Get-Content $p -Raw | ConvertFrom-Json).tables.workspaces.PSObject.Properties.Value |
  Select-Object title, path
```

不要手改：手改最容易反斜杠翻倍，或存成带 BOM 的 UTF-8。

### 6. 改会话头 cwd

用当时修复用的脚本（就在 `D:\codex_workspace`）：

```powershell
node D:\codex_workspace\repair_dsh_session_headers.mjs $s --check    # 只列出要改的，不写
node D:\codex_workspace\repair_dsh_session_headers.mjs $s --apply    # 写入
```

脚本做了什么：只处理目录名以 `--D-deepseek_workspace` 开头的工作区；只把 cwd 从 `F:\deepseek_workspace` 改成 `D:\deepseek_workspace`；**只重写第 1 帧**（头行），后面的消息帧原样拼回；写入前自校验（重新解压新头帧，并核对正文帧 sha256 未变）；先写 `.repair-temp` 再 rename 覆盖。它不生成 `.bak`，回滚靠第 1 步的整份备份。

注意：这个脚本是本次修复专用的，写死了旧路径、新路径、目录前缀，并要求命中数**正好 15 个**，数量或状态不对就直接报错，不会改半个文件。如果第 2 步里已经一个 F: 头都不剩了，跳过第 6 步；如果只剩一部分，脚本会报 `unexpected cwd`，这种情况让 AI 会话按同样思路逐会话处理。换路径或换数量时，改脚本里这几处：第 64 行目录前缀、65–66 行旧路径、68 行新路径、80 行期望数量。别手改二进制。

### 7. 投影缓存 + 启动验证

投影缓存里也存了 cwd。最省事的办法是把整个缓存目录改名，让 DSH 启动时重建：

```powershell
Rename-Item "$env:USERPROFILE\.dsh\storages\session_projcache" "session_projcache.old"
```

启动 DSH 后逐项确认：

- 重新跑第 2 步的校验：所有 `cwd` 都是 `D:` 开头，`headerId` 无重复
- 每个工作区点开，会话都在、能打开、消息完整
- 界面正常的话，`session_projcache.old` 可以删了

（健康状态下再跑 `repair_dsh_session_headers.mjs --check` 会报 `unexpected cwd` 或 `expected 15 ..., got 0`，这是正常的：它只适用于「15 个 D 目录、头里都是 F:」那一刻。）

## 三、下次迁移的正确顺序

1. 完全退出 DSH
2. robocopy 复制工作区数据（读源盘，安全）
3. **move** 会话目录：`--F-*` → `--D-*`，每个 session ID 只留一份
4. 改 workspace.json 的 path
5. 改会话头 cwd（盘符或路径变了才需要）
6. 启动 DSH，逐个工作区验证
7. 旧目录留作只读备份，不加回工作区列表

顺序的原因：配置是「指路」，目录是「住址」，会话头是「身份证地址」。先改任何一处再启动，都会出现「指向空目录」的中间态。

## 四、四条硬规则

1. 迁移只用 move，不用 copy——copy 必然产生重复 session ID。
2. 改 `.dsh` 下的状态文件之前先退出 DSH。
3. 手改 JSON 之后验证三件事：能解析、反斜杠没翻倍、无 BOM。
4. 隔离目录只挪不删，确认 DSH 正常后再清理。

## 五、本次现场

- 修复脚本：`D:\codex_workspace\repair_dsh_session_headers.mjs`、`D:\codex_workspace\validate_dsh_sessions.mjs`
- 备份：`D:\codex_workspace\dsh-recovery-20261026-2110`；旧目录隔离：`%USERPROFILE%\.dsh\recovery-quarantine-20261026-2115`（4 个 `--F-*`，未删）
- DSH 源码（查编码/校验规则用）：`D:\codex_workspace\dsh_asar_inspect`
- 现状：6 个工作区、19 个会话目录，无重复 headerId，会话头 cwd 全部为 D:
