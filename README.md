# LLM_learn
技术学习社区--个人知识网络

## 上传文件夹到 GitHub

仓库地址：

```text
https://github.com/oppokj/LLM_learn.git
```

如果本地已经有仓库目录，例如：

```powershell
D:\Users\Learn\LLM_learn
```

先把要上传的文件夹复制到这个仓库目录中，然后选择下面任意一种方式上传。

### 方式一：HTTPS 上传

适合网络可以正常访问 `https://github.com` 的情况。

```powershell
cd D:\Users\Learn\LLM_learn

git status
git add 要上传的文件夹名
git commit -m "add folder"
git push origin main
```

例如上传 `Technical_Report`：

```powershell
cd D:\Users\Learn\LLM_learn

git add Technical_Report
git commit -m "add Technical_Report"
git push origin main
```

如果 `git push` 提示连接不上 `github.com:443`，并且本机有代理，可以给 Git 配置代理后再推送。下面以代理端口 `7890` 为例：

```powershell
git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890

cd D:\Users\Learn\LLM_learn
git push origin main
```

如果之后不想让 Git 继续使用代理，可以取消配置：

```powershell
git config --global --unset http.proxy
git config --global --unset https.proxy
```

### 方式二：SSH 443 上传

适合 HTTPS 访问 GitHub 失败，但 `ssh.github.com:443` 可以连接的情况。

第一次使用时，需要先生成 SSH key：

```powershell
ssh-keygen -t ed25519 -f C:\Users\kexu18\.ssh\id_ed25519_github -C "kexu18@github"
```

查看公钥内容：

```powershell
Get-Content C:\Users\kexu18\.ssh\id_ed25519_github.pub
```

然后打开 GitHub SSH Keys 页面：

```text
https://github.com/settings/keys
```

点击 `New SSH key`，把上一步输出的公钥粘贴进去并保存。

接着把当前仓库改成走 SSH 443：

```powershell
cd D:\Users\Learn\LLM_learn

git remote set-url origin ssh://git@ssh.github.com:443/oppokj/LLM_learn.git
git config core.sshCommand "ssh -i C:/Users/kexu18/.ssh/id_ed25519_github -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
```

验证 SSH 是否可用：

```powershell
ssh -i C:\Users\kexu18\.ssh\id_ed25519_github -o IdentitiesOnly=yes -T -p 443 git@ssh.github.com
```

如果看到类似下面的提示，说明认证成功：

```text
Hi oppokj! You've successfully authenticated, but GitHub does not provide shell access.
```

之后上传文件夹：

```powershell
cd D:\Users\Learn\LLM_learn

git status
git add 要上传的文件夹名
git commit -m "add folder"
git push origin main
```

例如上传 `Technical_Report`：

```powershell
cd D:\Users\Learn\LLM_learn

git add Technical_Report
git commit -m "add Technical_Report"
git push origin main
```

### 常用检查命令

查看当前分支和待提交内容：

```powershell
git status --short --branch
```

查看远程仓库地址：

```powershell
git remote -v
```

查看最近一次提交：

```powershell
git log -1 --stat --oneline
```

注意：GitHub 普通 Git 上传不支持超过 100MB 的单个文件。如果文件过大，需要使用 Git LFS。
