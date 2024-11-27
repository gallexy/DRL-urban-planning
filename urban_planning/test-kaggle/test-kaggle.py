import os
import json
import subprocess

# 定义元数据
metadata = {
    "id": None,
    "title": "My Kaggle testing",
    "code_file": "run-torch.py",  # 替换为你的代码文件名
    "language": "python",
    "kernel_type": "script",    # 如果是 notebook，改为 "notebook"
    "is_private": True,
    "enable_gpu": False,
    "enable_internet": True,
    "enable_run_on_commit": True,
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": []
}

# 将元数据保存为 kernel-metadata.json
#with open('kernel-metadata.json', 'w') as f:
#    json.dump(metadata, f, indent=4)

# 使用 Kaggle API 上传 Kernel
#subprocess.run(['kaggle', 'kernels', 'push'])

# 可选：检查 Kernel 状态
subprocess.run(['kaggle', 'kernels', 'status', 'liuxiaohua72/kaggletesting'])