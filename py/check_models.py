#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型自检脚本
============

用途：在不启动 Flask 的情况下检查 4 个模型文件、网络结构、预处理是否匹配，
并打印推理设备与测试准确率。

运行：
    python py/check_models.py

说明：叶片病灶分割 + 物种/疾病识别不在这张表里（它不走 model_registry），
自检请单独执行：
    python -c "import sys; sys.path.insert(0, 'py'); from leaf_disease_models import get_pipeline; print(get_pipeline().status())"
"""

import sys
from pathlib import Path

# 先配置控制台编码，避免 Windows GBK 控制台输出中文/符号报错
sys.path.insert(0, str(Path(__file__).resolve().parent))
import console_encoding  # noqa: F401,E402

from model_registry import TASKS, device_name, preload_models, REGISTRY, MODELS_DIR  # noqa: E402


def main() -> int:
    print("=" * 78)
    print("植物智能识别系统 · 模型自检")
    print("=" * 78)
    print(f"推理设备: {device_name()}")
    print(f"模型目录: {MODELS_DIR}")
    print("-" * 78)

    print("正在加载模型…")
    errors = preload_models(list(TASKS.keys()))

    for item in REGISTRY.status():
        if item["loaded"]:
            state = "[OK] 已加载"
        elif item["exists"]:
            state = "[!] 文件存在但未加载"
        else:
            state = "[X] 文件缺失"
        accuracy = item["accuracy_text"] if item["accuracy"] is not None else "—"
        print(f"{state:<16} {item['name']:<20} {item['architecture']:<24} "
              f"{item['size_mb'] or 0:>8.2f} MB  准确率 {accuracy}")
        print(f"                  {item['path']}")
        if item.get("error"):
            print(f"                  错误: {item['error']}")

    print("-" * 78)
    if errors:
        print("部分模型加载失败：")
        for key, message in errors.items():
            print(f"  - {TASKS[key]['name']}: {message}")
        return 1

    print("全部模型加载成功，可以启动：python py/main.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
