# -*- coding: utf-8 -*-
"""上传文件统一存储。

目录结构：
    uploads/<module>/<YYYY-MM-DD>/<uuid>.<ext>

`module` 使用固定白名单，避免路径穿越。
文件由主应用统一对外提供（见 main.py 的 /uploads 路由），前端拿到的是普通 URL，
不需要再转 base64。
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_ROOT = PROJECT_ROOT / "uploads"
UPLOAD_URL_PREFIX = "/uploads"

MODULE_FOLDERS = {
    "grape",
    "grape_segment",
    "mango_leaf",
    "mango_variety",
    "leaf_disease",
}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
for _module in MODULE_FOLDERS:
    (UPLOAD_ROOT / _module).mkdir(parents=True, exist_ok=True)


def save_uploaded_file(file_storage, module: str) -> Tuple[Path, str, str]:
    """保存上传文件，返回 (本地路径, 前端 URL, 文件名)。"""
    if module not in MODULE_FOLDERS:
        raise ValueError(f"未知上传模块: {module}")

    suffix = Path(file_storage.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("仅支持 PNG、JPG、JPEG 图片")

    date_folder = datetime.now().strftime("%Y-%m-%d")
    target_dir = UPLOAD_ROOT / module / date_folder
    target_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{uuid.uuid4().hex}{suffix}"
    target_path = target_dir / filename
    file_storage.save(str(target_path))

    url = f"{UPLOAD_URL_PREFIX}/{module}/{date_folder}/{filename}"
    return target_path, url, filename
