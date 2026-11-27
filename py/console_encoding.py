#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
控制台编码配置
==============

解决 Windows 下两种相反的乱码场景：

1. **真实控制台**（PowerShell / Windows Terminal / cmd）：
   把代码页切到 UTF-8（CP65001），Python 输出 UTF-8，中文正常。

2. **PyCharm Run 控制台 / IDE / 输出重定向**：
   这些场景下 stdout 是管道，PyCharm 默认按系统 ANSI 编码（简体中文为
   GBK/CP936）解码。如果 Python 强制输出 UTF-8，就会看到
   ``鈥? 鐗╂...`` 这类乱码。因此这里保持 IDE 期望的编码，只把
   无法编码的符号替换掉，避免 UnicodeEncodeError。

可通过环境变量覆盖：
    PLANT_CONSOLE_ENCODING=utf-8   # 或在 PyCharm 里设置 UTF-8 控制台编码
    PLANT_CONSOLE_ENCODING=gbk
"""

import os
import sys

_DESIRED_ENCODING = None


def _windows_ansi_encoding() -> str:
    """返回 Windows 当前 ANSI 代码页对应的 Python 编码名（中文系统为 cp936）。"""
    try:
        import ctypes

        code_page = ctypes.windll.kernel32.GetACP()
        if code_page:
            return f"cp{code_page}"
    except Exception:
        pass

    try:
        import locale

        return locale.getpreferredencoding(False) or "utf-8"
    except Exception:
        return "utf-8"


def _is_interactive_console() -> bool:
    try:
        return bool(sys.stdout and sys.stdout.isatty())
    except Exception:
        return False


def _reconfigure(stream, encoding: str, errors: str = "replace") -> None:
    """安全地重设文本流编码；不支持时保持原样。"""
    if stream is None or not encoding:
        return
    try:
        stream.reconfigure(encoding=encoding, errors=errors)
    except (AttributeError, OSError, ValueError):
        # 某些 IDE / 日志接管对象不支持 reconfigure，保持默认即可
        pass


def setup_console_encoding() -> str:
    """根据运行环境配置标准流，返回最终使用的编码名。"""
    global _DESIRED_ENCODING
    override = (os.environ.get("PLANT_CONSOLE_ENCODING") or "").strip()
    if not override:
        # 尊重用户显式设置的 PYTHONIOENCODING（如 PyCharm 控制台设为 UTF-8）
        override = (os.environ.get("PYTHONIOENCODING") or "").split(":")[0].strip()

    if os.name != "nt":
        encoding = override or "utf-8"
        for stream in (sys.stdin, sys.stdout, sys.stderr):
            _reconfigure(stream, encoding)
        _DESIRED_ENCODING = encoding
        return encoding

    # PyCharm 的 Run 控制台按系统 ANSI 编码解码；不要强行改成 UTF-8。
    is_pycharm = os.environ.get("PYCHARM_HOSTED") == "1"
    interactive = _is_interactive_console() and not is_pycharm

    if interactive and not override:
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleCP(65001)
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except Exception:
            pass
        encoding = "utf-8"
    else:
        encoding = override or _windows_ansi_encoding()

    for stream in (sys.stdin, sys.stdout, sys.stderr):
        _reconfigure(stream, encoding)
    _DESIRED_ENCODING = encoding
    return encoding


def restore_console_encoding() -> str:
    """重新应用控制台编码。

    部分第三方库（例如 ultralytics 在导入时会调用
    ``sys.stdout.reconfigure(encoding="utf-8")``）会打乱已经设置好的编码，
    导致 PyCharm 控制台再次出现中文乱码。导入这些库后调用本函数即可恢复。
    """
    global _DESIRED_ENCODING
    encoding = _DESIRED_ENCODING or setup_console_encoding()
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        _reconfigure(stream, encoding)
    return encoding


# 模块导入时自动执行
CONSOLE_ENCODING = setup_console_encoding()
