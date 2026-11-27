# -*- coding: utf-8 -*-
"""植物智能识别系统 - Flask 主应用。

1. 葡萄病害识别          ResNet50 + CBAM                    /grape
2. 葡萄果实分割识别      YOLOv8-seg                         /grape_segment
3. 芒果品种识别          EfficientNet-B4                    /mango_variety
4. 芒果叶片病害识别      MDCN（DenseNet-121）               /mango_leaf
5. 叶片病灶分割 + 识别   UNet++ + DINOv2 ViT-S/14           /leaf_disease

"""

# 首先导入编码配置（必须在最开始）
import console_encoding

import os
import sys
import time
from datetime import date, datetime
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, session, url_for

# 以 main.py 的上级目录作为项目根目录，避免从 py/ 启动时路径漂移
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = PROJECT_ROOT / "templates"
STATIC_DIR = PROJECT_ROOT / "static"
MODELS_DIR = PROJECT_ROOT / "models"
DATABASE_DIR = PROJECT_ROOT / "database"

# 导入用户认证模块
from user import (
    get_db_connection,
    get_user_uploads,
    query_user_uploads,
    user_endpoints,
)

from upload_storage import UPLOAD_ROOT

from model_registry import (
    REGISTRY,
    TASKS,
    device_name,
    model_meta,
    preload_models,
)

# 导入各个子应用蓝图（只导入存在的模块）
try:
    from app_mango_leaf import app_mango_leaf_blueprint

    HAS_MANGO_LEAF = True
except ImportError as exc:
    HAS_MANGO_LEAF = False
    print(f"[WARN] 芒果叶片病害模块未找到: {exc}")

try:
    from app_mango_variety import app_mango_variety_blueprint

    HAS_MANGO_VARIETY = True
except ImportError as exc:
    HAS_MANGO_VARIETY = False
    print(f"[WARN] 芒果品种识别模块未找到: {exc}")

try:
    from app_grape import app_grape_blueprint

    HAS_GRAPE = True
except ImportError as exc:
    HAS_GRAPE = False
    print(f"[WARN] 葡萄病害识别模块未找到: {exc}")

try:
    from app_grape_segment import app_grape_segment_blueprint

    HAS_GRAPE_SEGMENT = True
except ImportError as exc:
    HAS_GRAPE_SEGMENT = False
    print(f"[WARN] 葡萄分割模块未找到: {exc}")

try:
    from app_leaf_disease import leaf_disease_blueprint

    HAS_LEAF_DISEASE = True
except ImportError as exc:
    HAS_LEAF_DISEASE = False
    print(f"[WARN] 叶片病害分割识别模块未找到: {exc}")

# 创建 Flask 应用，显式绑定项目根目录下的前端资源
app = Flask(
    __name__,
    template_folder=str(TEMPLATES_DIR),
    static_folder=str(STATIC_DIR),
    static_url_path="/static",
)
app.secret_key = os.environ.get("PLANT_SECRET_KEY", os.urandom(24).hex())
app.config["JSON_AS_ASCII"] = False
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB
# 改了 templates/ 里的页面后刷新浏览器就能看到，不用重启进程
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True

# 初始化用户认证路由
user_endpoints(app)

# 注册存在的蓝图
if HAS_MANGO_LEAF:
    app.register_blueprint(app_mango_leaf_blueprint(), url_prefix="/mango_leaf")
    print("[OK] 芒果叶片病害识别模块已加载")

if HAS_MANGO_VARIETY:
    app.register_blueprint(app_mango_variety_blueprint(), url_prefix="/mango_variety")
    print("[OK] 芒果品种识别模块已加载")

if HAS_GRAPE:
    app.register_blueprint(app_grape_blueprint(), url_prefix="/grape")
    print("[OK] 葡萄病害识别模块已加载")

if HAS_GRAPE_SEGMENT:
    app.register_blueprint(app_grape_segment_blueprint(), url_prefix="/grape_segment")
    print("[OK] 葡萄果实分割识别模块已加载")

if HAS_LEAF_DISEASE:
    app.register_blueprint(leaf_disease_blueprint(), url_prefix="/leaf_disease")
    print("[OK] 叶片病害分割识别模块已加载")


# ==================== 页面路由 ====================

def _page_guard():
    """未登录时跳转首页；返回重定向响应或 None。"""
    if "login_name" not in session:
        return redirect(url_for("home"))
    return None


@app.route("/")
def home():
    """首页 - 登录页面"""
    return render_template("index.html")


def _leaf_stats():
    """叶片分割+识别管线的统计信息；模块没加载就返回空字典。"""
    if not HAS_LEAF_DISEASE:
        return {}
    try:
        from leaf_disease_models import pipeline_status
        return pipeline_status()
    except Exception as exc:  # noqa: BLE001
        print(f"[Dashboard] 读取叶片模型状态失败: {exc}")
        return {}


@app.route("/dashboard")
def dashboard():
    """仪表板。

    模型数 / 类别数按「注册表 + 叶片管线」实时计算，不写死，避免以后加模块统计过期。
    叶片那条是 UNet++ 和 DINOv2 两条网络，所以模型 +2、能力 +1。
    """
    guard = _page_guard()
    if guard:
        return guard

    model_count = len(TASKS)
    class_count = sum(len(task.get("classes") or []) for task in TASKS.values())
    ability_count = len(TASKS)

    leaf = _leaf_stats()
    if leaf:
        if leaf.get("segment_exists") and leaf.get("classifier_exists"):
            model_count += 2
        class_count += len(leaf.get("species") or []) + len(leaf.get("diseases") or [])
        ability_count += 1

    return render_template(
        "dashboard.html",
        username=session.get("login_name"),
        model_count=model_count,
        class_count=class_count,
        ability_count=ability_count,
        device_label="CUDA" if "CUDA" in device_name() else "CPU",
    )


@app.route("/upload")
def upload():
    """上传记录页面"""
    guard = _page_guard()
    if guard:
        return guard
    return render_template("upload.html")


@app.route("/statistics")
def statistics():
    """统计页面"""
    guard = _page_guard()
    if guard:
        return guard
    return render_template("statistics.html")


_PRELOAD_TRIED = False


def _ensure_models_preloaded():
    """无论用 python main.py 还是 flask run 启动，状态接口都能反映真实加载情况。"""
    global _PRELOAD_TRIED
    if _PRELOAD_TRIED:
        return
    _PRELOAD_TRIED = True
    preload_models(list(TASKS.keys()))


def _leaf_model_entry():
    """把「叶片病灶分割 + 物种/疾病识别」包装成一条 registry 记录。

    它不走 model_registry（两条网络 + 一套预处理），但状态页和顶部徽章都应该算上它，
    否则会出现「仪表板写 6 个模型、徽章写 4/4」这种自相矛盾。
    """
    leaf = _leaf_stats()
    if not leaf:
        return None
    try:
        from leaf_disease_models import CLS_CHECKPOINT as leaf_cls, SEG_CHECKPOINT as leaf_seg
    except Exception:  # noqa: BLE001
        return None

    ready = bool(leaf.get("segment_exists") and leaf.get("classifier_exists"))
    total_mb = sum(p.stat().st_size for p in (leaf_seg, leaf_cls) if p.exists()) / (1024 * 1024)
    return {
        "key": "leaf_disease",
        "name": "叶片病灶分割 + 物种识别",
        "architecture": "UNet++（{}）+ {}".format(
            leaf.get("segment_name", "EfficientNet-B0"),
            leaf.get("classifier_name", "DINOv2 ViT-S/14"),
        ),
        "path": str(leaf_cls),
        "filename": leaf_cls.name,
        "exists": ready,
        "loaded": ready,          # 这里表示「权重就绪」，叶片模型是按需加载的
        "status": "权重就绪" if ready else "文件缺失",
        "size_mb": round(total_mb, 2),
        "accuracy": 0.7126,
        "accuracy_label": "联合准确率",
        "accuracy_text": "71.26%",
        "accuracy_hint": "87 张测试集：物种 86.21% / 病害 81.61% / 联合 71.26%；分割 Dice 85.11%",
        "device": device_name(),
        "classes": list(leaf.get("species") or []) + list(leaf.get("diseases") or []),
        "classes_en": list(leaf.get("species") or []) + list(leaf.get("diseases") or []),
        "error": None,
    }


def _model_status_payload():
    _ensure_models_preloaded()
    models = REGISTRY.status()
    leaf_entry = _leaf_model_entry()
    if leaf_entry:
        models = models + [leaf_entry]
    loaded = [item for item in models if item["loaded"]]
    return {
        "models": models,
        "all_loaded": all(item["exists"] for item in models),
        "loaded_count": len(loaded),
        "total_count": len(models),
        "device": device_name(),
        "cuda": device_name().startswith("CUDA"),
    }


@app.route("/model_status")
def model_status():
    """模型状态页面"""
    guard = _page_guard()
    if guard:
        return guard

    payload = _model_status_payload()
    models = payload["models"]
    missing = [item for item in models if not item["exists"]]
    return render_template(
        "model_status.html",
        models=models,
        all_loaded=payload["all_loaded"],
        missing=missing,
        device=payload["device"],
        loaded_count=payload["loaded_count"],
        total_count=payload["total_count"],
    )


@app.route("/profile")
def profile():
    """个人资料页面"""
    guard = _page_guard()
    if guard:
        return guard

    conn = get_db_connection()
    cursor = conn.cursor()
    user_name = session.get("login_name")
    cursor.execute(
        "SELECT username, email, datetime(created_at, 'localtime') AS created_at "
        "FROM users WHERE username = ?",
        (user_name,),
    )
    user = cursor.fetchone()
    conn.close()

    return render_template(
        "profile.html",
        username=user["username"] if user else user_name,
        email=user["email"] if user else None,
        created_at=user["created_at"] if user else None,
    )


@app.route("/settings")
def settings():
    """系统设置页面"""
    guard = _page_guard()
    if guard:
        return guard
    return render_template("settings.html")


# ==================== 模型状态 API ====================

@app.route("/api/model_status", methods=["GET"])
def api_model_status():
    """模型加载状态（供页面自动刷新）"""
    payload = _model_status_payload()
    payload["success"] = True
    return jsonify(payload)


@app.route("/api/model_info", methods=["GET"])
def api_model_info():
    """兼容旧接口：以模型 key 返回模型信息。"""
    result = {}
    for item in REGISTRY.status():
        result[item["key"]] = {
            "name": item["name"],
            "model": item["architecture"],
            "classes": len(item["classes"]),
            "path": item["path"],
            "status": "loaded" if item["loaded"] else ("exists" if item["exists"] else "not_found"),
            "accuracy": item["accuracy"],
            "device": item["device"],
            "url": {
                "mango_leaf": "/mango_leaf/test_predict_mango_leaf",
                "mango_variety": "/mango_variety/test_predict_mango_variety",
                "grape_disease": "/grape/test_predict_grape",
                "grape_segment": "/grape_segment/test_predict_grape_segment",
            }.get(item["key"], ""),
        }
    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    """健康检查：确认服务与模型状态。"""
    payload = _model_status_payload()
    return jsonify({
        "status": "ok",
        "device": payload["device"],
        "models_loaded": payload["loaded_count"],
        "models_total": payload["total_count"],
        "all_files_exists": payload["all_loaded"],
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })


# ==================== 用户记录 API ====================


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    """提供 uploads 目录下的上传图片，仅登录用户可访问。"""
    if "login_name" not in session:
        return redirect(url_for("home"))
    return send_from_directory(UPLOAD_ROOT, filename)


def _normalise_image_path(value: str) -> str:
    """数据库中的图片路径统一成 /static/... 形式。"""
    path = (value or "").replace("\\", "/")
    if not path.startswith("/"):
        path = "/" + path
    return path


def _safe_project_path(image_path: str):
    """只允许删除 static / uploads 目录内的文件。"""
    relative = (image_path or "").replace("\\", "/").lstrip("/")
    if not relative:
        return None
    candidate = (PROJECT_ROOT / relative).resolve()
    allowed_roots = [
        (STATIC_DIR).resolve(),
        (PROJECT_ROOT / "uploads").resolve(),
    ]
    if any(root == candidate or root in candidate.parents for root in allowed_roots):
        return candidate
    return None


@app.route("/api/user_uploads", methods=["GET"])
def api_user_uploads():
    """获取用户上传记录（支持筛选与分页）。"""
    if "login_name" not in session:
        return jsonify({"error": "未登录"}), 401

    page = max(1, request.args.get("page", 1, type=int))
    per_page = request.args.get("per_page", type=int)
    if per_page is None:
        per_page = request.args.get("limit", 9, type=int)
    per_page = min(max(1, per_page or 9), 60)
    model_type = (request.args.get("model_type") or "").strip()
    disease_class = (request.args.get("disease_class") or "").strip()

    rows, total = query_user_uploads(
        session["login_name"],
        page=page,
        per_page=per_page,
        model_type=model_type or None,
        disease_class=disease_class or None,
    )

    data = []
    for row in rows:
        item = dict(row)
        item["image_path"] = _normalise_image_path(item.get("image_path"))
        item["upload_time"] = item.get("created_at")
        data.append(item)

    pages = (total + per_page - 1) // per_page if total else 0
    return jsonify({
        "success": True,
        "data": data,
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": pages,
    })


@app.route("/api/delete_record/<int:record_id>", methods=["DELETE"])
def delete_record(record_id):
    """删除上传记录。"""
    if "login_name" not in session:
        return jsonify({"success": False, "error": "未登录"}), 401

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT image_path FROM user_uploads WHERE id = ? AND user_name = ?",
            (record_id, session["login_name"]),
        )
        record = cursor.fetchone()
        if not record:
            conn.close()
            return jsonify({"success": False, "error": "记录不存在或无权删除"}), 404

        cursor.execute("DELETE FROM user_uploads WHERE id = ?", (record_id,))
        conn.commit()
        conn.close()

        file_path = _safe_project_path(record["image_path"])
        if file_path and file_path.exists():
            try:
                file_path.unlink()
            except OSError as exc:
                print(f"[Records] 删除图片失败: {exc}")

        return jsonify({"success": True, "message": "删除成功"})
    except Exception as exc:  # noqa: BLE001
        print(f"[Records] 删除记录失败: {exc}")
        return jsonify({"success": False, "error": "删除失败，请稍后重试"}), 500


@app.route("/api/clear_all_records", methods=["DELETE"])
def api_clear_all_records():
    """清空当前用户的所有记录。"""
    if "login_name" not in session:
        return jsonify({"success": False, "error": "未登录"}), 401

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT image_path FROM user_uploads WHERE user_name = ?",
            (session["login_name"],),
        )
        records = cursor.fetchall()
        cursor.execute(
            "DELETE FROM user_uploads WHERE user_name = ?",
            (session["login_name"],),
        )
        conn.commit()
        conn.close()

        for record in records:
            file_path = _safe_project_path(record["image_path"])
            if file_path and file_path.exists():
                try:
                    file_path.unlink()
                except OSError as exc:
                    print(f"[Records] 删除图片失败: {exc}")

        return jsonify({"success": True, "message": "所有记录已清空"})
    except Exception as exc:  # noqa: BLE001
        print(f"[Records] 清空记录失败: {exc}")
        return jsonify({"success": False, "error": "清空失败"}), 500


@app.route("/api/delete_account", methods=["DELETE"])
def api_delete_account():
    """删除账户及其全部记录。"""
    if "login_name" not in session:
        return jsonify({"success": False, "error": "未登录"}), 401

    data = request.get_json(silent=True) or {}
    if data.get("username") != session["login_name"]:
        return jsonify({"success": False, "error": "用户名不匹配"}), 400

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT image_path FROM user_uploads WHERE user_name = ?",
            (session["login_name"],),
        )
        records = cursor.fetchall()
        cursor.execute("DELETE FROM user_uploads WHERE user_name = ?", (session["login_name"],))
        cursor.execute("DELETE FROM users WHERE username = ?", (session["login_name"],))
        conn.commit()
        conn.close()

        for record in records:
            file_path = _safe_project_path(record["image_path"])
            if file_path and file_path.exists():
                try:
                    file_path.unlink()
                except OSError as exc:
                    print(f"[Account] 删除图片失败: {exc}")

        session.clear()
        return jsonify({"success": True, "message": "账户已删除"})
    except Exception as exc:  # noqa: BLE001
        print(f"[Account] 删除账户失败: {exc}")
        return jsonify({"success": False, "error": "删除失败"}), 500


@app.route("/api/update_profile", methods=["POST"])
def api_update_profile():
    """更新用户资料。"""
    if "login_name" not in session:
        return jsonify({"success": False, "error": "未登录"}), 401

    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip()
    password = (data.get("password") or "").strip()

    if password and len(password) < 6:
        return jsonify({"success": False, "error": "密码长度至少 6 位"}), 400

    try:
        from werkzeug.security import generate_password_hash

        conn = get_db_connection()
        cursor = conn.cursor()
        if email:
            cursor.execute(
                "UPDATE users SET email = ? WHERE username = ?",
                (email, session["login_name"]),
            )
        if password:
            cursor.execute(
                "UPDATE users SET password_hash = ? WHERE username = ?",
                (generate_password_hash(password), session["login_name"]),
            )
        conn.commit()
        conn.close()
        return jsonify({"success": True, "message": "更新成功"})
    except Exception as exc:  # noqa: BLE001
        print(f"[Profile] 更新资料失败: {exc}")
        return jsonify({"success": False, "error": "更新失败"}), 500


@app.route("/api/statistics", methods=["GET"])
def api_statistics():
    """统计数据：总量、趋势、模型分布、病害排行、时段分布。"""
    if "login_name" not in session:
        return jsonify({"error": "未登录"}), 401

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        user_name = session["login_name"]

        cursor.execute(
            "SELECT model_type, COUNT(*) AS count FROM user_uploads "
            "WHERE user_name = ? GROUP BY model_type",
            (user_name,),
        )
        model_stats = [
            {"model": row["model_type"], "count": row["count"]}
            for row in cursor.fetchall()
        ]

        cursor.execute(
            "SELECT COUNT(*) AS total FROM user_uploads WHERE user_name = ?",
            (user_name,),
        )
        total = cursor.fetchone()["total"]

        cursor.execute(
            "SELECT COUNT(*) AS today_count FROM user_uploads "
            "WHERE user_name = ? AND DATE(created_at, 'localtime') = DATE('now', 'localtime')",
            (user_name,),
        )
        today_count = cursor.fetchone()["today_count"]

        cursor.execute(
            "SELECT DATE(created_at, 'localtime') AS date, COUNT(*) AS count FROM user_uploads "
            "WHERE user_name = ? AND datetime(created_at, 'localtime') >= datetime('now', 'localtime', '-7 days') "
            "GROUP BY DATE(created_at, 'localtime') ORDER BY date",
            (user_name,),
        )
        trend = [{"date": row["date"], "count": row["count"]} for row in cursor.fetchall()]

        cursor.execute(
            "SELECT disease_class, COUNT(*) AS count FROM user_uploads "
            "WHERE user_name = ? GROUP BY disease_class ORDER BY count DESC LIMIT 10",
            (user_name,),
        )
        disease_stats = [
            {"name": row["disease_class"], "count": row["count"]}
            for row in cursor.fetchall()
        ]

        cursor.execute(
            "SELECT AVG(confidence) AS avg_confidence FROM user_uploads "
            "WHERE user_name = ? AND confidence IS NOT NULL",
            (user_name,),
        )
        avg_row = cursor.fetchone()
        avg_confidence = float(avg_row["avg_confidence"] or 0)

        cursor.execute(
            "SELECT strftime('%H', created_at, 'localtime') AS hour, COUNT(*) AS count FROM user_uploads "
            "WHERE user_name = ? GROUP BY hour",
            (user_name,),
        )
        hour_rows = cursor.fetchall()
        conn.close()

        hourly = [0, 0, 0, 0]
        for row in hour_rows:
            if row["hour"] is None:
                continue
            bucket = min(3, int(row["hour"]) // 6)
            hourly[bucket] += row["count"]

        return jsonify({
            "success": True,
            "total": total,
            "today_count": today_count,
            "model_stats": model_stats,
            "trend": trend,
            "disease_stats": disease_stats,
            "avg_confidence": round(avg_confidence, 2),
            "hourly_stats": hourly,
            "device": device_name(),
        })
    except Exception as exc:  # noqa: BLE001
        print(f"[Statistics] 获取统计信息失败: {exc}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "获取统计信息失败"}), 500


# ==================== 错误处理 ====================

@app.errorhandler(404)
def not_found(_error):
    if request.path.startswith("/api/") or request.path == "/health":
        return jsonify({"error": "接口不存在"}), 404
    return (
        "<div style='font-family:system-ui;padding:60px;text-align:center'>"
        "<h1>404</h1><p>页面不存在，<a href='/dashboard'>返回仪表板</a></p></div>",
        404,
    )


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "图片超过 20MB，请压缩后重试"}), 413


@app.errorhandler(500)
def internal_error(_error):
    if request.path.startswith("/api/") or request.path == "/health":
        return jsonify({"error": "服务器内部错误"}), 500
    return (
        "<div style='font-family:system-ui;padding:60px;text-align:center'>"
        "<h1>500</h1><p>服务器内部错误，请查看控制台日志。</p></div>",
        500,
    )


# ==================== 启动应用 ====================

def _print_banner():
    payload = _model_status_payload()
    print("=" * 72)
    print("植物智能识别系统")
    print("=" * 72)
    print(f"访问地址: http://127.0.0.1:5000")
    print(f"推理设备: {payload['device']}")
    print(f"控制台编码: {console_encoding.CONSOLE_ENCODING}")
    print(f"模型状态: {payload['loaded_count']}/{payload['total_count']} 已加载")
    print("-" * 72)
    print("页面路由:")
    print("  登录          /")
    print("  仪表板        /dashboard")
    print("  芒果叶片病害  /mango_leaf/test_predict_mango_leaf")
    print("  芒果品种      /mango_variety/test_predict_mango_variety")
    print("  葡萄病害      /grape/test_predict_grape")
    print("  葡萄果实分割  /grape_segment/test_predict_grape_segment")
    print("  叶片分割+识别 /leaf_disease/test_predict")
    print("  模型状态      /model_status")
    print("=" * 72)
    print("测试账户: admin/admin123, test/test123")
    print("=" * 72)


if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (STATIC_DIR / "img").mkdir(parents=True, exist_ok=True)
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    print("正在加载模型…")
    started = time.time()
    _ensure_models_preloaded()
    errors = {item["key"]: item["error"] for item in REGISTRY.status() if item.get("error")}
    print(f"模型加载完成，用时 {time.time() - started:.1f}s")
    if errors:
        print("以下模型加载失败（预测接口会返回 503 并说明原因）:")
        for key, message in errors.items():
            print(f"  - {TASKS[key]['name']}: {message}")

    _print_banner()
    # 关闭 reloader，避免调试模式重复加载模型、占用额外显存
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
