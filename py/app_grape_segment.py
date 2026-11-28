#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""葡萄果实实例分割 —— 预测页面 + 检测接口。

路由
----
GET  /grape_segment/test_predict_grape_segment  预测页面
POST /grape_segment/submit_and_segment          上传图片 → 果实计数 + 品种统计 + 可视化

模型
----
model_registry 的 ``grape_segment`` 任务：YOLOv8-seg，权重 ``models/grape_segment_model.pt``，
3 个品种类别（霞多丽 / 灰皮诺 / 黑皮诺）。输入尺寸、置信度阈值等口径见任务定义。
"""

import base64
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Blueprint, jsonify, redirect, render_template, request, session

from model_registry import (
    GRAPE_VARIETY_CLASSES,
    ModelNotReady,
    PROJECT_ROOT,
    TASKS,
    get_detector_entry,
    model_meta,
)
from ood_detection import check_image_quality, rejection_payload, success_fields, task_hints
from user import save_upload_record
from upload_storage import save_uploaded_file

VARIETY_CN = {item["en"]: item["cn"] for item in GRAPE_VARIETY_CLASSES}


def _data_url(image_bgr: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("图像编码失败")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


def app_grape_segment_blueprint():
    """创建葡萄实例分割蓝图。"""
    blueprint = Blueprint("grape_segment", __name__)

    # ---------------------------------------------------------------- 预测页面
    @blueprint.route("/test_predict_grape_segment")
    def test_segment():
        # 未登录直接回登录页，避免页面半渲染
        if "login_name" not in session:
            return redirect("/")
        return render_template(
            "test_predict_grape_segment.html",
            class_names=[item["cn"] for item in GRAPE_VARIETY_CLASSES],
            model_ready=True,
            model_name=TASKS["grape_segment"]["architecture"],
        )

    # ---------------------------------------------------------------- 预测接口
    @blueprint.route("/submit_and_segment", methods=["POST"])
    def submit_and_segment():
        if "login_name" not in session:
            return jsonify({"error": "登录已过期，请重新登录"}), 401

        uploaded = request.files.get("file")
        if uploaded is None or not uploaded.filename:
            return jsonify({"error": "请选择需要分割的图片"}), 400

        extension = Path(uploaded.filename).suffix.lower()
        if extension not in {".png", ".jpg", ".jpeg"}:
            return jsonify({"error": "仅支持 PNG、JPG、JPEG 图片"}), 400

        # 先落盘再推理：即使后面预测失败，上传记录也留得住
        image_path, upload_url, filename = save_uploaded_file(uploaded, "grape_segment")

        try:
            detector = get_detector_entry()
            started = time.perf_counter()
            results = detector.model.predict(
                source=str(image_path), conf=0.25, iou=0.7, verbose=False
            )[0]
            elapsed_ms = (time.perf_counter() - started) * 1000

            rendered = results.plot(boxes=True, labels=True, masks=True)
            result_url = _data_url(rendered)

            class_counts = {item["cn"]: 0 for item in GRAPE_VARIETY_CLASSES}
            confidences = []
            detections = []

            if results.boxes is not None:
                names = getattr(results, "names", {}) or detector.names
                for cls_id, conf in zip(results.boxes.cls.tolist(), results.boxes.conf.tolist()):
                    cls_id = int(cls_id)
                    en_name = str(names.get(cls_id, detector.names.get(cls_id, f"类别{cls_id}")))
                    cn_name = VARIETY_CN.get(en_name, en_name)
                    class_counts[cn_name] = class_counts.get(cn_name, 0) + 1
                    confidences.append(float(conf))
                    x1, y1, x2, y2 = [float(v) for v in results.boxes.xyxy[len(detections)].tolist()]
                    detections.append({
                        "class": cn_name,
                        "class_en": en_name,
                        "confidence": round(float(conf) * 100, 2),
                        "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    })

            grape_count = len(detections)
            average_confidence = float(np.mean(confidences) * 100) if confidences else 0.0
        # 权重没准备好：返回 503 和排查提示，不用随机结果糊弄用户
        except ModelNotReady as exc:
            return jsonify({
                "error": str(exc),
                "hint": "请确认 models/grape_segment_model.pt 存在，并安装 ultralytics。",
                "model": model_meta("grape_segment"),
            }), 503
        except Exception as exc:  # noqa: BLE001
            print(f"[Grape Segment] 推理失败: {exc}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"分割失败: {exc}"}), 500

        summary = " | ".join(f"{name}:{count}" for name, count in class_counts.items() if count > 0)

        # ---- 非本模型样本：图像本身不合格，或画面里根本没有葡萄果实
        try:
            quality = check_image_quality(cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB))
        except Exception:  # noqa: BLE001 - 读图失败就跳过这一层，别影响分割结果
            quality = []
        hard = [item for item in quality if item["level"] == "reject"]

        if hard or grape_count == 0:
            reasons = [item["message"] for item in hard] or [
                "画面中没有检测到任何葡萄果实（置信度阈值 25%）"
            ]
            print(f"[Grape Segment] 拒绝非本模型样本 · {'；'.join(reasons)}")
            return jsonify(rejection_payload(
                {
                    "status": "rejected",
                    "score": None,
                    "scores": {},
                    "reasons": reasons,
                    "message": "这张图片不像能识别的葡萄果实样本，系统没有输出分割结果。",
                    "hints": task_hints("grape_segment", calibrated=False),
                    "quality": quality,
                    "calibrated": False,
                },
                model_meta("grape_segment", elapsed_ms),
                {"upload_image": upload_url},
            ))

        # 只检测到零星果实：能给结果，但提醒一下可能拍得不全
        ood_like = {
            "status": "suspicious" if grape_count < 2 else "ok",
            "score": None,
            "scores": {},
            "reasons": [item["message"] for item in quality if item["level"] == "warn"],
            "message": "画面中只检测到 1 颗果实，分割结果可能不完整。" if grape_count < 2 else "",
            "hints": task_hints("grape_segment", calibrated=False),
            "quality": quality,
            "calibrated": False,
        }

        # 写库失败只记日志，不能因为统计信息失败让用户看不到识别结果
        try:
            save_upload_record(
                user_name=session["login_name"],
                image_name=filename,
                image_path=upload_url,
                disease_class=summary or "未检测到葡萄",
                model_type=TASKS["grape_segment"]["model_type"],
                confidence=average_confidence,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[Grape Segment] 保存记录失败: {exc}")

        response = {
            **success_fields(ood_like),
            "upload_image": upload_url,
            "result_base64": result_url,
            "grape_count": grape_count,
            "confidence": round(average_confidence, 2),
            "class_counts": class_counts,
            "detections": detections,
            "model": model_meta("grape_segment", elapsed_ms),
            "elapsed_ms": round(elapsed_ms, 2),
            "message": "分割完成",
        }
        print(f"[Grape Segment] 检测到 {grape_count} 颗 · 平均置信度 {average_confidence:.2f}% · {elapsed_ms:.0f} ms")
        return jsonify(response)

    return blueprint
