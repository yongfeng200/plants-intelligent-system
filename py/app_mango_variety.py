#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""芒果品种识别 —— 预测页面 + 识别接口。

路由
----
GET  /mango_variety/test_predict_mango_variety          预测页面
POST /mango_variety/submit_and_predict_mango_variety    上传图片 → 8 个品种概率 + 品种介绍

模型
----
model_registry 的 ``mango_variety`` 任务：EfficientNet-B4（Mango Dataset 上从零训练），
权重 ``models/mango_variety_best_checkpoint.pth``（缺失时回退 ``mango_variety_model.pth``）。
"""

from pathlib import Path

from flask import Blueprint, jsonify, redirect, render_template, request, session
from PIL import Image

from model_registry import (
    ModelNotReady,
    PROJECT_ROOT,
    TASKS,
    model_meta,
    run_classification,
)
from ood_detection import rejection_payload, success_fields
from user import save_upload_record
from upload_storage import save_uploaded_file

VARIETY_INFO = {
    "Anwar Ratool": {
        "description": "Anwar Ratool 是巴基斯坦著名的芒果品种，果实中等大小，果肉细腻香甜，纤维较少，口感极佳。",
        "characteristics": ["果实重量：200–300 克", "成熟季节：6–7 月", "果肉颜色：金黄色", "甜度：高"],
    },
    "Chaunsa (Black)": {
        "description": "Chaunsa Black 是 Chaunsa 系列中的优质品种，果实较大，果皮呈深黄色至绿黄色，果肉香浓。",
        "characteristics": ["果实重量：250–400 克", "成熟季节：7–8 月", "果肉颜色：深黄色", "甜度：极高"],
    },
    "Chaunsa (Summer Bahisht)": {
        "description": "Summer Bahisht 是 Chaunsa 的夏季品种，以浓郁的香气和极佳的口感著称，是最受欢迎的品种之一。",
        "characteristics": ["果实重量：300–450 克", "成熟季节：7–8 月", "果肉颜色：金黄色", "甜度：极高"],
    },
    "Chaunsa (White)": {
        "description": "Chaunsa White 以独特的白色果肉和细腻的口感闻名，果实大，汁液丰富。",
        "characteristics": ["果实重量：300–400 克", "成熟季节：7–8 月", "果肉颜色：淡黄色", "甜度：高"],
    },
    "Dosehri": {
        "description": "Dosehri 是一种中等大小的芒果品种，果实椭圆形，果肉黄色细腻，纤维少，香气浓郁。",
        "characteristics": ["果实重量：150–250 克", "成熟季节：6–7 月", "果肉颜色：黄色", "甜度：中高"],
    },
    "Fajri": {
        "description": "Fajri 是早熟品种，果实较小但香气浓郁，果肉橙黄色，口感香甜。",
        "characteristics": ["果实重量：150–200 克", "成熟季节：5–6 月", "果肉颜色：橙黄色", "甜度：中"],
    },
    "Langra": {
        "description": "Langra 是印度—巴基斯坦传统品种，果实中等大小，果皮绿色即使成熟也不变黄，果肉芳香。",
        "characteristics": ["果实重量：200–300 克", "成熟季节：7–8 月", "果肉颜色：黄色", "甜度：高"],
    },
    "Sindhri": {
        "description": "Sindhri 是巴基斯坦信德省的著名品种，果实大、果形漂亮，果肉金黄色，纤维极少。",
        "characteristics": ["果实重量：300–500 克", "成熟季节：6–7 月", "果肉颜色：金黄色", "甜度：高"],
    },
}


def app_mango_variety_blueprint():
    """创建芒果品种识别蓝图。"""
    blueprint = Blueprint("mango_variety", __name__)

    # ---------------------------------------------------------------- 预测页面
    @blueprint.route("/test_predict_mango_variety")
    @blueprint.route("/test_predict2")
    def test_predict():
        # 未登录直接回登录页，避免页面半渲染
        if "login_name" not in session:
            return redirect("/")
        return render_template(
            "test_predict_mango_variety.html",
            model_ready=True,
            model_name=TASKS["mango_variety"]["architecture"],
        )

    # ---------------------------------------------------------------- 预测接口
    @blueprint.route("/submit_and_predict_mango_variety", methods=["POST"])
    def submit_and_predict():
        if "login_name" not in session:
            return jsonify({"error": "登录已过期，请重新登录"}), 401

        test_file = request.files.get("file")
        if test_file is None or not test_file.filename:
            return jsonify({"error": "请选择需要识别的图片"}), 400

        extension = Path(test_file.filename).suffix.lower()
        if extension not in {".png", ".jpg", ".jpeg"}:
            return jsonify({"error": "仅支持 PNG、JPG、JPEG 图片"}), 400

        # 先落盘再推理：即使后面预测失败，上传记录也留得住
        image_path, upload_url, filename = save_uploaded_file(test_file, "mango_variety")

        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
            result = run_classification("mango_variety", image)
        # 权重没准备好：返回 503 和排查提示，不用随机结果糊弄用户
        except ModelNotReady as exc:
            return jsonify({
                "error": str(exc),
                "hint": "请确认 models/mango_variety_best_checkpoint.pth 或 mango_variety_model.pth 存在。",
                "model": model_meta("mango_variety"),
            }), 503
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"图像处理失败: {exc}"}), 500

        info = VARIETY_INFO.get(result["predict_en"], {
            "description": "暂无详细信息",
            "characteristics": ["请咨询专业人员"],
        })

        # 不是芒果果实样本：不给预测，也不写进识别记录
        if not result["valid"]:
            print(
                f"[Mango Variety] 拒绝非本模型样本 · "
                f"{'；'.join(result['ood']['reasons']) or result['ood']['message']}"
            )
            return jsonify(rejection_payload(
                result["ood"], result["model"], {"upload_image": upload_url}
            ))

        # 写库失败只记日志，不能因为统计信息失败让用户看不到识别结果
        try:
            save_upload_record(
                user_name=session["login_name"],
                image_name=filename,
                image_path=upload_url,
                disease_class=result["predict"],
                model_type=result["model_type"],
                confidence=result["confidence"],
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[Mango Variety] 保存记录失败: {exc}")

        response = {
            **success_fields(result["ood"]),
            "upload_image": upload_url,
            "predict": result["predict"],
            "predict_cn": result["predict_cn"],
            "predict_en": result["predict_en"],
            "confidence": round(result["confidence"], 2),
            "pred_pros": result["pred_pros"],
            "variety_labels": result["labels"],
            "variety_labels_en": result["labels_en"],
            "description": info["description"],
            "characteristics": info["characteristics"],
            "model": result["model"],
        }
        print(f"[Mango Variety] {result['predict_en']} · {response['confidence']:.2f}% · {result['elapsed_ms']:.0f} ms")
        return jsonify(response)

    return blueprint
