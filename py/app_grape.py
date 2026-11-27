#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""葡萄病害识别 —— 预测页面 + 识别接口。

路由
----
GET  /grape/test_predict_grape         预测页面
POST /grape/submit_and_predict_grape   上传图片 → 4 类葡萄病害概率 + 防治建议

模型
----
model_registry 的 ``grape_disease`` 任务：ResNet50 + CBAM，
权重 ``models/grape_disease_model.pth``（缺失时回退 ``grape_disease_best_checkpoint.pth``）。
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
from user import save_upload_record
from upload_storage import save_uploaded_file

DISEASE_INFO = {
    "Black Rot": {
        "description": "黑腐病是葡萄最严重的真菌病害之一，主要危害果实、叶片和新梢。果实染病后出现褐色圆形病斑，"
                       "逐渐扩大，果实干缩成僵果；叶片病斑呈褐色、边缘深色。",
        "advice": [
            "农业防治：清除病残体，减少病源",
            "化学防治：喷施代森锰锌或多菌灵",
            "防治时期：花前至幼果期重点防治",
            "预防措施：改善通风条件，降低湿度",
        ],
    },
    "ESCA": {
        "description": "黑麻疹又称埃斯卡病（ESCA），是一种复杂的真菌病害，主要危害老龄葡萄树。"
                       "叶片出现黄化和枯死，果实表面出现黑色斑点，严重时整株枯死。",
        "advice": [
            "修剪管理：剪除病枝，减少传播",
            "伤口保护：修剪后及时涂抹保护剂",
            "树势管理：加强水肥管理，增强抗性",
            "定期检查：及早发现并处理病株",
        ],
    },
    "Healthy": {
        "description": "葡萄叶片健康，生长良好，叶色浓绿，无病害症状。继续保持良好的栽培管理措施。",
        "advice": [
            "定期巡查：观察植株生长状况",
            "均衡施肥：保持营养供应充足",
            "适当灌溉：避免过干或过湿",
            "预防为主：做好病虫害预防工作",
        ],
    },
    "Leaf Blight": {
        "description": "叶斑病是由真菌引起的常见病害，主要危害叶片。病斑初期为褐色小点，后扩大成不规则形状，"
                       "严重时导致早期落叶，影响果实品质。",
        "advice": [
            "清园管理：清除落叶，减少越冬病源",
            "药剂防治：喷施波尔多液或甲基托布津",
            "防治时期：生长季节定期预防",
            "栽培管理：合理密植，改善通风透光",
        ],
    },
}


def app_grape_blueprint():
    """创建葡萄病害识别蓝图。"""
    blueprint = Blueprint("grape", __name__)

    # ---------------------------------------------------------------- 预测页面
    @blueprint.route("/test_predict_grape")
    def test_predict():
        # 未登录直接回登录页，避免页面半渲染
        if "login_name" not in session:
            return redirect("/")
        return render_template(
            "test_predict_grape.html",
            model_ready=True,
            model_name=TASKS["grape_disease"]["architecture"],
        )

    # ---------------------------------------------------------------- 预测接口
    @blueprint.route("/submit_and_predict_grape", methods=["POST"])
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
        image_path, upload_url, filename = save_uploaded_file(test_file, "grape")

        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
            result = run_classification("grape_disease", image)
        # 权重没准备好：返回 503 和排查提示，不用随机结果糊弄用户
        except ModelNotReady as exc:
            return jsonify({
                "error": str(exc),
                "hint": "请确认 models/grape_disease_model.pth 存在，并检查服务器启动日志。",
                "model": model_meta("grape_disease"),
            }), 503
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"图像处理失败: {exc}"}), 500

        info = DISEASE_INFO.get(result["predict_en"], {
            "description": "暂无详细信息",
            "advice": ["建议咨询专业农技人员"],
        })

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
            print(f"[Grape Disease] 保存记录失败: {exc}")

        response = {
            "upload_image": upload_url,
            "predict": result["predict"],
            "predict_cn": result["predict_cn"],
            "predict_en": result["predict_en"],
            "confidence": round(result["confidence"], 2),
            "pred_pros": result["pred_pros"],
            "disease_labels": result["labels"],
            "disease_labels_en": result["labels_en"],
            "description": info["description"],
            "advice": info["advice"],
            "model": result["model"],
        }
        print(f"[Grape Disease] {result['predict_en']} · {response['confidence']:.2f}% · {result['elapsed_ms']:.0f} ms")
        return jsonify(response)

    return blueprint
