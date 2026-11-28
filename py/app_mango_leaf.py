#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""芒果叶片病害识别 —— 预测页面 + 识别接口。

路由
----
GET  /mango_leaf/test_predict_mango_leaf          预测页面
POST /mango_leaf/submit_and_predict_mango_leaf    上传图片 → 8 类病害概率 + 防治建议

模型
----
model_registry 的 ``mango_leaf`` 任务：MDCN（DenseNet-121 + 分类头），
权重 ``models/mango_mdcn_best_checkpoint.pth``（回退 ``mango_mdcn_model.pth``），
224×224 + MangoLeaf 数据集自身的均值方差。
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

DISEASE_INFO = {
    "Anthracnose": {
        "description": "炭疽病是芒果最常见的真菌病害之一，主要危害叶片、花序和果实。病斑初期为褐色小点，"
                       "后扩大成不规则形状，中央灰白色、边缘深褐色，严重时造成落叶和落果。",
        "advice": [
            "农业防治：及时清除病叶病果，集中销毁，减少病源",
            "化学防治：喷施 70% 甲基托布津 800–1000 倍液",
            "防治时期：花期和幼果期重点防治",
            "预防措施：增施有机肥，提高树体抗病能力",
        ],
    },
    "Bacterial Canker": {
        "description": "细菌性溃疡病是一种严重的细菌性病害，可侵染叶片、枝条和果实。病斑呈水渍状，"
                       "后变褐色凹陷，周围有黄色晕圈，潮湿时可见菌脓。",
        "advice": [
            "修剪病枝：及时剪除病枝病叶并烧毁",
            "药剂防治：喷施 72% 农用链霉素 3000–4000 倍液",
            "预防为主：雨季前喷施保护性杀菌剂",
            "加强管理：避免偏施氮肥，增强树势",
        ],
    },
    "Cutting Weevil": {
        "description": "切叶象甲是一种害虫，成虫啃食叶片边缘呈半圆形缺刻，幼虫在叶片内部取食，"
                       "造成叶片卷曲枯萎，严重影响光合作用。",
        "advice": [
            "人工捕杀：震落成虫集中捕杀",
            "药剂防治：喷施 2.5% 溴氰菊酯 2000–3000 倍液",
            "防治时期：新梢期是防治关键期",
            "清园管理：冬季清除落叶，减少虫源",
        ],
    },
    "Die Back": {
        "description": "枝枯病是真菌性病害，主要危害枝条，导致枝条从顶端向下干枯死亡。"
                       "病部皮层腐烂、木质部变褐，严重影响树势和产量。",
        "advice": [
            "剪除病枝：发现病枝及时剪除并烧毁",
            "涂抹药剂：伤口处涂抹 50% 多菌灵可湿性粉剂",
            "增强树势：合理施肥，提高树体抗病能力",
            "定期检查：早发现早处理，防止扩散",
        ],
    },
    "Gall Midge": {
        "description": "瘿蚊幼虫寄生在芒果嫩叶和花序上，刺激组织异常增生，造成组织肿胀变形，"
                       "形成虫瘿，严重影响生长发育。",
        "advice": [
            "摘除虫瘿：发现虫瘿及时摘除并销毁",
            "药剂防治：新梢期喷施 20% 吡虫啉 2000 倍液",
            "关键时期：抽梢期是防治的关键时期",
            "生物防治：保护天敌，如寄生蜂等",
        ],
    },
    "Healthy": {
        "description": "叶片健康，生长状况良好，叶色浓绿，无病虫害症状，继续保持良好的管理措施。",
        "advice": [
            "定期巡查：继续观察植株健康状况",
            "科学施肥：保持均衡营养供应",
            "合理灌溉：根据天气和土壤湿度调节",
            "预防为主：做好日常病虫害预防工作",
        ],
    },
    "Powdery Mildew": {
        "description": "白粉病是真菌性病害，叶片、花序和幼果表面出现白色粉状物，影响光合作用和果实发育，"
                       "病部组织逐渐变黄枯萎。",
        "advice": [
            "改善环境：注意通风透光，降低湿度",
            "药剂防治：喷施 15% 三唑酮可湿性粉剂 1500 倍液",
            "防治时期：花期前后是防治关键期",
            "交替用药：避免产生抗药性",
        ],
    },
    "Sooty Mould": {
        "description": "煤污病又称煤烟病，是由蚜虫、介壳虫等害虫分泌的蜜露引起的次生性病害，"
                       "叶片表面覆盖黑色霉层，影响光合作用。",
        "advice": [
            "防治害虫：首先要防治蚜虫、介壳虫等",
            "清洗叶片：喷清水或碱性溶液清洗",
            "修剪枝条：改善通风透光条件",
            "综合防治：结合药剂防治害虫和病害",
        ],
    },
}


def app_mango_leaf_blueprint():
    """创建芒果叶片病害识别蓝图。"""
    blueprint = Blueprint("mango_leaf", __name__)

    # ---------------------------------------------------------------- 预测页面
    @blueprint.route("/test_predict_mango_leaf")
    def test_predict():
        # 未登录直接回登录页，避免页面半渲染
        if "login_name" not in session:
            return redirect("/")
        return render_template(
            "test_predict_mango_leaf.html",
            model_ready=True,
            model_name=TASKS["mango_leaf"]["architecture"],
        )

    # ---------------------------------------------------------------- 预测接口
    @blueprint.route("/submit_and_predict_mango_leaf", methods=["POST"])
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
        image_path, upload_url, filename = save_uploaded_file(test_file, "mango_leaf")

        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
            result = run_classification("mango_leaf", image)
        # 权重没准备好：返回 503 和排查提示，不用随机结果糊弄用户
        except ModelNotReady as exc:
            return jsonify({
                "error": str(exc),
                "hint": "请确认 models 目录下的权重文件存在，并检查服务器启动日志。",
                "model": model_meta("mango_leaf"),
            }), 503
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"图像处理失败: {exc}"}), 500

        info = DISEASE_INFO.get(result["predict_en"], {
            "description": "暂无详细信息",
            "advice": ["建议咨询专业农技人员"],
        })

        # 不是芒果叶片样本：不给预测，也不写进识别记录，避免污染统计
        if not result["valid"]:
            print(
                f"[Mango Leaf] 拒绝非本模型样本 · "
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
        except Exception as exc:  # noqa: BLE001 - 记录失败不影响识别结果
            print(f"[Mango Leaf] 保存记录失败: {exc}")

        response = {
            **success_fields(result["ood"]),
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
        print(f"[Mango Leaf] {result['predict_en']} · {response['confidence']:.2f}% · {result['elapsed_ms']:.0f} ms")
        return jsonify(response)

    return blueprint
