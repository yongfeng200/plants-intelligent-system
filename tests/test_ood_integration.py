# -*- coding: utf-8 -*-
"""端到端验收：通过 Flask 接口验证「非本模型样本」拦截。

需要 torch 环境（会真的加载权重、真的推理），用项目环境运行：
    D:\\WorkSpace\\MiniConda\\envs\\myenv\\python.exe tests\\test_ood_integration.py

检查点：
1. 本模型样本 → valid=True，正常返回预测；
2. 别家模型的样本 → valid=False，且**不返回** predict / confidence；
3. 坏图被前端/后端拦住；
4. 被拒的图片不写进识别记录，不污染统计。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))

from main import app  # noqa: E402

MANGO_LEAF_DIR = ROOT / "Mango Leaf Dataset" / "test"
GRAPE_LEAF_DIR = ROOT / "Grape Disease Dataset" / "test"
LEAF_DISEASE_DIR = ROOT / "Leaf Disease Segment" / "classification" / "test"

SUFFIXES = {".jpg", ".jpeg", ".png"}
TEST_USER = "ood_e2e_user"
TEST_PASSWORD = "ood_e2e_pass"


def first_image(root: Path, class_hint: str | None = None) -> Path:
    classes = sorted(p for p in root.iterdir() if p.is_dir())
    if class_hint:
        classes = [p for p in classes if class_hint.lower() in p.name.lower()]
    for class_dir in classes:
        for file in class_dir.rglob("*"):
            if file.is_file() and file.suffix.lower() in SUFFIXES:
                return file
    raise FileNotFoundError(f"{root} 下找不到测试图片")


def _client():
    app.config["TESTING"] = True
    client = app.test_client()
    client.post("/register", json={
        "username": TEST_USER, "password": TEST_PASSWORD, "email": "",
    })
    response = client.post("/login", json={"username": TEST_USER, "password": TEST_PASSWORD})
    assert response.status_code == 200, f"登录失败: {response.status_code} {response.get_data(as_text=True)[:200]}"
    return client


def _upload(client, url: str, path: Path, filename: str | None = None):
    with open(path, "rb") as handle:
        data = {"file": (handle, filename or path.name)}
        response = client.post(url, data=data, content_type="multipart/form-data")
    payload = response.get_json()
    assert response.status_code == 200, f"{url} 返回 {response.status_code}: {str(payload)[:300]}"
    return payload


def test_mango_leaf_accepts_own_sample():
    client = _client()
    payload = _upload(client, "/mango_leaf/submit_and_predict_mango_leaf", first_image(MANGO_LEAF_DIR))
    assert payload["valid"] is True, payload.get("reasons")
    assert payload["predict"]


def test_mango_leaf_rejects_grape_leaf():
    """用户实际遇到的问题：葡萄黑腐病叶片传到芒果叶片模型，必须拒绝。"""
    client = _client()
    payload = _upload(
        client, "/mango_leaf/submit_and_predict_mango_leaf",
        first_image(GRAPE_LEAF_DIR, "Black Rot"),
    )
    assert payload["valid"] is False, payload
    assert "predict" not in payload
    assert payload["reasons"]


def test_grape_model_rejects_mango_leaf():
    client = _client()
    payload = _upload(client, "/grape/submit_and_predict_grape", first_image(MANGO_LEAF_DIR))
    assert payload["valid"] is False, payload
    for key in ("predict", "confidence", "pred_pros"):
        assert key not in payload, f"拒绝时不应返回 {key}"
    assert payload["message"]
    assert payload["reasons"]
    assert payload["upload_image"].lower().endswith(tuple(SUFFIXES))


def test_grape_model_accepts_own_sample():
    client = _client()
    payload = _upload(client, "/grape/submit_and_predict_grape", first_image(GRAPE_LEAF_DIR))
    assert payload["valid"] is True, payload.get("reasons")
    assert payload["predict"]


def test_frontend_reasons_contain_no_internal_metrics():
    """前端会直接渲染 reasons，里面不能出现马氏距离之类的内部指标。"""
    client = _client()
    payload = _upload(client, "/grape/submit_and_predict_grape", first_image(MANGO_LEAF_DIR))
    text = " ".join(payload.get("reasons") or [])
    for banned in ("马氏", "能量", "z 分数", "logsumexp"):
        assert banned not in text, f"reasons 里不应出现「{banned}」：{text}"


def test_leaf_pipeline_accepts_own_sample():
    client = _client()
    payload = _upload(client, "/leaf_disease/submit_and_predict", first_image(LEAF_DISEASE_DIR))
    assert payload["valid"] is True, payload.get("reasons")
    assert payload["species_cn"]
    assert payload["mask_base64"].startswith("data:image/png;base64,")


def test_leaf_pipeline_rejects_out_of_domain_leaf():
    client = _client()
    payload = _upload(client, "/leaf_disease/submit_and_predict", first_image(MANGO_LEAF_DIR))
    if payload["valid"]:
        assert payload.get("warning"), "放行时必须给出 warning"
    else:
        assert "predict" not in payload and payload["reasons"]


def test_rejected_upload_is_not_recorded():
    client = _client()
    before = client.get("/api/user_uploads?per_page=1").get_json()["total"]
    payload = _upload(client, "/grape/submit_and_predict_grape", first_image(MANGO_LEAF_DIR))
    assert payload["valid"] is False
    after = client.get("/api/user_uploads?per_page=1").get_json()["total"]
    assert after == before, f"被拒图片不应写库：{before} -> {after}"


def main() -> int:
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failures = 0
    for name, func in tests:
        try:
            func()
            print(f"  [PASS] {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  [FAIL] {name}: {exc}")
            import traceback
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} 通过")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
