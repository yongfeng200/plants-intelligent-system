# -*- coding: utf-8 -*-
"""ood_detection 模块的单元测试（不需要 torch）。

运行：
    python tests/test_ood_detection.py
    # 或
    pytest tests/test_ood_detection.py -v
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "py"))

import ood_detection as ood  # noqa: E402

DIM = 12
K = 3
N_PER = 150


def _make_data(seed: int = 0, spread: float = 1.0):
    """造 K 个可分的高斯簇，模拟倒数第二层特征。"""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 6, size=(K, DIM))
    features, labels = [], []
    for c in range(K):
        features.append(centers[c] + rng.normal(0, spread, size=(N_PER, DIM)))
        labels += [f"class{c}"] * N_PER
    return np.vstack(features), labels, centers, rng


def _logits(features, centers, scale=0.05, bias=None):
    """logits = -scale * 到各中心的平方距离，越近越像。"""
    dist = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
    logits = -dist * scale
    if bias is not None:
        logits = logits + bias
    return logits


def _probs(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _profile(seed=0, spread=1.0, with_holdout=False):
    features, labels, centers, rng = _make_data(seed, spread)
    logits = _logits(features, centers)
    probs = _probs(logits)
    holdout = None
    if with_holdout:
        holdout = np.vstack([
            centers[c] + rng.normal(0, spread, size=(N_PER // 2, DIM)) for c in range(K)
        ])
    profile = ood.fit_profile(
        "mango_leaf", features, logits, probs, labels, pca_dim=6,
        holdout_features=holdout,
    )
    return profile, centers, rng


# ---------------------------------------------------------------------------
# 校准
# ---------------------------------------------------------------------------

def test_fit_profile_shapes_and_thresholds():
    profile, _, _ = _profile()
    assert profile.feature_dim == DIM
    assert profile.pca_dim == 6
    assert profile.class_means.shape == (K, 6)
    assert profile.precision.shape == (6, 6)
    # 主判据是马氏距离阈值，提示阈值必须比拒绝阈值更松
    assert np.isfinite(profile.maha_reject)
    assert profile.maha_warn < profile.maha_reject
    assert profile.reject_z > profile.warn_z
    assert profile.reject_z >= 2.0
    assert 2.5 <= profile.z_ind <= 6.0
    assert 0 < profile.ref_pmax <= 1


def test_holdout_raises_threshold_when_it_is_harder():
    """留出集更难时阈值必须被抬高，否则留出集会成批误拒。"""
    features, labels, centers, rng = _make_data(seed=5)
    logits = _logits(features, centers)
    probs = _probs(logits)
    without = ood.fit_profile("mango_leaf", features, logits, probs, labels, pca_dim=6)
    held = np.vstack([centers[c] + rng.normal(0, 3.0, size=(N_PER // 2, DIM)) for c in range(K)])
    with_hold = ood.fit_profile("mango_leaf", features, logits, probs, labels,
                                pca_dim=6, holdout_features=held)
    assert with_hold.maha_reject > without.maha_reject


def test_fit_rejects_too_few_samples():
    features = np.zeros((3, DIM))
    logits = np.zeros((3, K))
    probs = _probs(logits)
    try:
        ood.fit_profile("mango_leaf", features, logits, probs, ["a", "b", "c"])
    except ValueError as exc:
        assert "样本太少" in str(exc)
    else:
        raise AssertionError("样本太少时应当报错")


# ---------------------------------------------------------------------------
# 判定
# ---------------------------------------------------------------------------

def test_in_distribution_is_mostly_accepted():
    """阈值按「误拒率约 5%」标定，所以同分布样本应当绝大多数放行。"""
    profile, centers, rng = _profile(seed=0)
    guard = ood.OODGuard("mango_leaf", profile)

    rejected = 0
    total = 100
    for _ in range(total):
        sample = centers[0] + rng.normal(0, 1.0, size=(1, DIM))
        logits = _logits(sample, centers)
        result = guard.evaluate(sample[0], logits[0], _probs(logits)[0])
        if result["status"] == ood.STATUS_REJECTED:
            rejected += 1
    assert rejected <= total * 0.12, f"同分布样本误拒过多：{rejected}/{total}"


def test_far_sample_with_confident_logits_is_rejected():
    """关键用例：模型很“自信”，但特征根本不在训练分布里 —— 必须靠特征层拦下。"""
    profile, centers, rng = _profile(seed=0)
    guard = ood.OODGuard("mango_leaf", profile)

    sample = rng.normal(0, 1.0, size=(1, DIM)) + 200.0
    logits = np.array([[12.0, -3.0, -4.0]])   # 99.9% 的“自信”
    probs = _probs(logits)
    assert probs.max() > 0.99

    result = guard.evaluate(sample[0], logits[0], probs[0])
    assert result["status"] == ood.STATUS_REJECTED, result
    assert result["reasons"], "拒绝时必须给出可读的原因"


def test_reasons_do_not_leak_internal_numbers():
    """给用户看的原因里不能出现马氏距离、能量这些内部指标。"""
    profile, centers, rng = _profile(seed=0)
    guard = ood.OODGuard("mango_leaf", profile)
    sample = rng.normal(0, 1.0, size=(1, DIM)) + 200.0
    logits = np.array([[12.0, -3.0, -4.0]])
    result = guard.evaluate(sample[0], logits[0], _probs(logits)[0])
    text = " ".join(result["reasons"])
    for banned in ("马氏", "能量", "分数", "分布"):
        assert banned not in text, f"原因里不应出现「{banned}」：{text}"


def test_boundary_sample_is_flagged_or_rejected():
    profile, centers, rng = _profile(seed=1)
    guard = ood.OODGuard("mango_leaf", profile)

    sample = centers[0] + rng.normal(0, 4.0, size=(1, DIM))
    logits = _logits(sample, centers)
    result = guard.evaluate(sample[0], logits[0], _probs(logits)[0])
    assert result["status"] in {ood.STATUS_SUSPICIOUS, ood.STATUS_REJECTED}, result


def test_feature_dim_mismatch_falls_back_instead_of_crashing():
    profile, _, _ = _profile(seed=0)
    guard = ood.OODGuard("mango_leaf", profile)

    logits = np.array([[9.0, 0.0, 0.0]])
    probs = _probs(logits)
    result = guard.evaluate(np.zeros(5), logits[0], probs[0])  # 维度不对
    assert result["calibrated"] is False
    assert result["status"] == ood.STATUS_OK  # 高置信度时按正常放行


def test_guard_without_profile_uses_confidence_only():
    guard = ood.OODGuard("mango_leaf", profile=None)
    guard.profile = None

    logits = np.array([[12.0, 0.0, 0.0]])
    ok = guard.evaluate(np.zeros(DIM), logits[0], _probs(logits)[0])
    assert ok["status"] == ood.STATUS_OK
    assert ok["calibrated"] is False

    flat = np.array([[1.0, 1.0, 1.0]])
    bad = guard.evaluate(np.zeros(DIM), flat[0], _probs(flat)[0])
    assert bad["status"] == ood.STATUS_REJECTED


# ---------------------------------------------------------------------------
# 图像质量
# ---------------------------------------------------------------------------

def test_image_quality_tiny_image_rejected():
    tiny = np.zeros((20, 20, 3), dtype=np.uint8)
    issues = ood.check_image_quality(tiny)
    assert issues and issues[0]["level"] == "reject"
    assert issues[0]["code"] == "too_small"


def test_image_quality_flat_image_rejected():
    flat = np.full((128, 128, 3), 200, dtype=np.uint8)
    issues = ood.check_image_quality(flat)
    assert issues and issues[0]["level"] == "reject"
    assert issues[0]["code"] == "flat"


def test_image_quality_normal_image_passes():
    rng = np.random.default_rng(3)
    noisy = rng.integers(0, 255, size=(128, 128, 3), dtype=np.uint8)
    issues = ood.check_image_quality(noisy)
    assert not [item for item in issues if item["level"] == "reject"]


def test_guard_rejects_unusable_image_before_model_scores():
    profile, centers, _ = _profile(seed=0)
    guard = ood.OODGuard("mango_leaf", profile)
    sample = centers[0]
    logits = _logits(sample[None, :], centers)
    result = guard.evaluate(
        sample, logits[0], _probs(logits)[0],
        image_rgb=np.zeros((10, 10, 3), dtype=np.uint8),
    )
    assert result["status"] == ood.STATUS_REJECTED
    assert result["score"] is None


# ---------------------------------------------------------------------------
# 存取与响应结构
# ---------------------------------------------------------------------------

def test_save_and_load_roundtrip():
    profile, _, _ = _profile(seed=0)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mango_leaf.npz"
        ood.save_profile(profile, path)
        assert path.exists()
        loaded = ood.load_profile("mango_leaf", path)

    assert loaded is not None
    assert loaded.feature_dim == profile.feature_dim
    assert loaded.pca_dim == profile.pca_dim
    assert loaded.classes == profile.classes
    assert np.allclose(loaded.mean, profile.mean)
    assert np.allclose(loaded.precision, profile.precision)
    assert abs(loaded.maha_reject - profile.maha_reject) < 1e-6
    assert abs(loaded.maha_warn - profile.maha_warn) < 1e-6


def test_missing_profile_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        assert ood.load_profile("nope", Path(tmp) / "nope.npz") is None


def test_rejection_payload_has_no_prediction_fields():
    ood_result = {
        "status": ood.STATUS_REJECTED,
        "score": 3.2,
        "scores": {"max_prob": 0.2},
        "reasons": ["图片的整体特征与模型学习过的样本差别较大"],
        "message": "这张图片不是可识别的芒果叶片样本，系统没有输出预测结果。",
        "hints": ["上传单张芒果叶片的清晰特写"],
        "quality": [],
        "calibrated": True,
    }
    payload = ood.rejection_payload(ood_result, {"name": "x"}, {"upload_image": "/a.jpg"})
    assert payload["valid"] is False
    assert payload["upload_image"] == "/a.jpg"
    assert payload["model"] == {"name": "x"}
    for key in ("predict", "confidence", "pred_pros"):
        assert key not in payload, f"拒绝时不应返回 {key}"


def test_success_fields_marks_suspicious_with_warning():
    ok = ood.success_fields({"status": ood.STATUS_OK})
    assert ok == {"valid": True, "ood_status": ood.STATUS_OK}

    warn = ood.success_fields({
        "status": ood.STATUS_SUSPICIOUS,
        "message": "差异较大",
        "reasons": ["a"],
        "hints": ["b"],
    })
    assert warn["valid"] is True
    assert warn["warning"]["message"] == "差异较大"


# ---------------------------------------------------------------------------
# 运行器
# ---------------------------------------------------------------------------

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
