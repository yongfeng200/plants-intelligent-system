# 植物智能识别系统

基于 Flask + PyTorch 的植物病害识别与实例分割平台。**所有预测结果都来自真实训练权重**，模型缺失时会返回明确的 503 错误，不再使用随机模拟结果。

## 功能与模型

| 功能 | 页面路由 | 权重文件 | 网络结构 | 类别 | 测试准确率 |
| --- | --- | --- | --- | --- | --- |
| 葡萄病害识别 | `/grape/test_predict_grape` | `models/grape_disease_model.pth` | ResNet50 + CBAM | 4 | 99.89% |
| 葡萄果实分割识别 | `/grape_segment/test_predict_grape_segment` | `models/grape_segment_model.pt` | YOLOv8-seg | 3（品种） | — |
| 芒果品种识别 | `/mango_variety/test_predict_mango_variety` | `models/mango_variety_best_checkpoint.pth`（回退 `_model.pth`） | EfficientNet-B4 | 8 | 100% |
| 芒果叶片病害识别 | `/mango_leaf/test_predict_mango_leaf` | `models/mango_mdcn_best_checkpoint.pth`（回退 `_model.pth`） | MDCN（DenseNet-121） | 8 | 99.75% |
| 叶片病灶分割 + 物种/疾病识别 | `/leaf_disease/test_predict` | `models/leaf_disease_segment_model.pth`、`models/leaf_disease_classify_model.pth` | UNet++（EfficientNet-B0）+ DINOv2 ViT-S/14 双头 | 5 物种 / 10 病害 | 联合 71.3% |

> 「叶片病灶分割 + 物种/疾病识别」先用 UNet++ 预测病灶 mask 并统计病斑个数与面积占比，再用 DINOv2 微调模型在整叶原图上识别物种与病害；联合概率只在合法的「物种×病害」组合上归一化。

### 与训练脚本严格对齐

- 芒果叶片：`Resize(224, 224)` + `Normalize([0.6690, 0.6907, 0.6845], [0.2018, 0.1920, 0.2639])`
- 芒果品种 / 葡萄病害：`Resize(224, 224)` + ImageNet 归一化
- 输出统一为 8 / 8 / 4 类顺序，与 `train/runs/*_metrics.json` 中的 `class_to_idx` 一致
- 网络结构、权重 key、预处理全部在 `py/model_registry.py` 中集中定义

## 非本模型样本拒识

分类网络是闭集 softmax：**只要喂一张图，它一定会给出一个类别**。用户把芒果叶片传到葡萄病害页面、传截图或别的作物，模型照样会输出「黑腐病 99.8%」。所以每个预测接口都加了一层「这张图到底是不是本模型的样本」的判定，判定不通过就**直接不给结论**。

判定分三层（`py/ood_detection.py`），**以特征层的马氏距离为主判据**：

| 层 | 看什么 | 代价 | 拦什么 |
| --- | --- | --- | --- |
| 图像层 | 分辨率、对比度、清晰度 | 微秒级 | 纯色图、缩略图、严重模糊 |
| 模型层 | max softmax | 已有前向，免费 | 模型自己完全没把握的图（< 25% 直接拒绝） |
| 特征层 | 倒数第二层特征 → 标准化 + PCA → **到最近类原型的马氏距离** | 一次矩阵乘 | 模型很「自信」但特征根本不像训练数据 |

> 一开始把三个指标加权合成一个分数，实测发现反而稀释了信号：在芒果叶片模型上，合成分数在误拒 1.2% 时只能拦下 33.8% 的葡萄叶片；改成单用马氏距离阈值，误拒 0.8% 时能拦 94.6%。所以特征距离当主判据，置信度只作为补充。

阈值按数据说话，取「拟合集 95 分位」和「留出集 98 分位」中较大的那个：只看拟合集会被少数离群图片把阈值撑大（芒果叶片实测 fit p99 比 val p99 高 1.9），只看留出集又会在两个切分难度差异大时把阈值压得过低。判定结果分三档：

- `ok`：正常返回预测
- `suspicious`：照常返回预测，但响应里多一个 `warning`，页面顶部挂一条可关闭的提醒
- `rejected`：**不返回 `predict` / `confidence`**，只返回原因和「建议上传」的提示，也不写进识别记录（不污染统计）

界面上只出现「原因 + 建议上传」，马氏距离、能量分数这些内部数值只放在响应的 `ood.scores` 里供开发排查，不在页面上展示。

### 校准（首次部署或换了数据集后跑一次）

```powershell
# 用仓库自带的测试集拟合阈值，写进 models/ood_stats/*.npz
python py/build_ood_stats.py

# 只处理某个模型 / 每类只取 N 张 / 跑得快点
python py/build_ood_stats.py --task grape_disease --limit 40

# 只验证，不重新校准
python py/build_ood_stats.py --verify-only

# 交叉验证：把 A 模型的数据集喂给 B 模型，看能不能拦住误传
python py/build_ood_stats.py --cross
```

每份校准档案只有 100 KB 左右，需要跟着部署走。没有校准文件时服务不会崩，会退化成只看置信度的保守模式，并在启动日志里提示。

### 实测效果（本仓库测试集）

误拒率 = 本模型自己的样本被判为「拒绝」的比例，越低越好：

| 模型 | 马氏距离拒绝阈值 | 误拒率（留出集） | 疑似率 |
| --- | --- | --- | --- |
| 葡萄病害 | 23.31 | 2.08% | 0.00% |
| 叶片分割+识别 | 11.28 | 2.30% | 1.15% |
| 芒果叶片 | 8.89 | 1.04% | 1.88% |
| 芒果品种 | 11.82 | 2.19% | 1.25% |

交叉验证（行 = 被测试的模型，列 = 图片来源，数值 = 被拒绝或被标疑似的比例）：

| 模型 \ 图片 | 葡萄病害 | 叶片识别 | 芒果叶片 | 芒果品种 |
| --- | --- | --- | --- | --- |
| 葡萄病害 | **2%** | 25% | 82% | 12% |
| 叶片识别 | 46% | **1%** | 34% | 99% |
| 芒果叶片 | **100%** | **100%** | 8% | 97% |
| 芒果品种 | 91% | 95% | 81% | **0%** |

最典型的一条：**葡萄黑腐病叶片传到芒果叶片模型，100% 被拒绝**（改造前是 62%，芒果模型会以 94% 的置信度给出「炭疽病」）。

**已知局限**（如实记录，不是待办）：

- 葡萄病害模型对芒果**果实**图的拦截只有 12%。它的留出集里有几张极端离群图（p99 达 33），把阈值抬到了 23.31，判定因此偏松；试过提高 PCA 维度（32→96）和用叶绿素像素占比做先验都不行——前者只会同时抬高误拒率，后者被数据否掉（芒果叶片图本身中位绿色占比只有 14.7%，果实图反而 48%）。
- `叶片识别` 对葡萄病害图只拦 46%，这是符合预期的：葡萄本来就是它 5 个物种之一（`Grape_Black_rot` 类）。
- 拒绝阈值取到拟合集 95 分位，意味着芒果叶片模型在**它自己的测试集**上会拒掉约 5%（留出集只有 1.04%）：宁可让用户重拍一张，也不要给出错误诊断。
- 想调松紧不用重新校准：设置环境变量 `PLANT_OOD_STRICTNESS`，`1.0` 是默认，`1.3` 更宽松（更少拒绝），`0.7` 更严格。
- 阈值是按「测试集/留出集」标定的。如果线上图片的拍摄设备、光照、背景和测试集差得远，误拒会变多，建议用真实场景的图片重新跑一次 `build_ood_stats.py --limit 200`。


## 快速开始

```powershell
# 1. 安装依赖（建议使用已装好 torch 的 conda 环境）
pip install -r requirements.txt

# 2. 检查模型是否可加载（可选）
python py/check_models.py

# 3. 启动服务
python py/main.py
```

浏览器访问 <http://127.0.0.1:5000>。

测试账户（首次启动自动创建）：

- `admin` / `admin123`
- `test` / `test123`

> 不要使用已删除的旧入口 `python app.py`。当前唯一入口是 `py/main.py`。

## 页面

| 路由 | 说明 | 需要登录 |
| --- | --- | --- |
| `/` | 登录 | 否 |
| `/register` | 注册 | 否 |
| `/dashboard` | 仪表板（5 个 AI 能力入口） | 是 |
| `/upload` | 我的识别记录（分页 / 筛选 / 删除） | 是 |
| `/statistics` | 统计分析（趋势、模型分布、时段、TOP10） | 是 |
| `/model_status` | 模型加载状态与准确率 | 是 |
| `/profile`、`/settings` | 个人资料 / 设置 | 是 |
| `/mango_leaf/...`、`/mango_variety/...`、`/grape/...`、`/grape_segment/...` | 5 个预测页面 | 是 |

页面统一使用 `static/css/plant-ui.css` 与 `static/js/plant-ui.js`：拖拽上传、按钮加载态、Toast 提示、概率图表动画、模型信息徽章，并尊重系统「减少动效」设置。

## API

### 预测接口

| 方法 | 地址 | 说明 |
| --- | --- | --- |
| POST | `/mango_leaf/submit_and_predict_mango_leaf` | 芒果叶片病害 |
| POST | `/mango_variety/submit_and_predict_mango_variety` | 芒果品种 |
| POST | `/grape/submit_and_predict_grape` | 葡萄病害 |
| POST | `/grape_segment/submit_and_segment` | 葡萄实例分割 |

请求：`multipart/form-data`，字段名 `file`，支持 PNG/JPG/JPEG，最大 20MB。

响应示例（分类）：

```json
{
  "valid": true,
  "ood_status": "ok",
  "upload_image": "/static/img/mango_leaf_test/xxxx.jpg",
  "predict": "切叶象甲",
  "predict_cn": "切叶象甲",
  "predict_en": "Cutting Weevil",
  "confidence": 99.98,
  "pred_pros": [0.0001, 0.0001, 0.9998, 0.0, 0.0, 0.0, 0.0, 0.0],
  "disease_labels": ["炭疽病", "细菌性溃疡病", "切叶象甲", "枝枯病", "瘿蚊", "健康", "白粉病", "煤污病"],
  "disease_labels_en": ["Anthracnose", "Bacterial Canker", "Cutting Weevil", "Die Back", "Gall Midge", "Healthy", "Powdery Mildew", "Sooty Mould"],
  "description": "切叶象甲是一种害虫……",
  "advice": ["人工捕杀：……"],
  "model": {
    "key": "mango_leaf",
    "name": "芒果叶片病害识别",
    "architecture": "MDCN · DenseNet-121",
    "accuracy_text": "99.75%",
    "device": "CUDA · NVIDIA GeForce RTX 4060 Laptop GPU",
    "loaded": true,
    "checkpoint": "mango_mdcn_best_checkpoint.pth",
    "elapsed_ms": 37.2
  }
}
```

模型未就绪时返回 `503`：

```json
{ "error": "未找到模型文件，已尝试: ...", "hint": "请确认 ...", "model": { "loaded": false } }
```

不是本模型的样本时返回 `200` + `valid=false`（**不含 `predict` / `confidence` / `pred_pros`**）：

```json
{
  "valid": false,
  "status": "rejected",
  "upload_image": "/static/img/grape_test/xxxx.jpg",
  "message": "这张图片不是可识别的葡萄叶片样本，系统没有输出预测结果。",
  "reasons": ["图片的整体特征与模型学习过的样本差别较大"],
  "hints": [
    "上传单张葡萄叶片的清晰特写，叶片占画面主体",
    "不要上传葡萄果实、芒果/番茄等其他作物、截图",
    "本模型只能识别：4 类葡萄叶片状态：黑腐病、黑麻疹（ESCA）、健康、叶斑病"
  ],
  "checks": [],
  "ood": { "score": 12.4, "scores": { "mahalanobis": 1984.9, "max_prob": 0.98 }, "calibrated": true },
  "model": { "key": "grape_disease" }
}
```

`reasons` 是给用户看的，只讲人话；内部指标（马氏距离、能量、阈值）只出现在 `ood` 里，页面不渲染。

判定为「疑似」时仍然返回预测，但多一个 `warning` 字段（页面顶部会自动挂一条可关闭的提醒）。前端 `plant-ui.js` 里的 `bindUpload` 会统一处理这两种情况，页面脚本不需要额外判断。

### 状态 / 记录接口

| 方法 | 地址 | 说明 |
| --- | --- | --- |
| GET | `/api/model_status` | 模型状态、设备、准确率 |
| GET | `/api/model_info` | 兼容旧接口 |
| GET | `/health` | 健康检查（模型数量 / 设备） |
| GET | `/api/user_uploads?page=1&per_page=9&model_type=&disease_class=` | 分页记录 |
| DELETE | `/api/delete_record/<id>` | 删除单条记录 |
| DELETE | `/api/clear_all_records` | 清空当前用户记录 |
| GET | `/api/statistics` | 趋势 / 模型分布 / 时段 / 排行 / 平均置信度 |
| POST | `/api/update_profile` | 修改邮箱 / 密码 |
| DELETE | `/api/delete_account` | 注销账户 |

## 目录结构

```
plants-intelligent-system/
├── py/
│   ├── main.py                 # 唯一入口
│   ├── model_registry.py       # 模型结构 / 权重 / 预处理 / 注册表
│   ├── ood_detection.py        # 非本模型样本判定（图像层 + 置信度 + 马氏距离）
│   ├── build_ood_stats.py      # 校准 / 验证脚本
│   ├── app_mango_leaf.py       # 芒果叶片病害
│   ├── app_mango_variety.py    # 芒果品种
│   ├── app_grape.py            # 葡萄病害
│   ├── app_grape_segment.py    # 葡萄实例分割
│   ├── user.py                 # 认证与记录
│   └── check_models.py         # 模型自检
├── tests/                      # 拒识逻辑单元测试 + 接口端到端测试
├── templates/                  # 前端页面（无冗余页面）
├── static/
│   ├── css/plant-ui.css        # 统一动效与交互
│   ├── js/plant-ui.js
│   └── img/                    # 上传与测试图片
├── models/                     # 训练权重
│   └── ood_stats/              # 拒识校准档案（由 build_ood_stats.py 生成）
├── database/plant_system.db    # SQLite
└── train/                      # 训练 Notebook 与指标
```

## 配置

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PLANT_MODELS_DIR` | `<项目根>/models` | 模型目录，便于部署或测试时复用外部权重 |
| `PLANT_SECRET_KEY` | 随机生成 | Flask session 密钥，生产环境请固定 |
| `PLANT_CONSOLE_ENCODING` | 自动识别 | 控制台输出编码；PyCharm 默认自动用 cp936，若把 Run 控制台改成 UTF-8 可设为 `utf-8` |
| `PLANT_OOD_DIR` | `<模型目录>/ood_stats` | 拒识校准档案目录 |
| `PLANT_OOD_STRICTNESS` | `1.0` | 拒识松紧。`>1` 更宽松（更少拒绝），`<1` 更严格，改它不用重新校准 |

其他：上传目录为 `static/img/<任务>_test`，最大请求 20MB，SQLite 数据库 `database/plant_system.db`。

## 常见问题

**Q：PyCharm / 控制台中文乱码？**
`py/console_encoding.py` 会区分运行环境：

- PowerShell / Windows Terminal 等真实控制台：自动切到 UTF-8（CP65001）
- PyCharm Run 控制台（管道）：自动使用系统 ANSI 编码（简体中文为 cp936），不再强制 UTF-8
- `ultralytics` 在导入时会强制把 stdout 改成 UTF-8，已在模型加载后自动恢复项目配置的编码

启动日志中会打印当前编码，例如 `控制台编码: cp936`。如果仍乱码：

1. 确认日志里的“控制台编码”与 PyCharm 设置 → 编辑器 → 文件编码 → **控制台输出编码** 一致
2. 或设置环境变量 `PLANT_CONSOLE_ENCODING=utf-8` 后重启服务
3. 不要同时设置 `PYTHONIOENCODING` 为不匹配的值

**Q：用户上传了不属于这个模型的图片怎么办？**
已经内置拒识，不会给出瞎猜的类别，详见「非本模型样本拒识」一节。如果发现误拒偏多，先把 `PLANT_OOD_STRICTNESS` 调到 `1.3` 试试；仍然不行就用真实场景的图片重新跑 `python py/build_ood_stats.py --limit 200`。

**Q：预测页面提示“模型文件未找到”？**
运行 `python py/check_models.py` 查看具体是哪个权重缺失；确认文件名与上表一致。接口会返回 503，不会显示随机结果。

**Q：第一次预测很慢？**
服务启动时会对分类模型做一次预热（`model_registry.py`），之后单张图片通常几十毫秒。YOLOv8-seg 首次调用会稍慢，之后正常。

**Q：端口被占用？**
修改 `py/main.py` 末尾 `app.run(..., port=5000)`。

## 权重与测试集

仓库包含**推理所需的 6 个权重（约 349 MB）**和**5 个测试集（1686 个文件 / 359.5 MB）**，克隆后装好依赖即可直接启动并复现下表的准确率；训练集与训练产物不在仓库内。

### 推理权重

| 文件 | 大小 |
| --- | --- |
| `models/mango_mdcn_best_checkpoint.pth` | 27.2 MB |
| `models/mango_variety_model.pth` | 67.7 MB |
| `models/grape_disease_model.pth` | 92.0 MB |
| `models/grape_segment_model.pt` | 52.3 MB |
| `models/leaf_disease_segment_model.pth` | 25.4 MB |
| `models/leaf_disease_classify_model.pth` | 84.6 MB |
### 测试集

| 路径 | 文件数 | 大小 |
| --- | --- | --- |
| `Grape Detection Segment/yolo_seg_dataset/test` | 152（76 图 + 76 标注） | 267.4 MB |
| `Grape Disease Dataset/test` | 887 | 56.6 MB |
| `Leaf Disease Segment/classification/test` | 87 | 8.3 MB |
| `Mango Dataset/test` | 160 | 13.9 MB |
| `Mango Leaf Dataset/test` | 400 | 13.4 MB |

### 未入库内容

| 内容 | 大小 | 原因 |
| --- | --- | --- |
| 5 个数据集的训练集与验证集 | ~5.1 GB | 体积过大 |
| `models/*_best_checkpoint.pth`（葡萄病害、芒果品种） | 289 / 212 MB | 单文件超 100 MiB；含优化器状态，与已入库的 `_model.pth` 等价 |
| `train/sam_vit_b_01ec64.pth` | 375 MB | 单文件超 100 MiB；仅训练用，SAM 官方可下载 |
| `train/runs/**/weights/*.pt`、`best_model.pth` | ~230 MB | 训练检查点 |
| `train/yolov8m*.pt`、`yolo11n.pt` | ~110 MB | 预训练主干，ultralytics 自动下载 |
| `uploads/`、`database/*.db` | — | 运行时生成 |

> `model_registry.py` 按顺序回退：葡萄病害优先找 `grape_disease_model.pth`（已入库），芒果品种优先找 `mango_variety_best_checkpoint.pth`、缺失时自动回退 `mango_variety_model.pth`（已入库），因此仓库里的权重足够跑通全部 5 个页面。

需要完整复现训练时：

1. 把训练集与验证集放回各数据集目录，目录名保持原样，Notebook 中的相对路径即可继续使用。
2. 缺失的检查点放到 `models/` 或用 `PLANT_MODELS_DIR` 指向外部目录。
3. 运行 `python py/check_models.py` 自检，缺失的项会逐个列出。
4. `database/` 与 `uploads/` 目录会在首次启动时自动创建。

## 说明

本项目仅用于学习与研究。训练数据、Notebook 和权重文件来自项目自身的 `train/` 与数据集目录。
