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
│   ├── app_mango_leaf.py       # 芒果叶片病害
│   ├── app_mango_variety.py    # 芒果品种
│   ├── app_grape.py            # 葡萄病害
│   ├── app_grape_segment.py    # 葡萄实例分割
│   ├── user.py                 # 认证与记录
│   └── check_models.py         # 模型自检
├── templates/                  # 前端页面（无冗余页面）
├── static/
│   ├── css/plant-ui.css        # 统一动效与交互
│   ├── js/plant-ui.js
│   └── img/                    # 上传与测试图片
├── models/                     # 训练权重
├── database/plant_system.db    # SQLite
└── train/                      # 训练 Notebook 与指标
```

## 配置

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PLANT_MODELS_DIR` | `<项目根>/models` | 模型目录，便于部署或测试时复用外部权重 |
| `PLANT_SECRET_KEY` | 随机生成 | Flask session 密钥，生产环境请固定 |
| `PLANT_CONSOLE_ENCODING` | 自动识别 | 控制台输出编码；PyCharm 默认自动用 cp936，若把 Run 控制台改成 UTF-8 可设为 `utf-8` |

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

**Q：预测页面提示“模型文件未找到”？**
运行 `python py/check_models.py` 查看具体是哪个权重缺失；确认文件名与上表一致。接口会返回 503，不会显示随机结果。

**Q：第一次预测很慢？**
服务启动时会对分类模型做一次预热（`model_registry.py`），之后单张图片通常几十毫秒。YOLOv8-seg 首次调用会稍慢，之后正常。

**Q：想换模型目录？**
设置 `PLANT_MODELS_DIR=D:\path\to\models` 后启动即可，无需改代码。

**Q：端口被占用？**
修改 `py/main.py` 末尾 `app.run(..., port=5000)`。

**Q：为什么删掉了 `camera_grape_segment.html`、`py/app.py`、`app_mango_leaf_segment.py`？**
它们对应的后端模型/接口并不存在或已废弃（摄像头接口、旧版单文件入口、GradCAM 诊断页的模板缺失导致访问必然 500），保留只会误导使用。叶片诊断请用 `/leaf_disease/test_predict`：UNet++ 分割病灶 + DINOv2 识别物种与病害。

## 权重与数据集

仓库只包含源码、页面、Notebook 与训练指标，**不包含**数据集和模型权重（合计约 6.5 GB，远超 GitHub 限制）：

| 内容 | 大小 | 说明 |
| --- | --- | --- |
| `models/*.pth`、`models/*.pt` | ~1.0 GB | 5 个功能的推理权重 |
| 5 个数据集目录（`Grape Detection Segment` 等） | ~5.5 GB | 训练用原始数据 |
| `train/runs/**` 里的图片与 `.pt` | ~230 MB | 训练曲线与检查点 |
| `train/sam_vit_b_01ec64.pth`、`train/yolov8m*.pt` 等 | ~460 MB | 预训练主干，ultralytics / SAM 会自动下载 |

克隆后按需补齐：

1. 把权重放回 `models/`（文件名见上表），或用环境变量 `PLANT_MODELS_DIR` 指向外部目录。
2. 数据集放回项目根目录，目录名保持原样，Notebook 中的相对路径即可继续使用。
3. 运行 `python py/check_models.py` 自检，缺失的项会逐个列出。
4. `database/` 与 `uploads/` 目录会在首次启动时自动创建。

## 说明

本项目仅用于学习与研究。训练数据、Notebook 和权重文件来自项目自身的 `train/` 与数据集目录。
