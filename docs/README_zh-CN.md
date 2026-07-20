# Acoustic Agent 中文说明

Acoustic Agent 是一个用于室内声场仿真和 RIR 生成的 Python 引擎，同时提供
Geometry 与 Floorplan 共用的本地 WebGL 工作台。项目支持几何房间、户型图场景、
本地生成的定制户型、语义材料、家具、指向性声源、Mono/阵列/HRTF 接收器，
以及静态和动态轨迹。

## 完整资源

源码和 Python 安装包必须同时包含以下四个运行资源，不能为了缩小发行包而省略：

- `cipic_124.sofa`：默认 CIPIC subject 124 HRTF。
- `sadie_h12.sofa`：SADIE II H12 HRTF。
- `acoustic_materials_v3.sqlite3`：3741 条六频带材料记录。
- `floorplan_v1.sqlite3`：15376 个经过筛选的户型图场景。

两个 SQLite 数据库由 Git LFS 管理，两个 SOFA 文件直接保存在 Git 中。安装前需要
执行 `git lfs pull`，安装后建议执行：

```bash
acoustic-agent verify-resources --hashes
```

## 安装

```bash
git lfs install
git clone <repository-url>
cd acoustic-agent
git lfs pull

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Windows PowerShell 使用 `.venv\Scripts\Activate.ps1` 激活环境。

## 启动 Web

```bash
acoustic-agent web
```

- Geometry：<http://127.0.0.1:8765/geometry>
- Floorplan：<http://127.0.0.1:8765/floorplan>
- Custom：<http://127.0.0.1:8765/custom>

修改端口或资源路径：

```bash
acoustic-agent web --port 9000
acoustic-agent web --floorplan-resource /path/to/floorplan_v1.sqlite3
```

## Geometry 示例

```python
from acoustic_agent import AcousticAgent

room = {
    "shape": "rectangle",
    "size": [6.0, 4.0, 2.8],
    "material_profile": {
        "wall": "auto",
        "floor": "auto",
        "ceiling": "auto",
    },
    "material_seed": 42,
}

agent = AcousticAgent(
    room=room,
    receiver_model={"type": "mono"},
    source_model={"type": "omni"},
    quality="simulation",
    duration_s=2.0,
    fs=16000,
)

result = agent.run(
    source=[1.2, 1.1, 1.5],
    receiver=[4.7, 2.8, 1.4],
)
rir = result.rir
```

## Floorplan 示例

Floorplan 场景由 Mohamed Abouagour 和 Eleftherios Garyfallidis 发布的 ResPlan
户型图数据集转换而来。论文为《ResPlan: A Large-Scale Vector-Graph Dataset of
17,000 Residential Floor Plans》，arXiv:2508.14006（2025）；数据地址为
https://www.kaggle.com/datasets/resplan/resplan。

```python
from acoustic_agent import AcousticAgent

agent = AcousticAgent.from_floorplan(
    idx=0,
    placement="same_room",  # random / same_room / cross_room
    seed=42,
    material_seed=2026,
    receiver_model={"type": "mono"},
    source_model={"type": "omni"},
    quality="simulation",
    duration_s=2.0,
    fs=16000,
)

print(agent.rooms)
print(agent.placement)
rir = agent.run().rir
```

同房间和跨房间使用同一个完整户型模型。确认的室内门按敞开 portal 处理，墙体连接
处没有门墙的区域按全高连通区域处理，未匹配的入户门保持关闭，窗户保持玻璃表面。

## 定制户型示例

定制户型不需要 GPT 或 VLM API。内置的本地生成器可以解析简短的中英文房间描述，
生成可编辑、可校验的米制户型 JSON，再复用 Floorplan 的门洞、材料和 RIR 求解流程。

```python
from acoustic_agent import AcousticAgent, FloorplanBuilder

spec = FloorplanBuilder.from_text(
    "12m x 9m，三室两厅一厨两卫，一个储物间",
    seed=42,
)
agent = AcousticAgent.from_floorplan_spec(
    spec,
    source_room="living_0",
    receiver_room="bedroom_2",
    quality="preview",
    duration_s=2.0,
    fs=16000,
)
rir = agent.run().rir
```

上传图片只会在浏览器本地显示，不会上传到服务器。点击 “Copy Codex prompt”，把
提示词和图片交给 Codex，再将它输出的 JSON 粘贴回页面并点击 “Apply floor plan”。
Width 与 Depth 始终等比例校准，Height 独立设置。完整格式与测试方式见
[`CUSTOM_FLOORPLAN.md`](CUSTOM_FLOORPLAN.md)。

## 档位

| 档位 | Rays | Bounces |
| --- | ---: | ---: |
| `preview` | 8192 | 32 |
| `simulation` | 32768 | 64 |
| `fine` | 65536 | 96 |
| `reference` | 131072 | 96 |

跨房间求解可以根据场景连通关系自适应提高 bounce，默认最低为 96，最高为 128。
`duration_s` 只决定输出 RIR 的长度，不会替代档位中的射线数量和反射深度。

## 开发与测试

```bash
pytest
python -m build
python -m twine check dist/*
```

详细安装、配置和资源说明见：

- [`INSTALLATION.md`](INSTALLATION.md)
- [`CONFIGURATION.md`](CONFIGURATION.md)
- [`FLOORPLAN.md`](FLOORPLAN.md)
- [`CUSTOM_FLOORPLAN.md`](CUSTOM_FLOORPLAN.md)
- [`RESOURCES.md`](RESOURCES.md)

项目代码和项目原创文档采用 Apache-2.0。SOFA 和数据库资源保留各自授权条件。
Floorplan V1 是 ResPlan 的转换资源，遵循 CC BY-NC-SA 4.0，要求署名、非商业使用、
标注修改并以相同许可共享衍生数据。材料数据库采用混合来源条款。具体见根目录
`THIRD_PARTY_NOTICES.md` 及各资源目录中的 `DATA_LICENSE.md`。
