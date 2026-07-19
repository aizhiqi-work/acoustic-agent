# Acoustic Agent 中文说明

Acoustic Agent 是一个用于室内声场仿真和 RIR 生成的 Python 引擎，同时提供
Geometry 与 ResPlan 共用的本地 WebGL 工作台。项目支持几何房间、户型图场景、
语义材料、家具、指向性声源、Mono/阵列/HRTF 接收器，以及静态和动态轨迹。

## 完整资源

源码和 Python 安装包必须同时包含以下四个运行资源，不能为了缩小发行包而省略：

- `cipic_124.sofa`：默认 CIPIC subject 124 HRTF。
- `sadie_h12.sofa`：SADIE II H12 HRTF。
- `acoustic_materials_v3.sqlite3`：3741 条六频带材料记录。
- `resplan_v1.sqlite3`：15376 个经过筛选的户型图场景。

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
- ResPlan：<http://127.0.0.1:8765/resplan>

修改端口或资源路径：

```bash
acoustic-agent web --port 9000
acoustic-agent web --resplan-resource /path/to/resplan_v1.sqlite3
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

## ResPlan 示例

```python
from acoustic_agent import AcousticAgent

agent = AcousticAgent.from_resplan(
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
- [`RESOURCES.md`](RESOURCES.md)

项目代码和项目原创文档采用 Apache-2.0。SOFA 和数据库资源保留各自授权条件。
材料数据库与 ResPlan 原始数据目前缺少独立许可证文件，在真正公开发布到 GitHub 前，
仓库维护者必须确认来源与再分发权利。具体见根目录 `THIRD_PARTY_NOTICES.md`。
