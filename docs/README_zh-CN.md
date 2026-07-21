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

agent = AcousticAgent.create(
    scene="geometry",
    room=room,
    source=[1.2, 1.1, 1.5],
    receiver=[4.7, 2.8, 1.4],
    receiver_model={"type": "mono"},
    source_model={"type": "omni"},
    quality="simulation",
    duration_s=2.0,
    fs=16000,
)

result = agent.run()
rir = result.rir
```

## Floorplan 示例

Floorplan 场景由 Mohamed Abouagour 和 Eleftherios Garyfallidis 发布的 ResPlan
户型图数据集转换而来。论文为《ResPlan: A Large-Scale Vector-Graph Dataset of
17,000 Residential Floor Plans》，arXiv:2508.14006（2025）；数据地址为
https://www.kaggle.com/datasets/resplan/resplan。

```python
from acoustic_agent import AcousticAgent

agent = AcousticAgent.create(
    scene="floorplan",
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

Floorplan 和 Custom 页面中的 Acoustic furniture 支持语义自动摆放。紧凑程度可选
`sparse`、`balanced` 或 `compact`；相同 seed 可复现。自动布局会避开门洞、声源、
麦克风和已有手工家具，生成后仍可拖动、旋转、删除或继续添加家具。

## 定制户型示例

定制户型不需要 GPT 或 VLM API。内置的本地生成器可以解析简短的中英文房间描述，
生成可编辑、可校验的米制户型 JSON，再复用 Floorplan 的门洞、材料和 RIR 求解流程。

```python
from acoustic_agent import AcousticAgent, FloorplanBuilder

spec = FloorplanBuilder.from_text(
    "12m x 9m，三室两厅一厨两卫，一个储物间",
    seed=42,
)
agent = AcousticAgent.create(
    scene="custom",
    spec=spec,
    source_room="living_0",
    receiver_room="bedroom_2",
    quality="preview",
    duration_s=2.0,
    fs=16000,
)
rir = agent.run().rir
```

## 多声源、音乐与噪声

RIR 描述的是声源位置到接收器之间的传播，与输入是人声、钢琴还是噪声无关。同一
位置只替换音频时可以复用 RIR；背景声位于另一位置时，应单独计算一条 RIR：

```python
from acoustic_agent import mix_audio_at_snr, render_audio

sources = agent.run_sources({
    "voice": [1.2, 1.1, 1.5],
    "piano_1": [4.8, 1.0, 1.2],
})
voice_wet = render_audio(voice_samples, sources["voice"].rir)
piano_wet = render_audio(piano_samples, sources["piano_1"].rir)
room_mix = mix_audio_at_snr(voice_wet, piano_wet, snr_db=10, normalize=True)
```

Web 工作台内置项目口播、背景人声、两段钢琴、粉红噪声底声，也支持固定 seed 的
白噪声/粉红噪声/棕噪声和用户上传音频。SNR 是两路信号分别经过 RIR 后在接收端的
宽带 RMS 比值，并不是简单的背景音量。
启用 Background source 后会出现独立坐标和 3D 标记，并在麦克风运动时逐帧更新背景
RIR。修改音频或 SNR 不需要重算；修改背景位置需要重新运行仿真。

## 动态与批量生产

静态与动态统一使用 `run`：

```python
dynamic = agent.run(motion={
    "mode": "approach",
    "moving": "receiver",
    "distance_m": 1.0,
    "keyframe_spacing_m": 0.25,
})
rir_frames = dynamic.rirs
```

同一场景的大量坐标对使用 `agent.run_batch(pairs, workers=4)`；大量不同户型、
不同档位或不同运动模式使用 `AcousticAgent.run_many(jobs, workers=4)`。两种结果均可
调用 `save_npz(...)`，跨场景归档还会写入 JSON manifest。完整说明见
[`API.md`](API.md)。

Custom 页面支持两种输入：户型图模式把所选图片和 “Copy ChatGPT prompt” 得到的
提示词一起交给 ChatGPT；文本模式会把住宅描述直接写入提示词。两种方式都只需将
ChatGPT 输出的 JSON 粘贴回页面并点击 “Apply floor plan”。图片只在浏览器本地预览，
不会上传到服务器。Width 与 Depth 始终等比例校准，Height 独立设置。完整格式与测试方式见
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

- [`API.md`](API.md)
- [`INSTALLATION.md`](INSTALLATION.md)
- [`CONFIGURATION.md`](CONFIGURATION.md)
- [`FLOORPLAN.md`](FLOORPLAN.md)
- [`CUSTOM_FLOORPLAN.md`](CUSTOM_FLOORPLAN.md)
- [`RESOURCES.md`](RESOURCES.md)

项目代码和项目原创文档采用 Apache-2.0。SOFA 和数据库资源保留各自授权条件。
Floorplan V1 是 ResPlan 的转换资源，遵循 CC BY-NC-SA 4.0，要求署名、非商业使用、
标注修改并以相同许可共享衍生数据。材料数据库采用混合来源条款。具体见根目录
`THIRD_PARTY_NOTICES.md` 及各资源目录中的 `DATA_LICENSE.md`。
