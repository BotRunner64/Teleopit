# g1_bridge_sdk

C++ DDS 桥接库，用 pybind11 封装 unitree_sdk2，让 Python 以接近零延迟（< 0.5 ms）访问 Unitree G1 的实时通信接口。

所有 DDS 发布/订阅运行在原生 C++ 线程中，Python 侧只需调用简单的 get/set 方法。

## 依赖

- CMake >= 3.10
- GCC >= 9.4（支持 C++17）
- pybind11 >= 2.6
- Unitree SDK2（已内置于 `thirdparty/unitree_sdk2/`，无需手动安装）
- Cyclone DDS（unitree_sdk2 依赖，通常随 unitree_sdk2_python 附带）

## 安装

```bash
pip install pybind11
pip install third_party/g1_bridge_sdk
```

安装时自动触发 CMake 构建，编译 `g1_bridge_sdk.cpython-*.so`。

如编译失败，可先运行系统依赖安装脚本：

```bash
bash third_party/g1_bridge_sdk/setup_ubuntu.sh
```

## Python API

```python
import g1_bridge_sdk

bridge = g1_bridge_sdk.G1Bridge(
    network_interface="eth0",   # 连接 G1 的网卡
    publish_hz=200              # 指令发布频率（默认 200 Hz）
)
```

| 方法 | 说明 |
|------|------|
| `wait_for_state(timeout_sec=5.0)` | 阻塞等待第一帧 LowState，超时返回 False |
| `get_state()` | 返回 `(qpos[29], qvel[29], quat[4], ang_vel[3])` numpy 数组 |
| `get_state_counter()` | 返回累计收到的 LowState 帧数 |
| `get_wireless_remote()` | 返回 40 字节无线遥控数据 |
| `get_mode_machine()` | 返回当前 mode_machine 值 |
| `set_target(target, kp, kd)` | 设置目标关节位置和 PD 增益（各 29 元素） |
| `lock_joints()` | 锁定当前关节位置 |
| `set_damping()` | 切换为阻尼模式（急停用） |
| `start_publish()` | 启动指令发布线程 |
| `stop_publish()` | 停止指令发布线程 |
| `check_mode()` | 查询当前运动模式，返回 `(code, name)` |
| `select_mode(name)` | 切换运动模式（如 `"ai"`, `"normal"`） |
| `release_mode()` | 释放当前模式，进入低级控制 |

## 使用场景

- **Onboard sim2real**：`scripts/run/run_onboard_sim2real.py`（G1 NX 机载端）
- **Pico4 真机遥操作**：`scripts/run/run_sim2real.py`
- **独立站立测试**：`scripts/run/standalone_standing.py`

详细部署说明见 [sim2real.md](sim2real.md)。
