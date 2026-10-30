# 小组项目：推动技能接入说明

## 贡献与衔接

Buqing Ou 的抓取放置流程继续使用原来的规划器、环境和执行器。
本次贡献增加定距推动环境、SAC 策略、中文/英文推动指令校验，并把两类任务接入同一个 `run_instruction`。
任务分派先识别动作类别，再由对应的 LLM 规划器和确定性校验器决定是否执行。
分派本身不是 LLM 推理，SAC 也不是由 LLM 现场训练。

默认自动分派：推动使用冻结的 SAC，抓取放置仍使用原来的 scripted 默认值。
Buqing 的 SAC 抓取放置命令继续显式使用 `--executor rl --rl-model <carry checkpoint>`。

## 运行前提

在仓库根目录执行以下命令。沿用小组原有 ManiSkill 环境；推动还使用
`imageio`、`imageio-ffmpeg`、`scipy`、`matplotlib`（见项目 requirements.txt）。
本次验证环境：Python 3.12、ManiSkill 3.0.1、SAPIEN 3.0.3、Stable-Baselines3 2.9.0、
PyTorch 2.14.1；Windows CPU 物理仿真。机器人资产需按 ManiSkill 的提示安装。

Windows 的 `run.cmd` 依次查找 `.venv/python.exe`、`.venv/Scripts/python.exe`。
也可以在 PowerShell 中把 `MYPICKPLACE_PYTHON` 设置为自己已配置环境的 python.exe 绝对路径。
启动器不会安装依赖；失败时返回退出码，不再等待按键。
Linux/macOS 或已激活环境可直接运行：

```bash
python MyPickPlace-ManiSkill/MyPickPlace-ManiSkill/scripts/run_instruction.py --help
```

推动目前仅支持本地 Ollama；服务默认地址 `http://127.0.0.1:11434`，模型 `llama3:8b`。
在安装好 Ollama 后运行 `ollama pull llama3:8b`，确保服务已启动。
`--provider`、`--model`、`--base-url` 及 `LLM_PROVIDER`、`LLM_MODEL`、`LLM_BASE_URL`
在两类任务中沿用同样含义；`--model` 始终指语言模型。
推动策略通过 `--push-model` 指定，不会误用抓取放置的 `--rl-model`。

## 同一个入口的命令

中文推动，使用仓库附带的约 0.95 MB 策略，无需重新训练：

```powershell
.\run.cmd run_instruction --instruction "把蓝色方块向前推10厘米。" --seed 540001 --output runs/my_push
```

英文推动，只检查规划、不执行机器人动作：

```powershell
.\run.cmd run_instruction --instruction "Push the blue cube forward 5 cm." --plan-only --output runs/my_push_plan
```

原有抓取放置默认流程：

```powershell
.\run.cmd run_instruction --instruction "Put the blue cube in the green goal region." --output runs/my_pick_place
```

原有 SAC 抓取放置流程（需要 Buqing 提供相应策略文件）：

```powershell
.\run.cmd run_instruction --instruction "Put the blue cube in the green goal region." --executor rl --rl-model runs/carry_curriculum_40000/training/best_model.zip
```

每次使用新的输出目录，或省略 `--output` 自动生成时间戳目录，避免覆盖证据。
`--skill auto|push|pick_place` 可显式选择任务；冲突或混合动作会拒绝执行。
`--plan-file` 仍只用于原有抓取放置离线重放，日志会明确标为离线来源。

## 输出接口

两个任务均生成 `run.json`，保留原有 `plan`、`execution`、`status` 字段，新增：

| 字段 | 含义 |
|---|---|
| schema_version | 当前为 1 |
| skill | push 或 pick_place |
| environment_id | 使用的模拟环境 |
| control_steps | 实际物理控制步数 |
| physical_success | 实际成功与否；未执行为 null |
| reason | 拒绝、失败或执行结果原因 |

两种 `plan` 的内部字段仍按各自任务定义；调用方应先读取 `skill`。
推动另存 `push/llm_exchange.json`、`plan.json`、`result.json`、`push_trace.json`
及成功生成的视频 `execution.mp4`。失败与拒绝也写日志；没有脚本推动补救或隐式物理重试。
退出码：0=完成或仅规划成功；2=拒绝；1=错误或执行失败。

## 范围和结果

推动场景固定为单个蓝色方块；“向前”明确对应世界坐标 +X。
接受 3–15 cm，训练和主要评估距离为 5、10、15 cm；区间内其他距离不代表经过同等验证。
接近和撤回为脚本动作，推动阶段由 SAC 控制 XY；Z、姿态和夹爪固定。
成功要求最终纵向及横向误差均不超过 1 cm、低速度和稳定性达标，且没有过度抬起/倾斜。
不是任意物体视觉识别，也不支持“先推再抓”、其他方向、多物体或多步任务。

此前独立推动评估：60 次 SAC 成功 58 次（5 cm 20/20，10 cm 20/20，15 cm 18/20）；
相同场景脚本基线也是 58/60，因此不能声称 SAC 优于脚本。
这些结果来自整合前的冻结模型，随附 `models/push` 的摘要与逐次评估可核对。
本次统一入口复测与原有测试结果见 `integration_validation.json`。
抓取放置的 scripted 实际执行已验证；仓库没有 Buqing 的 carry checkpoint，
因此本次未声称完成其 SAC 模型的端到端执行验证。

## 训练与测试

```powershell
.\run.cmd train_push --steps 12000 --seed 1 --output runs/push_training
.\run.cmd evaluate_push_language --output runs/push_language_checks
```

使用同一个已配置环境执行全部测试：

```bash
python -m unittest discover -s MyPickPlace-ManiSkill/MyPickPlace-ManiSkill/tests
```

`models/push/config.json` 与 checkpoint 必须放在一起，执行前核对控制契约。
训练、视频和运行产物在 `runs/` 下，不应提交虚拟环境、Ollama 模型或本机路径到 GitHub。
