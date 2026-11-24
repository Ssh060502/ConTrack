# Waterbottle：真实物理参数 adapter

服务器环境为 env_isaaclab。第一轮只验证质量补偿，使用 offhand（不是 lift）基础 policy。

## 默认资产和训练设置

- session：`logs/rsl_rl/sharpa_waterbottle_baseline/grab-s1_waterbottle_offhand_1-rby1a_sharpa-24_504-2026_11_16-12_27_58`
- base checkpoint：该目录下 `model_1999.pt`。
- data：自动读取 session 的 `params/env.yaml` 中顶层 reference_path，使用 BaseLoader 兼容 Isaac YAML 标签。此前采集记录的路径为 `/home/shshao/ConTrack/data/sharpa/grab-s1_waterbottle_offhand_1-rby1a_sharpa-24_504.h5`。
- conditions：`1:2,3:2`，质量倍率 1/3；物体静动摩擦系数均为 2。物体惯量随质量同倍率缩放。实际属性有读回检查和 physics.json。
- correction_scale：0.1，单动作维修正均值的上限；探索噪声独立叠加，执行动作仍裁剪。
- objective：明确 task-only，actor 的 task weight 恒为 1；保留 task+penalty 的原始回报和两个 critic 输出。style 回报仍记录，未加入额外持稳奖励或 residual 惩罚。
- 原 policy 和其归一化统计冻结；新 adapter、critic、探索噪声训练。
- 每环境固定一种物理条件；首帧 reset、关闭 threshold curriculum、pose-break log-only/self-collisions 保持开启，与当前比较流程一致。没有丢弃首条或添加物理 warmup。
- 默认 1000 次 PPO 更新，每次 32 个控制步；建议 200 个环境，两条件各 100 个。

## 先检查

```bash
cd ~/ConTrack
conda activate env_isaaclab
python -m unittest discover -s tests -p 'test_oracle*.py' -v
bash scripts/run_waterbottle_adapter.sh --gpu 0 --num_envs 20 \
  --max_iterations 10 --save_interval 5 --logger tensorboard
```

预期出现零 adapter 与 base 完全一致的断言通过提示、10 次更新、冻结哈希检查完成、Finished 输出。新目录下应有 diagnostics.jsonl；episode 完成后出现每条件 task_return 等曲线。前几次 rollout 未完成 episode 时不伪造回报值，只有 completed_episodes=0。

## 正式训练

```bash
tmux new -s waterbottle_adapter
cd ~/ConTrack
conda activate env_isaaclab
bash scripts/run_waterbottle_adapter.sh --gpu 0 --num_envs 200 \
  --conditions '1:2,3:2' --correction_scale 0.1 --objective task-only \
  --max_iterations 1000 --save_interval 100 \
  --logger wandb --project ConTrack-adapter
```

Ctrl+b 松开再按 d 离开 tmux。运行目录为 `logs/oracle_adapter/waterbottle_train_adapter_<timestamp>_<id>`，包含 model_*.pt、adapter_final.pt、runner_config.json、run_metadata.json、physics.json、diagnostics.jsonl。不要把 mug adapter 加载到 waterbottle。

## 日志

同一个 W&B run 下分别查看 `Oracle/mass1_friction2/...` 和 `Oracle/mass3_friction2/...`：

- task_return/style_return/penalty_return/environment_return：本次 rollout 中完整结束的 episode 的累计回报均值。仅在有完整 episode 时记录；不是成功率，不是每秒归一化值。
- task_plus_penalty_return：task-only actor 使用的未折扣回报。environment_return 是环境原始总回报，不能与实际 actor 目标混为一谈。
- completed_episodes：本次 rollout 结束数量；episode_steps：完成 episode 平均长度。
- correction_abs_mean、correction_abs_peak：修正绝对值的均值和 rollout 峰值；另拆 arm/finger。
- correction_near_limit_fraction：修正达到自身上限 95% 的动作元素比例。
- base_boundary_fraction：原动作 abs>=0.999 的元素比例；不是 adapter 导致的裁剪。
- mean_outside_bounds_fraction：base+correction 超出 [-1,1] 的比例。
- sample_outside_bounds_fraction：加探索噪声后动作超出边界的比例。
- sample_clip_abs_mean：采样动作裁剪造成的平均绝对变化。
- exploration_abs_mean：采样动作相对均值的平均绝对变化。
- position_error_m、rotation_error_rad：非终止步骤的物体跟踪误差，排除自动 reset 后状态。
- carrying_phase_position_error_m/rotation_error_rad：只在参考物体原点比参考初始值高 3 cm 的非终止步骤记录，不代表已实际抓住。

上述动作统计按 rollout 的环境—时间—动作元素聚合，峰值单独取最大值；不上传每一步完整张量。完整 episode 回报跨 rollout 累加，不在日志刷新时清空半条 episode。所有标量同步写入 diagnostics.jsonl。

PPO 原有任务权重日志位于 `Curriculum/cmdp/task_weight`；task-only 应恒为 1。也在 diagnostics.jsonl 中记录。未加入“自动成功率”，仍需固定评估和视频分类。

## 评估

```bash
# 同一参数下先评估基础模型
bash scripts/run_waterbottle_adapter.sh --gpu 0 --mode evaluate --policy base \
  --num_envs 1 --conditions '3:2' --episodes 10 --video

# 换为本次保存的模型；也可选 model_100.pt、model_500.pt
bash scripts/run_waterbottle_adapter.sh --gpu 0 --mode evaluate --policy adapter \
  --adapter_checkpoint /完整训练目录/adapter_final.pt \
  --num_envs 1 --conditions '3:2' --episodes 10 --video
```

默认条件 `1:2` 和未见质量 `6:2` 分别重复同样的 base/adapter 对比；输出在 `waterbottle_evaluate_*`，含 episodes.h5、videos/ 和元数据。evaluation 的实际 checkpoint 会同时记录到 H5 和 run_metadata.json。

## 参数和兼容性

支持 `--session DIR --base_checkpoint FILE --data FILE` 覆盖资产路径，`--conditions` 覆盖物理条件，`--correction_scale` 覆盖修正上限，`--objective inherit` 使用源 CMDP 配置。

加载 adapter_checkpoint 后 policy/algorithm 配置以 checkpoint 保存值为准，CLI 的 correction_scale/objective 不会覆盖已训模型。续训恢复优化器，但环境会新建并从首帧开始，不恢复 PhysX 状态或半条 episode，不能宣称逐步完全复现。原 mug 启动脚本仍默认 mug 资产、0.5 上限和原 CMDP 设置。

本地没有 Torch/Isaac，Torch 网络和日志测试需在服务器运行；离线数值测试与语法检查不等于完成真实仿真验证。

## model_1000 结果复核与参数输入对照

固定使用 `waterbottle_train_adapter_20261118_195820_40067b/model_1000.pt`，不选择最新文件。同步 evaluate_selected_adapters.py、oracle_adapter.py、oracle_physics.py 以及相关测试后运行。

```bash
python -m unittest discover -s tests -p 'test_selected_adapter_plan.py' -v
python -m unittest discover -s tests -p 'test_oracle_physics.py' -v
```

先检查三个进程、每组两条：

```bash
python scripts/evaluate_selected_adapters.py \
  --experiment waterbottle_mass --gpu 0 --episodes 2 \
  --conditions '3:2' --seeds 42 --compare-nominal-mass-input
```

三组 actual mass 都是 ×3、friction 都是 2。base 不使用参数输入，correct adapter 使用读回质量约 0.16605 kg，oracle_nominal_mass adapter 输入约 0.05535 kg，但仿真质量和惯量仍为 ×3。只修改 observation 的 oracle mass 一维，不写物理属性，也不修改网络权重、摩擦输入或 reset 流程。

正式复核（15 组、每组十条）：

```bash
python scripts/evaluate_selected_adapters.py \
  --experiment waterbottle_mass --gpu 0 --episodes 10 \
  --conditions '1:2,3:2' --seeds 42 43 44 --compare-nominal-mass-input
```

每 seed 为五组：base/default、adapter/default、base/mass3、adapter/mass3、adapter/mass3 但网络输入默认质量。默认条件下输入本来相同，不重复制造“错标签”组。每组各用独立新进程；seed 不保证初始位姿多样性，仍是同参考首帧初始化的复现检查。

未见质量评估（18 组、每组十条）：

```bash
python scripts/evaluate_selected_adapters.py \
  --experiment waterbottle_mass --gpu 0 --episodes 10 \
  --conditions '2:2,4:2,6:2' --seeds 42 43 44
```

两批分别创建 `outputs/adapter_evaluation/selected_adapters_<time>_<id>/`。每组目录包含 seed、实际参数及是否 oracle_nominal_mass，内部是 episodes.h5、videos/ 和元数据。plan.json 保存每组实际 seed 和条件。入口同时检查任务完成文件、实际属性跨控制组一致，以及网络输入质量是否符合指定对照。

网络输入控制也可直接用于单次评估：`--oracle-mass-scale 1`。该参数仅允许 evaluate/adapter，不允许训练或 base 模式。physics.json/H5 的 physics_json 仍记录真实物理属性，run_metadata.json/H5 的 metadata_json 另记 oracle_mass_scale_override 和 oracle_input_mass_kg，不得将网络标签当作仿真质量。

若正确输入和默认输入都改善，可能主要是一般性策略修正，不能宣称网络依赖质量标签；若正确输入稳定优于默认输入，才支持显式质量条件的贡献。上述测试不更新任何模型。
