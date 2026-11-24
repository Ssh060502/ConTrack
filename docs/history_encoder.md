# 历史编码器原型：离线预训练，不影响正在运行的 RL

这一阶段只学习“历史能否预测物理差异和下一步动力学”，不输出控制修正，不接 cross-attention，也不代表抓取成功率会提高。无需加载 base policy checkpoint 或启动 Isaac Sim。

## 输入与因果对齐

决策时刻 t 的 encoder 输入为最近至多 32 个已完成 transition：

```text
H_t = [(x_i, u_i, x_(i+1)-x_i)]，i = max(0,t-32), ..., t-1
历史 Transformer → 64 维表示 z_t
参数 head：z_t → log(mass_kg)、log(object_friction)
动力学 head：(z_t, x_t, u_t) → x_(t+1)-x_t
```

- x：两手关节位置和速度、各模拟 link 接触力 net_force_w。当前 waterbottle 为 176 维。
- u：已执行的裁剪后 policy residual（58 维）和实际关节位置目标（58 维），共 116 维。ConTrack 的 residual 需要加参考关节位置后才是实际控制目标，因此不能只记录 residual。
- token 共 468 维。没有读取 policy_observation（其中可能包含 oracle 参数），没有使用物体 ground-truth 位姿、参考序列、phase、成功标签或文件编号。
- 编码器只看到 i<t 的动作。动力学 head 的 u_t 是训练时已知的当前动作，只用于预测随后变化，不进入 z_t；x_(t+1) 只作为监督目标。
- 变长历史右侧 padding，并使用 padding mask 和 causal mask。初始 t=0 尚无 transition，不在这个预训练实验中直接推断；以后控制部署需要明确先验和冷启动处理。
- net_force_w 是仿真 link 合力，不是 Sharpa 真机 tactile taxel 数据。要在真机使用，后续必须对齐传感器定义、坐标、频率和标定。

## 数据准备

接受旧采集器和 oracle adapter 评估的 episodes.h5，两者实际参数字段自动兼容。只选指定工具、单一 reference clip、同一 control_dt；过滤明确关闭自碰撞的诊断文件，拒绝未完成数据和跨 reset episode。精确重复的状态/动作/物理条件会去重。

成功和失败轨迹都保留；失败并不等于无效动力学数据。统计归一化仅使用训练 episodes，包含 x、u 和 log 参数。参数作为标签，不作为输入。

默认按完整 episode、按物理条件分层，约 60/20/20 划分 train/validation/test。窗口从各自 episode 生成，绝不跨 split 或 reset。若某条件少于 3 条独立记录，停止并要求补数据。

**默认 episode split 仍共享采集进程/批次、参考动作和已见物理条件，不能宣称独立批次验证或物理泛化。**同批连续 reset 的相关性也没有消除。

`--split-unit file` 可以按采集文件划分，但每个条件至少要有三份独立 collection 文件，数据不足时会明确报错。多条件文件若产生交叉也会拒绝。未来应补独立进程批次、不同物理参数和组合，另做未见条件测试。

## 在服务器运行

先同步新增代码，并在 env_isaaclab 中运行：

```bash
cd ~/ConTrack
conda activate env_isaaclab
python -m unittest discover -s tests -p 'test_history*.py' -v

# 只审计和划分，不启动 GPU 或仿真；递归寻找 episodes.h5，自动筛 waterbottle
python scripts/train_history_encoder.py \
  --tool waterbottle --inputs outputs/physics_validation --prepare-only
```

如果该目录含多个 waterbottle reference clip 或不完整采集，请改为 `--inputs <默认参数目录> <mass6目录> <friction1目录> <friction0.5目录>`，不要把不同 clip 混进同一原型。

先两 epoch 检查训练接口，device 选有资源的 GPU，也可用 cpu：

```bash
python scripts/train_history_encoder.py \
  --tool waterbottle --inputs outputs/physics_validation \
  --device cuda:0 --epochs 2 --logger none
```

接口检查正常后，独立预训练：

```bash
python scripts/train_history_encoder.py \
  --tool waterbottle --inputs outputs/physics_validation \
  --history 32 --stride 2 --batch-size 128 --epochs 100 \
  --device cuda:0 --logger wandb --project ConTrack-history
```

这里 epochs 是离线数据遍历次数，与 PPO 更新轮次或 episode 条数不同。每次启动创建独立目录。数据在 CPU 内存，num_workers=0，不开新仿真，不修改环境或正在训练的 adapter。

## 损失与判断

参数损失：标准化 log(mass)、log(friction) 的 SmoothL1；动力学损失：按训练 state_std 归一化的下一步状态差分 SmoothL1。两者默认等权，可用 `--dynamics-weight` 修改。

物理参数预测用 kg 和摩擦系数 MAE 报告，同时记录训练集 log 参数均值所对应的常数预测器误差。动力学报告归一化差分 MSE，同时记录“始终预测变化为零”的基线。误差先按 episode 平均，再对 episodes 等权平均，并按条件分别报告。

检查的不只是 loss 下降，而是 validation 是否优于常数参数/零变化基线。固定离散条件的插值、组合泛化、真机参数识别尚未验证；接触和运动激励不足时，mass/friction 未必可以从短历史唯一识别。

可用 `--history 1` 与 `--history 32` 作初步对照，观察更长历史是否提供收益。未实现 contrastive、每层参数监督或 policy cross-attention，先保留可解释的辅助任务。

## 输出

目录：`outputs/history_encoder/waterbottle_history_<timestamp>_<id>/`。

- manifest.json：每条 episode 的来源、去重 hash、条件、划分、特征定义与限制。
- normalization.npz：仅训练集拟合的归一化统计。
- config.json：运行参数。
- metrics.jsonl：逐 epoch 训练和验证指标；W&B project 默认 ConTrack-history。
- history_best.pt：按 validation 的两项组合损失选择；不是按抓取成功率选择。
- history_last.pt：最后 epoch 模型。checkpoint 含架构、特征名、归一化、control_dt。
- test_metrics.json：训练结束后仅用最佳 validation 模型评估 test，不用 test 选模型。

下一阶段才是让控制 adapter 使用 z_t，并通过控制目标训练融合模块。参数预测好不等于已证明控制收益；已知参数 adapter 也不是所有历史方法的严格性能上界。

## 分项诊断现有模型，不必重训

同步新增代码后先执行 `python -m unittest discover -s tests -p 'test_history*.py' -v`。

```bash
RUN="$PWD/outputs/history_encoder/waterbottle_history_20261118_205100_2b9cb7"
python scripts/evaluate_history_encoder.py \
  --run "$RUN" --inputs outputs/physics_validation --device cuda:1
```

读取原 history_best.pt、manifest 和 checkpoint 归一化。按 episode 内容 hash 恢复原 validation/test，支持数据目录移动，不用文件时间或目录排序推测划分。任何原 episode 缺失、内容改变或输入特征映射不一致都会停止；额外数据不会加入原 split。

输出到原 run 下全新的 `diagnostics/<timestamp>_<id>/`，不会覆盖原模型和 test_metrics.json：

- diagnostics.json：validation/test 总体、分条件和分特征指标。
- conditions.csv：每个物理条件的质量/摩擦 MAE、相对误差、预测偏差和预测均值；含对应常数基线。
- episodes.csv：逐 episode 指标，方便定位个别轨迹。

动力学分 qpos/qvel/contact_force，分别报告模型和零变化预测器的归一化 MSE、物理单位 MSE、归一化 SmoothL1。当前机器人 qpos 的物理 MSE 单位为 rad²、qvel 为 (rad/s)²、接触力为 N²，不可将三个物理单位直接混成一个分数。

分项 SmoothL1 是为了检查训练目标与 MSE 的差异。全局 checkpoint 选择标准仍是原来的窗口加权组合 SmoothL1，没有改变 reward、预训练目标或重新选择原 checkpoint。对已有 test 反复分析属于描述性诊断，不再能当作新的独立测试；方法调整后需要另采独立批次确认。

## 固定数据的 history=1 / 32 对照

对照脚本依次重新训练两组，各自按 validation 选择 checkpoint，test 不参与选择。使用相同原始 episode、split、归一化拟合数据、样本时刻、训练 seed、batch size、学习率和 loss 权重，仅改变历史长度。

新版 DataLoader 使用独立固定随机生成器，让两组获得相同的窗口 batch 顺序。因此为避免将新版 shuffle 行为与历史长度混淆，推荐两组都重新训练；原来的 32 步模型仍保留，作为参考。网络初始化和 dropout 不保证逐随机数相同，单 seed 对照仅是初步证据。

先两 epoch 检查新评估/对照接口：

```bash
python scripts/run_history_comparison.py \
  --reference-run "$RUN" --inputs outputs/physics_validation \
  --histories 1 32 --device cuda:1 --epochs 2 --logger none
```

正式对照可在 tmux 中运行；进入新会话后需重新设置 RUN：

```bash
tmux new -s history_compare
cd ~/ConTrack
conda activate env_isaaclab
RUN="$PWD/outputs/history_encoder/waterbottle_history_20261118_205100_2b9cb7"
python scripts/run_history_comparison.py \
  --reference-run "$RUN" --inputs outputs/physics_validation \
  --histories 1 32 --device cuda:1 --epochs 100 \
  --logger wandb --project ConTrack-history
```

输出为 `outputs/history_comparison/history_comparison_<timestamp>_<id>/`，包含 history_1/、history_32/、comparison.json、comparison.csv、commands.json。网络顺序运行，不同时抢占两份资源。若 GPU 1 忙，改 device 或用 cpu 做小检查。

若只想训练单步模型，可用 `--histories 1`，但与旧 32 步运行的 batch 随机顺序并未严格匹配。也可直接训练入口使用 `--split-manifest <原 manifest.json>` 来锁定数据，不应仅依赖“仍然用 seed 42”来避免以后新增数据改变划分。
