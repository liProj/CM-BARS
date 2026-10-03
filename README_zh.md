# CM-BARS 中文使用说明

本工程对应论文《Zero-aware multitask probabilistic response surfaces for phenolic adsorption on cellulose nanofibers》，包含主模型、消融实验、对照模型、PFN-RSM、公开示例数据、论文结果记录及绘图代码。

论文主模型为 `code/cmbars2.py` 中的 `CMBARS2(use_gp=False)`，采用零值事件与正值 Beta 分布、低秩系数共享和分块收缩。`code/cmbars.py` 是早期 Tobit 对照模型。推荐通过根目录的 `run.py` 使用，入口已设置正确的主模型配置。

## 1. 环境安装

建议使用 Python 3.13。在工程根目录执行：

```bash
python -m venv .venv
```

Linux / DGX 激活环境：

```bash
source .venv/bin/activate
```

Windows PowerShell 激活环境：

```powershell
.venv\Scripts\Activate.ps1
```

安装主模型环境并检查依赖：

```bash
python -m pip install -r requirements.txt
python run.py doctor
```

若使用 PFN-RSM、LightGBM、XGBoost，则执行：

```bash
python -m pip install -r requirements-full.txt
```

CM-BARS 使用 CPU 运行；PFN 大规模预训练建议使用 GPU。DGX 已有适配设备的 PyTorch 环境可以继续使用。

## 2. 先跑通一个完整例子

```bash
python run.py check
python run.py reference
python run.py demo
python run.py predict --posterior results/demo/posterior.npz --query examples/query.csv
```

这些命令依次检查数据、查看论文保存的结果、运行短链示例、重新读取模型进行预测。`demo` 仅使用一条链、30 次预热和 40 次保留采样，用于检查运行流程，不能用它的结果替代正式论文实验。

`results/demo/` 包含模型文件 `posterior.npz`、预测表 `predictions.csv`、采样诊断 `diagnostics.csv` 和配置 `run.json`。预测表给出三个化合物的吸附均值、预测标准差、90% 区间和零值概率。

## 3. 使用自己的数据

训练 CSV 需要以下六列，名称必须一致：

```text
pH,additive_mM,cnf_pct,tannic_acid,p_coumaric_acid,acetosyringone
```

前三列依次为 pH、酚类浓度（mM）和 CNF 浓度（百分数）。例如 0.25% CNF 填写 `0.25`。后三列为单宁酸、对香豆酸和乙酰丁香酮的吸附百分比。三个响应均需提供，不允许缺失值。

当前工程固定使用 pH 4–8、添加剂 5–15 mM、CNF 0.10–0.40% 的范围，响应支持 `0 ≤ y < 100`。其他材料体系、不同响应数量或恰为 100% 的观测需要相应修改模型设定。

```bash
python run.py fit --data data/bbd_design_responses.csv
python run.py predict --posterior results/fit/posterior.npz --query examples/query.csv --output results/new_conditions.csv
```

查询表只需前三列。`fit` 默认两条链，每条预热和保留各 1000 次。论文全数据拟合预算为：

```bash
python run.py fit --warmup 3000 --samples 3000 --chains 4 --output results/full_fit
```

正式解释结果前，应检查采样诊断和预测区间。`p_zero` 是预测抽样中零值的比例；其余预测数值按吸附百分比尺度报告。

## 4. 重新运行论文验证

```bash
python run.py cv --methods CM-BARS "OLS-Quad (published)" ExtraTrees --protocols LOCO --workers 1
python run.py cv --methods CM-BARS "OLS-Quad (published)" --protocols RKF --repeats 20 --workers 1
python run.py analyze
```

LOCO 为留一条件验证，共 13 折，同一条件的所有中心点重复和三个化合物一起留出。RKF 为条件分组五折，重复 20 次。LOO 则留出单条试验记录，用 `--protocols LOO` 选择。

正式验证默认每条链预热和保留各 1000 次、两条链。`--fast` 缩减至各 300 次，适合调试。完整实验需要较长时间，建议先用单进程跑通，再根据 DGX 内存和 CPU 状况增加 `--workers`。

缓存按代码、数据和运行配置区分，避免把快速测试结果误用于正式结果。每次完成后会在 `results/runs/` 保存独立配置和指标快照。根目录 `results/cv_*.csv` 保存最新一次对应类型的指标，`reference_results/` 保留论文原有结果。

## 5. 使用与训练 PFN-RSM

工程已附带 50,000 步预训练权重，可以直接预测：

```bash
python run.py predict --model pfn --query examples/query.csv --output results/pfn_predictions.csv
python run.py cv --methods "PFN-RSM (prior-fitted)" --protocols LOCO
```

重新训练：

```bash
python run.py pfn-train --steps 50000 --batch 256 --device cuda
```

新权重保存至 `results/pfn_rsm.pt`，不会覆盖 `checkpoints/` 中的随附权重。存在新权重时默认优先使用它，也可通过 `PFN_CHECKPOINT` 指定文件。PFN 使用自己的合成任务先验，与 CM-BARS 的两部分 Beta 模型分别评价。

## 6. 查看结果和生成图表

```bash
python run.py reference
python run.py figures
```

第二条命令使用保存的数据重新生成英文图，输出在 `paper_figures/figures/`，不需要重新训练。原始实验绘图入口和全部消融名称见 [复现指南](docs/REPRODUCIBILITY.md)。

论文保存的平均 RMSE 为 OLS 10.431、CM-BARS 6.417、PFN-RSM 6.332、ExtraTrees 5.646。短链运行和新的依赖环境可能产生不同数值。历史 CM-BARS+ 缓存汇合存在间接信息泄漏，因此该部分保留为补充分析，不能作为无偏泛化性能。

## 7. 工程目录

| 目录 | 内容 |
|---|---|
| `code/` | 主模型、对照、验证、评分和原始绘图代码 |
| `data/` | 15 次试验数据及原论文系数、最优点 |
| `checkpoints/` | 随附 PFN-RSM 预训练权重 |
| `reference_results/` | 论文的保存结果，不随新运行覆盖 |
| `examples/` | 待预测条件示例 |
| `paper_figures/` | 英文绘图入口及对应数值数据 |
| `results/` | 本地新运行输出，不上传版本管理 |
| `tests/` | 数据隔离、模型保存和评分等测试 |
| `docs/` | 模型说明、复现步骤、验证记录 |

公开数据来自 García-Fuentevilla 等的 [原始研究](https://doi.org/10.3390/macromol6020022)，使用这些观测时请引用原始数据论文。完整英文说明见 [README](README.md)。
