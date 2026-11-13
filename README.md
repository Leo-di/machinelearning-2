# 机器学习与因子筛选：Lasso 稳定性优化

当前主文件是 [4.1机器学习与因子筛选_整合.ipynb](4.1机器学习与因子筛选_整合.ipynb)。仓库包含运行代码、公开来源的数据、已执行的 Notebook 和优化记录。

如需直接转发给同学，可下载 [完整代码与数据压缩包](Lasso作业_代码与数据.zip)。先完整解压，再运行 Notebook。

## 快速运行

推荐 Python 3.13，已验证环境为 Python 3.13.16。在仓库根目录或完整解压后的文件夹中运行：

**macOS / Linux**

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m jupyterlab
```

**Windows**

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m jupyterlab
```

打开主 Notebook，选择刚创建环境的 Python 内核，重启内核并依次运行全部单元。Notebook、`lasso_stability.py` 和 `tidy_finance.sqlite` 必须放在同一目录，并从该目录启动 Jupyter。初次安装依赖需要联网；安装完成后正常运行无需下载数据。

## 文件说明

| 文件或目录 | 用途 |
|---|---|
| `4.1机器学习与因子筛选_整合.ipynb` | 当前主版本，包含模型、目标函数、方法说明和运行输出 |
| `lasso_stability.py` | Lasso 调参、分块重采样、稳定筛选及重拟合 |
| `tidy_finance.sqlite` | 运行所需的四表数据库 |
| `requirements.txt` | 已验证依赖版本 |
| `lasso_stability_results/` | 当前主版本的稳定性、误差、调参及收敛日志 |
| `build_tidy_finance.py`、`data_sources/` | 数据重建脚本、原始公开数据和来源记录 |
| `运行说明.txt` | 详细运行与结果解释说明 |
| `4.1机器学习与因子筛选_Lasso方法对照.ipynb`、`lasso_method_results/` | 先前 one-SE 等方法对照，属于历史实验 |
| `lasso_results/` | 先前求解与搜索效率优化记录，属于历史实验 |

## 当前 Lasso 方法

- 按时间划分前 80% 训练期、后 20% 历史测试期。
- 原特征表保留 153 列；Lasso 内根据变量定义排除年份、重复无风险收益、q 市场项及相应交互项，有效候选为 111 个。
- 在训练期做 5 折外层扩展窗口验证；各折独立完成内层 3 折选参、100 次 12 个月移动分块重采样、筛选和重拟合。
- 保留入选频率至少 70%、非零系数方向一致率至少 90% 的变量，最多 5 个，允许空模型。
- 最终训练固定随机种子 42，种子 2026 用于复现性复核，不据此替换模型。
- 保留粗细搜索、热启动、固定容差、收敛检查及重试日志。

本次保存结果中，各行业保留 1–4 个变量；9/10 个行业的两个随机种子产生相同集合。相对同一外层验证流程的初始 Lasso 集合，跨折 Jaccard 在 6 个行业提高、4 个行业降低。完整结果见 `lasso_stability_results/`。

## 数据与解释口径

数据包含 Fama–French 三因子、十行业收益、q 因子和宏观变量。合并样本为 1967-01 至 2022-11，共 10 个行业、6,710 条月度记录。来源链接、数据版本、单位与兼容处理见 [数据来源说明](data_sources/数据来源与兼容说明.txt) 和 [来源清单](data_sources/manifest.json)。数据库由公开数据重建，历史数据修订可能使结果与原课堂输出不同。

本数据的 `macro_dy` 实际为股息增长率，不是通常的股息收益率。模型用同期变量解释同期收益，不是可实时实施的下一期收益预测。筛选频率是固定所选 alpha 条件下的经验稳定性诊断，不是显著性或因果证据，也不提供理论错误发现控制保证。历史测试集已在之前的方法比较中查看，因此本次结果属于探索性评估。

本次修改限于 Lasso 及直接对应说明。其他模型代码与原分析正文保留，后文历史文字数值需要与当前运行输出核对。
