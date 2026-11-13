"""Lasso tuning and block-bootstrap screening using training data only.

Selection frequencies are empirical diagnostics, not significance probabilities
or the error-control guarantees of classical stability selection.
"""
import warnings
from copy import deepcopy
from time import perf_counter

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer

MAX_FACTORS = 5
ALPHAS = np.logspace(-5, 0, 40)
FINE_POINTS = 17
LASSO_TOL = 1e-4
ITERATION_BUDGETS = (5_000, 50_000, 200_000)

def fit_sparse_lasso(x_train, y_train, folds):
    """仅接收训练期数据；先检查稀疏性，再滚动验证，最后局部细搜。"""
    started = perf_counter()
    fit_rows, search_rows = [], []
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    if not np.isfinite(x_train).all() or not np.isfinite(y_train).all():
        raise ValueError("Lasso 训练样本存在非有限值")
    if not folds:
        raise ValueError("至少需要一个时间验证折")

    scaler = StandardScaler().fit(x_train)
    z_full = np.asfortranarray(scaler.transform(x_train))
    fold_data = []
    for train, valid in folds:
        if (len(train) == 0 or len(valid) == 0 or min(train) < 0
                or max(train) >= min(valid) or max(valid) >= len(y_train)):
            raise ValueError("验证折必须位于训练期内部，且训练月份严格早于验证月份")
        fold_scaler = StandardScaler().fit(x_train[train])
        fold_data.append((
            np.asfortranarray(fold_scaler.transform(x_train[train])), y_train[train],
            fold_scaler.transform(x_train[valid]), y_train[valid],
        ))

    def new_lasso():
        return Lasso(tol=LASSO_TOL, warm_start=True, selection="cyclic")

    def fit_checked(estimator, z, y, stage, fold):
        # 每次重试沿用同一数据、同一 alpha 的系数，不放宽 tol。
        gap_limit = LASSO_TOL * float(np.mean((y - y.mean()) ** 2))
        for attempt, budget in enumerate(ITERATION_BUDGETS, 1):
            estimator.set_params(max_iter=budget)
            tic = perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                estimator.fit(z, y)
            convergence_messages = [str(w.message) for w in caught
                                    if issubclass(w.category, ConvergenceWarning)]
            for warning in caught:
                if not issubclass(warning.category, ConvergenceWarning):
                    warnings.warn(str(warning.message), warning.category)
            gap = float(estimator.dual_gap_)
            converged = (not convergence_messages and np.isfinite(gap)
                         and gap <= gap_limit + 10 * np.finfo(float).eps)
            loss = mean_squared_error(y, estimator.predict(z)) / 2
            fit_rows.append({
                "阶段": stage, "拟合样本": fold, "alpha": float(estimator.alpha),
                "尝试次数": attempt, "本次迭代上限": budget,
                "实际迭代数": int(estimator.n_iter_), "dual_gap": gap,
                "收敛阈值": gap_limit, "已收敛": bool(converged),
                "入选数": int(np.count_nonzero(estimator.coef_)),
                "目标函数": float(loss + estimator.alpha * np.abs(estimator.coef_).sum()),
                "耗时（秒）": perf_counter() - tic,
                "收敛警告": " | ".join(convergence_messages),
            })
            if converged:
                return True
        return False

    def search_stage(alphas, stage):
        rows, models = {}, {}
        path_model = new_lasso()
        # 由强到弱沿正则化路径热启动；每个模型的系数单独保存。
        for alpha in sorted(set(map(float, alphas)), reverse=True):
            path_model.set_params(alpha=alpha)
            ok = fit_checked(path_model, z_full, y_train, stage, "完整训练期")
            count = int(np.count_nonzero(path_model.coef_))
            rows[alpha] = {
                "阶段": stage, "alpha": alpha, "完整训练期已收敛": ok,
                "入选数": count, "验证收敛折数": 0, "验证MSE": None,
                "候选状态": "待验证" if ok and count <= MAX_FACTORS else
                             ("超过变量上限" if ok else "完整训练期未收敛"),
            }
            if ok and count <= MAX_FACTORS:
                models[alpha] = deepcopy(path_model)

        mse_by_alpha = {alpha: [] for alpha in models}
        for fold, (z, y, z_valid, y_valid) in enumerate(fold_data, 1):
            path_model = new_lasso()  # 不跨验证折共享系数或标准化参数。
            for alpha in sorted(models, reverse=True):
                path_model.set_params(alpha=alpha)
                if fit_checked(path_model, z, y, stage, f"验证折{fold}"):
                    mse_by_alpha[alpha].append(mean_squared_error(y_valid, path_model.predict(z_valid)))
        for alpha, errors in mse_by_alpha.items():
            rows[alpha]["验证收敛折数"] = len(errors)
            if len(errors) == len(fold_data):
                rows[alpha].update({"验证MSE": float(np.mean(errors)), "候选状态": "可选"})
            else:
                rows[alpha]["候选状态"] = "存在未收敛验证折"
        search_rows.extend(rows.values())
        eligible = [row for row in rows.values() if row["候选状态"] == "可选"]
        if not eligible:
            return None
        best = min(eligible, key=lambda row: (row["验证MSE"], row["alpha"]))
        return best, models[best["alpha"]]

    coarse = search_stage(ALPHAS, "粗搜")
    if coarse is None:
        raise ValueError("没有收敛且满足变量上限的候选；请检查求解日志或扩大 alpha 范围。")
    position = int(np.argmin(np.abs(ALPHAS - coarse[0]["alpha"])))
    lower, upper = ALPHAS[max(0, position - 1)], ALPHAS[min(len(ALPHAS) - 1, position + 1)]
    fine_alphas = [a for a in np.geomspace(lower, upper, FINE_POINTS)
                  if not np.any(np.isclose(a, ALPHAS, rtol=1e-12, atol=0))]
    fine = search_stage(fine_alphas, "细搜") if fine_alphas else None
    # 始终保留粗搜解；细搜未改善验证误差时不替换。
    best = min([item for item in [coarse, fine] if item is not None],
               key=lambda item: (item[0]["验证MSE"], item[0]["alpha"]))
    model = make_pipeline(scaler, best[1])  # 两步均已拟合，不再重复训练。
    summary = {
        "粗搜alpha": coarse[0]["alpha"], "最终alpha": best[0]["alpha"],
        "粗搜验证RMSE（百分点）": 100 * np.sqrt(coarse[0]["验证MSE"]),
        "最终验证RMSE（百分点）": 100 * np.sqrt(best[0]["验证MSE"]),
        "粗搜入选数": coarse[0]["入选数"], "最终入选数": best[0]["入选数"],
        "候选数": len(search_rows), "拟合尝试数": len(fit_rows),
        "重试次数": sum(row["尝试次数"] > 1 for row in fit_rows),
        "未收敛排除候选数": sum("未收敛" in row["候选状态"] for row in search_rows),
        "训练调参耗时（秒）": perf_counter() - started,
    }
    return model, float(np.sqrt(best[0]["验证MSE"])), {
        "搜索记录": search_rows, "求解日志": fit_rows, "优化汇总": summary,
    }


def block_indices(n, block_length, rng):
    """Sample overlapping consecutive blocks, then truncate to n observations."""
    if not 1 <= block_length <= n:
        raise ValueError("分块长度必须介于 1 和训练样本数之间")
    starts = rng.integers(0, n - block_length + 1, size=int(np.ceil(n / block_length)))
    return (starts[:, None] + np.arange(block_length)).ravel()[:n]


def mask_features(x, mask):
    return x * mask


def solve_checked(z, y, alpha, label, logs):
    model = Lasso(alpha=float(alpha), tol=LASSO_TOL, warm_start=True)
    gap_limit = LASSO_TOL * float(np.mean((y - y.mean()) ** 2))
    for attempt, budget in enumerate(ITERATION_BUDGETS, 1):
        start = perf_counter()
        model.set_params(max_iter=budget)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(z, y)
        messages = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
        for warning in caught:
            if not issubclass(warning.category, ConvergenceWarning):
                warnings.warn(str(warning.message), warning.category)
        gap = float(model.dual_gap_)
        ok = not messages and np.isfinite(gap) and gap <= gap_limit + 10 * np.finfo(float).eps
        logs.append({
            "拟合样本": label, "alpha": float(alpha), "尝试次数": attempt,
            "本次迭代上限": budget, "实际迭代数": int(model.n_iter_),
            "dual_gap": gap, "收敛阈值": gap_limit, "已收敛": bool(ok),
            "入选数": int(np.count_nonzero(model.coef_)),
            "目标函数": float(mean_squared_error(y, model.predict(z)) / 2 + alpha * np.abs(model.coef_).sum()),
            "耗时（秒）": perf_counter() - start, "收敛警告": " | ".join(messages),
        })
        if ok:
            return model
    raise RuntimeError(f"{label} 在重试后仍未收敛，停止计算，避免使用不可靠的入选频率")


def mean_jaccard(supports):
    values = []
    for i, a in enumerate(supports):
        for b in supports[:i]:
            union = np.count_nonzero(a | b)
            values.append(np.count_nonzero(a & b) / union if union else 1.0)
    return float(np.mean(values)) if values else None


def fit_stable_lasso(x, y, feature_names, *, n_resamples=100, block_length=12,
                     frequency_cutoff=0.70, sign_cutoff=0.90, seed=42):
    """Fit the entire screening procedure on the provided training sample only.

    Alpha is selected by three inner expanding-window splits. Bootstrap fits
    condition on that alpha; frequencies do not include alpha-selection uncertainty.
    Excluded columns and their interactions stay zero in the final model so that
    callers can retain their original feature ordering and plotting code.
    """
    start = perf_counter()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or x.shape != (len(y), len(feature_names)):
        raise ValueError("特征名称、矩阵和目标变量维度不一致")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("训练数据含非有限值")
    if len(y) < 132 or n_resamples < 2:
        raise ValueError("至少需要 132 条训练观测和两次重采样")
    if not 0 < frequency_cutoff <= 1 or not 0.5 < sign_cutoff <= 1:
        raise ValueError("入选频率阈值须在 (0,1]，符号一致率阈值须在 (0.5,1]")
    excluded_bases = {"factor_q_year", "factor_q_f", "factor_q_mkt"}
    eligible = np.array([name.split(" × ")[-1] not in excluded_bases for name in feature_names])
    columns = np.flatnonzero(eligible)
    if not len(columns):
        raise ValueError("没有可用的 Lasso 候选变量")
    x_candidates = x[:, columns]
    inner_folds = list(TimeSeriesSplit(n_splits=3, test_size=24).split(y))
    preliminary, _, tuning = fit_sparse_lasso(x_candidates, y, inner_folds)
    alpha = float(preliminary.named_steps["lasso"].alpha)
    base_support = np.zeros(x.shape[1], dtype=bool)
    base_support[columns] = preliminary.named_steps["lasso"].coef_ != 0

    rng = np.random.default_rng(seed)
    boot_coef = np.zeros((n_resamples, len(columns)))
    sampling_logs = []
    for draw in range(n_resamples):
        sampled = block_indices(len(y), block_length, rng)
        z = np.asfortranarray(StandardScaler().fit_transform(x_candidates[sampled]))
        fitted = solve_checked(z, y[sampled], alpha, f"重采样{draw + 1}", sampling_logs)
        boot_coef[draw] = fitted.coef_

    positive = (boot_coef > 0).sum(axis=0)
    negative = (boot_coef < 0).sum(axis=0)
    counts = positive + negative
    frequency = counts / n_resamples
    consistency = np.divide(np.maximum(positive, negative), counts,
                            out=np.zeros(len(columns)), where=counts > 0)
    majority_sign = np.sign(positive - negative)
    median_strength = np.median(np.abs(boot_coef), axis=0)
    passed = np.flatnonzero((frequency >= frequency_cutoff) & (consistency >= sign_cutoff))
    ranked = sorted(passed, key=lambda j: (-frequency[j], -consistency[j],
                                         -median_strength[j], feature_names[columns[j]]))
    retained = ranked[:MAX_FACTORS]
    mask = np.zeros(x.shape[1], dtype=bool)
    mask[columns[retained]] = True
    qualified_mask = mask.copy()
    scaler = StandardScaler().fit(x)
    z = np.asfortranarray(scaler.transform(x))
    # If restricted refitting reverses a selected variable's direction, discard
    # that variable and refit. Do not refill its place with an unstable variable.
    for step in range(MAX_FACTORS + 1):
        final = solve_checked(np.asfortranarray(z * mask), y, alpha,
                              f"稳定集合重拟合{step + 1}", sampling_logs)
        sign_conflicts = [columns[j] for j in retained
                          if mask[columns[j]] and final.coef_[columns[j]] != 0
                          and np.sign(final.coef_[columns[j]]) != majority_sign[j]]
        if not sign_conflicts:
            break
        mask[sign_conflicts] = False
    else:
        raise RuntimeError("稳定集合重拟合未终止")
    transformer = FunctionTransformer(mask_features, kw_args={"mask": mask}).fit(z)
    pipeline = make_pipeline(scaler, transformer, final)
    stability = []
    lookup = {column: j for j, column in enumerate(columns)}
    for column, name in enumerate(feature_names):
        j = lookup.get(column)
        final_selected = final.coef_[column] != 0
        if j is None:
            reason = "预先剔除非因子或重复市场项"
        elif frequency[j] < frequency_cutoff:
            reason = "入选频率不足"
        elif consistency[j] < sign_cutoff:
            reason = "符号一致率不足"
        elif not qualified_mask[column]:
            reason = "超过五变量上限"
        elif not mask[column]:
            reason = "重拟合后方向反转"
        elif not final_selected:
            reason = "重拟合系数为零"
        else:
            reason = "最终保留"
        stability.append({
            "变量": name, "纳入候选": bool(eligible[column]),
            "入选次数": int(counts[j]) if j is not None else 0,
            "重采样次数": n_resamples,
            "入选频率": float(frequency[j]) if j is not None else None,
            "符号一致率": float(consistency[j]) if j is not None and counts[j] else None,
            "多数方向": int(majority_sign[j]) if j is not None else 0,
            "系数绝对值中位数（百分点）": 100 * float(median_strength[j]) if j is not None else None,
            "最终入选": bool(final_selected), "最终系数（百分点）": 100 * float(final.coef_[column]),
            "处理结果": reason,
        })
    summary = {
        "alpha": alpha, "原始候选数": x.shape[1], "有效候选数": len(columns),
        "初始Lasso入选数": int(base_support.sum()),
        "达稳定阈值数": len(passed), "最终入选数": int(np.count_nonzero(final.coef_)),
        "重采样次数": n_resamples, "块长（月）": block_length,
        "入选频率阈值": frequency_cutoff, "符号一致率阈值": sign_cutoff,
        "随机种子": seed, "重试次数": sum(row["尝试次数"] > 1 for row in sampling_logs),
        "训练耗时（秒）": perf_counter() - start,
    }
    return pipeline, {
        "优化汇总": summary, "稳定性记录": stability,
        "参数搜索": tuning["搜索记录"], "调参求解": tuning["求解日志"],
        "重采样求解": sampling_logs, "初始支持集": base_support,
    }

