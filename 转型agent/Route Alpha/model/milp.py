"""RouteAlpha M3: 预算约束下的路由求解 (Gurobi)。

输入: ml_seperate.py 产出的长表 predictions
  列: [sample_id, eval_name, model, y_true, p_success, cost, ...]

核心 MILP (指派型 0-1 规划):
  变量  x[q, m] ∈ {0, 1}                     query q 是否路由到模型 m
  约束1 ∀q  Σ_m x[q, m] = 1                  每条 query 恰好选一个模型
  约束2 Σ_{q,m} cost[q,m] · x[q,m] ≤ Budget  全局预算硬约束
  目标  max Σ_{q,m} p_success[q,m] · x[q,m]   最大化期望成功数

约束/目标都集中在 solve_routing 里, 注释标了"可改"位置, 方便后续:
  - 加风险项 (减 λ·方差)
  - 加每模型容量上限
  - 改成最小成本 s.t. 成功数下限

evaluate_assignment 用 y_true 回测真实成功率(而非预测), 作为诚实评估。
baselines 给 always-cheap / always-expensive / random / oracle 对照。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger


# --------------------------------------------------------------------------- #
# 数据透视: 长表 → [Q, M] 矩阵
# --------------------------------------------------------------------------- #
@dataclass
class RoutingMatrices:
    queries: list[str]                 # 长度 Q
    models: list[str]                  # 长度 M
    p_success: np.ndarray              # [Q, M] 预测成功率
    cost: np.ndarray                   # [Q, M] 成本
    y_true: np.ndarray                 # [Q, M] 真实 0/1 (回测用)


def to_matrices(pred_df: pd.DataFrame, prob_col: str = "p_success") -> RoutingMatrices:
    queries = sorted(pred_df["sample_id"].unique().tolist())
    models = sorted(pred_df["model"].unique().tolist())
    qi = {q: i for i, q in enumerate(queries)}
    mi = {m: j for j, m in enumerate(models)}

    Q, M = len(queries), len(models)
    p = np.zeros((Q, M))
    c = np.full((Q, M), np.nan)
    y = np.zeros((Q, M))
    for row in pred_df.itertuples(index=False):
        i, j = qi[row.sample_id], mi[row.model]
        p[i, j] = getattr(row, prob_col)
        c[i, j] = row.cost
        y[i, j] = row.y_true
    # 成本缺失兜底: 用该模型中位数
    col_med = np.nanmedian(c, axis=0)
    inds = np.where(np.isnan(c))
    c[inds] = np.take(col_med, inds[1])
    c = np.nan_to_num(c, nan=0.0)
    return RoutingMatrices(queries, models, p, c, y)


# --------------------------------------------------------------------------- #
# 求解结果
# --------------------------------------------------------------------------- #
@dataclass
class RoutingResult:
    name: str
    assignment: pd.DataFrame                      # [sample_id, model, p_success, cost, y_true]
    total_cost: float
    expected_success: float                       # Σ p_success (模型自认为)
    realized_success_rate: float                  # 用 y_true 的真实成功率
    status: str = "ok"
    extra: dict = field(default_factory=dict)


def _assignment_frame(mats: RoutingMatrices, choice: np.ndarray) -> pd.DataFrame:
    rows = []
    for i, q in enumerate(mats.queries):
        j = int(choice[i])
        rows.append(
            {
                "sample_id": q,
                "model": mats.models[j],
                "p_success": float(mats.p_success[i, j]),
                "cost": float(mats.cost[i, j]),
                "y_true": int(mats.y_true[i, j]),
            }
        )
    return pd.DataFrame(rows)


def _summarize(name: str, mats: RoutingMatrices, choice: np.ndarray, status: str = "ok") -> RoutingResult:
    asg = _assignment_frame(mats, choice)
    return RoutingResult(
        name=name,
        assignment=asg,
        total_cost=float(asg["cost"].sum()),
        expected_success=float(asg["p_success"].sum()),
        realized_success_rate=float(asg["y_true"].mean()),
        status=status,
    )


# --------------------------------------------------------------------------- #
# MILP 求解 (Gurobi)
# --------------------------------------------------------------------------- #
def solve_routing(
    pred_df: pd.DataFrame,
    budget: float | None = None,
    budget_per_query: float | None = None,
    prob_col: str = "p_success",
    time_limit: float = 30.0,
    verbose: bool = False,
) -> RoutingResult:
    """在全局预算约束下最大化期望成功数。

    budget 优先; 否则 budget = budget_per_query * Q。
    """
    import gurobipy as gp
    from gurobipy import GRB

    mats = to_matrices(pred_df, prob_col=prob_col)
    Q, M = mats.p_success.shape
    if budget is None:
        if budget_per_query is None:
            raise ValueError("需提供 budget 或 budget_per_query")
        budget = float(budget_per_query) * Q
    logger.info(f"MILP: Q={Q} query, M={M} 模型, 预算={budget:.5f}")

    try:
        model = gp.Model("routealpha_routing")
        model.Params.OutputFlag = 1 if verbose else 0
        model.Params.TimeLimit = time_limit

        # 变量 x[q, m]
        x = model.addVars(Q, M, vtype=GRB.BINARY, name="x")

        # 约束1: 每条 query 恰好选一个模型  (可改: <=1 允许弃答)
        for i in range(Q):
            model.addConstr(gp.quicksum(x[i, j] for j in range(M)) == 1, name=f"one_{i}")

        # 约束2: 全局预算硬约束  (可改: 加每模型容量 / 延迟约束)
        model.addConstr(
            gp.quicksum(mats.cost[i, j] * x[i, j] for i in range(Q) for j in range(M)) <= budget,
            name="budget",
        )

        # 目标: 最大化期望成功数  (可改: 减 λ·风险项)
        model.setObjective(
            gp.quicksum(mats.p_success[i, j] * x[i, j] for i in range(Q) for j in range(M)),
            GRB.MAXIMIZE,
        )

        model.optimize()

        if model.SolCount == 0:
            logger.warning(f"MILP 无可行解 (status={model.Status}); 预算可能过紧")
            # 兜底: 每条选最便宜
            choice = mats.cost.argmin(axis=1)
            res = _summarize("milp(infeasible→cheapest)", mats, choice, status="infeasible")
            return res

        choice = np.array(
            [int(np.argmax([x[i, j].X for j in range(M)])) for i in range(Q)]
        )
        status = "optimal" if model.Status == GRB.OPTIMAL else f"gurobi_status_{model.Status}"
        res = _summarize("milp", mats, choice, status=status)
        res.extra = {"budget": budget, "mip_gap": getattr(model, "MIPGap", float("nan"))}
        logger.info(
            f"MILP {status}: 期望成功 {res.expected_success:.1f}, "
            f"真实成功率 {res.realized_success_rate:.3f}, 花费 {res.total_cost:.5f}/{budget:.5f}"
        )
        return res

    except gp.GurobiError as e:  # license / solver 问题
        logger.error(f"Gurobi 失败({e}); 回退贪心(按 p/cost 性价比, 受预算约束)")
        return _greedy_fallback(mats, budget)


def _greedy_fallback(mats: RoutingMatrices, budget: float) -> RoutingResult:
    """无 Gurobi 时的近似: 先给每条选最便宜, 再在预算内逐步升级性价比最高的。"""
    Q, M = mats.p_success.shape
    choice = mats.cost.argmin(axis=1)
    spent = sum(mats.cost[i, choice[i]] for i in range(Q))
    # 候选升级: (收益/额外成本)
    cand = []
    for i in range(Q):
        cur = choice[i]
        for j in range(M):
            dc = mats.cost[i, j] - mats.cost[i, cur]
            dp = mats.p_success[i, j] - mats.p_success[i, cur]
            if dc > 0 and dp > 0:
                cand.append((dp / dc, i, j, dc))
    cand.sort(reverse=True)
    for _, i, j, dc in cand:
        if spent + dc <= budget and mats.p_success[i, j] > mats.p_success[i, choice[i]]:
            spent += mats.cost[i, j] - mats.cost[i, choice[i]]
            choice[i] = j
    return _summarize("greedy_fallback", mats, choice, status="fallback")


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def baselines(pred_df: pd.DataFrame, seed: int = 42, prob_col: str = "p_success") -> dict[str, RoutingResult]:
    mats = to_matrices(pred_df, prob_col=prob_col)
    Q, M = mats.p_success.shape
    rng = np.random.default_rng(seed)

    out: dict[str, RoutingResult] = {}
    # always-cheap: 每条选最便宜模型
    out["always_cheap"] = _summarize("always_cheap", mats, mats.cost.argmin(axis=1))
    # always-expensive: 每条选最贵模型(常是最强)
    out["always_expensive"] = _summarize("always_expensive", mats, mats.cost.argmax(axis=1))
    # random
    out["random"] = _summarize("random", mats, rng.integers(0, M, size=Q))
    # oracle: 若有模型成功则选其中最便宜的成功模型, 否则选最便宜
    oracle = np.empty(Q, dtype=int)
    for i in range(Q):
        succ = np.where(mats.y_true[i] > 0)[0]
        oracle[i] = succ[mats.cost[i, succ].argmin()] if len(succ) else int(mats.cost[i].argmin())
    out["oracle"] = _summarize("oracle", mats, oracle)
    return out


def compare(
    pred_df: pd.DataFrame,
    budget_per_query: float,
    prob_col: str = "p_success",
    seed: int = 42,
) -> pd.DataFrame:
    """MILP + baselines 汇总成一张对比表。"""
    rows = []
    bs = baselines(pred_df, seed=seed, prob_col=prob_col)
    for r in bs.values():
        rows.append(
            {
                "策略": r.name,
                "真实成功率": round(r.realized_success_rate, 4),
                "期望成功率": round(r.expected_success / len(r.assignment), 4),
                "总成本": round(r.total_cost, 5),
                "状态": r.status,
            }
        )
    milp = solve_routing(pred_df, budget_per_query=budget_per_query, prob_col=prob_col)
    rows.append(
        {
            "策略": f"milp(预算={budget_per_query}/条)",
            "真实成功率": round(milp.realized_success_rate, 4),
            "期望成功率": round(milp.expected_success / len(milp.assignment), 4),
            "总成本": round(milp.total_cost, 5),
            "状态": milp.status,
        }
    )
    return pd.DataFrame(rows).sort_values("真实成功率", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 阶段二深化: Pareto / optimality gap / 降级失败率
# --------------------------------------------------------------------------- #
def downgrade_failure_rate(mats: RoutingMatrices, choice: np.ndarray) -> float:
    """路由失败但存在其他 model 可成功的「可避免失败」占比。"""
    failures = 0
    avoidable = 0
    for i in range(len(mats.queries)):
        j = int(choice[i])
        if mats.y_true[i, j] < 1:
            failures += 1
            if np.any(mats.y_true[i] > 0):
                avoidable += 1
    return avoidable / failures if failures else 0.0


def optimality_gap(milp_result: RoutingResult, oracle_result: RoutingResult) -> float:
    """oracle 真实成功率 − MILP 真实成功率（越小越好）。"""
    return oracle_result.realized_success_rate - milp_result.realized_success_rate


def pareto_sweep(
    pred_df: pd.DataFrame,
    budget_per_query_grid: list[float] | None = None,
    prob_col: str = "p_success",
) -> pd.DataFrame:
    """扫预算网格, 返回 Pareto 曲线数据。"""
    Q = pred_df["sample_id"].nunique()
    if budget_per_query_grid is None:
        budget_per_query_grid = [0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.006, 0.01]
    rows = []
    oracle = baselines(pred_df, prob_col=prob_col)["oracle"]
    for bpq in budget_per_query_grid:
        milp = solve_routing(pred_df, budget_per_query=bpq, prob_col=prob_col)
        mats = to_matrices(pred_df, prob_col=prob_col)
        choice = np.array([mats.models.index(m) for m in milp.assignment["model"]])
        rows.append(
            {
                "budget_per_query": bpq,
                "total_budget": bpq * Q,
                "total_cost": milp.total_cost,
                "realized_success_rate": milp.realized_success_rate,
                "expected_success_rate": milp.expected_success / len(milp.assignment),
                "optimality_gap": optimality_gap(milp, oracle),
                "downgrade_failure_rate": downgrade_failure_rate(mats, choice),
                "status": milp.status,
                "prob_col": prob_col,
            }
        )
    return pd.DataFrame(rows)


def calibration_routing_ablation(
    pred_df: pd.DataFrame,
    budget_per_query: float,
) -> pd.DataFrame:
    """raw vs calibrated 概率的路由对比 + gap + 降级失败率。"""
    rows = []
    for col, label in [("p_success_raw", "raw"), ("p_success_cal", "calibrated")]:
        if col not in pred_df.columns:
            continue
        milp = solve_routing(pred_df, budget_per_query=budget_per_query, prob_col=col)
        oracle = baselines(pred_df, prob_col=col)["oracle"]
        mats = to_matrices(pred_df, prob_col=col)
        choice = np.array([mats.models.index(m) for m in milp.assignment["model"]])
        rows.append(
            {
                "概率来源": label,
                "真实成功率": round(milp.realized_success_rate, 4),
                "总成本": round(milp.total_cost, 5),
                "optimality_gap": round(optimality_gap(milp, oracle), 4),
                "downgrade_failure_rate": round(downgrade_failure_rate(mats, choice), 4),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/predictions.parquet"
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    pred = pd.read_parquet(root / path)
    table = compare(pred, budget_per_query=0.002)
    print(table.to_string(index=False))
