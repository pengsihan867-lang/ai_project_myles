"""eval/judge.py — LLM-as-a-judge 最小实现 + 位置偏置量化 (M2.5)

================================================================================
这段代码想教会你一件事:  "用模型评模型" 最大的坑是 **位置偏置 (position bias)**
================================================================================

什么叫位置偏置?
  我们让一个 judge 模型去比较两个答案 A 和 B, 问它 "哪个更好"。
  一个不靠谱的 judge, 可能不管内容好坏, 总是偏向排在前面 (A 位置) 的答案。
  也就是说: 同一对答案, 只是把顺序对调一下, 它的结论就变了。

怎么发现和对付这个坑?  ——  swap-and-aggregate (正反各评一次, 取一致):
  第 1 次:  A = 好答案,  B = 坏答案   → 看 judge 选谁
  第 2 次:  A = 坏答案,  B = 好答案   → 把顺序换过来再问一次
  - 如果两次都选了 "同一个内容" (都选好答案)  → 一致, 这个判定可信, 采纳。
  - 如果换个顺序结论就翻了                      → 不一致, 说明判定受位置影响, 丢弃。

我们关心三个数字:
  1. position_flip_rate   位置翻转率: 换顺序后结论翻掉的比例 (越低越好, 这就是偏置大小)
  2. accept_rate          采纳率:     两次一致、可以采纳的比例 (= 1 - 翻转率)
  3. accuracy_on_accepted 采纳样本准确率: 在采纳的判定里, judge 是否真的选了更好的答案

本文件**完全可以离线跑**, 不需要任何 API key (默认用 mock_judge 模拟一个 judge)。
跑法:   python eval/judge.py
产出:   终端打印对比表 + 生成 eval/judge_report.md
"""

from __future__ import annotations

import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # "Route Alpha" 目录


# ============================================================================ #
# 第 1 步: 准备测试数据 —— 几对 (问题, 好答案, 坏答案)
# ----------------------------------------------------------------------------
# 因为我们"人为知道"哪个答案更好, 后面就能拿它当标准答案, 检验 judge 判得对不对。
# 好答案 = 紧扣问题、信息正确;  坏答案 = 跑题 / 空洞 / 错误。
# ============================================================================ #
DEMO_PAIRS = [
    {
        "prompt": "用一句话解释什么是过拟合 overfitting",
        "good": "过拟合是指模型把训练数据里的噪声也学了进去, 导致在训练集上表现很好, 但在新数据上泛化变差。",
        "bad": "过拟合是一种很常见的现象, 大家都听说过, 它和机器学习有关系。",
    },
    {
        "prompt": "Python 里 list 和 tuple 最主要的区别是什么",
        "good": "list 是可变的 (创建后能增删改元素), tuple 是不可变的 (创建后不能修改), 所以 tuple 更适合做固定数据和字典的 key。",
        "bad": "list 和 tuple 都是 Python 的数据类型, 用起来都差不多, 看个人习惯。",
    },
    {
        "prompt": "AUC 这个指标衡量的是分类器的什么能力",
        "good": "AUC 衡量的是排序能力: 随机取一个正样本和一个负样本, 模型给正样本更高分的概率, 0.5 等于瞎猜, 1.0 是完美排序。",
        "bad": "AUC 就是准确率的另一种说法, 数值越高说明模型预测得越准。",  # 这是错的, 故意当坏答案
    },
    {
        "prompt": "为什么要对预测概率做校准 calibration",
        "good": "因为模型输出的概率可能系统性偏高或偏低, 校准让 '预测 70%' 真的对应约 70% 的发生率, 这样概率才能拿来做决策。",
        "bad": "校准就是让模型更准一点的一个步骤, 做了总比不做好。",
    },
    {
        "prompt": "一句话说明 MILP (混合整数线性规划) 适合解决什么问题",
        "good": "MILP 适合在一堆线性约束 (比如总预算上限) 下, 对 '选或不选' 这类 0/1 决策求全局最优解。",
        "bad": "MILP 是一种数学方法, 可以用来算很多优化的东西, 非常强大。",
    },
    {
        "prompt": "解释一下什么是数据穿越 data leakage",
        "good": "数据穿越是指训练阶段不小心用到了本不该看到的未来/测试信息, 导致离线指标虚高, 上线后表现暴跌。",
        "bad": "数据穿越就是数据从一个地方传到另一个地方, 在工程里很常见。",  # 跑题
    },
]


# ============================================================================ #
# 第 2 步: judge 后端 —— 一个能"看两个答案、选一个更好"的函数
# ----------------------------------------------------------------------------
# 我们提供两种:
#   (A) make_mock_judge: 离线模拟, 不用联网。它用一个"质量代理"打分, 再叠加位置偏置。
#   (B) api_judge:       可选, 真用一个大模型 API 当 judge (后面想试再用, 默认不调用)。
# judge 函数统一签名:  judge(prompt, answer_a, answer_b) -> "A" 或 "B"
# ============================================================================ #
def _quality_proxy(prompt: str, answer: str) -> float:
    """一个粗糙的"答案质量"估计, 让 mock judge 有东西可依据。

    思路: 好答案通常更长、信息更密(我们造数据时好答案确实更充实)。
    这只是为了让模拟 judge 行为像那么回事, 不是真的 NLP 评分。
    """
    # 答案字符数(去掉空格)作为信息量的粗略代理; 封顶防止单纯堆字
    char_count = len(answer.replace(" ", ""))
    return min(char_count, 80) * 0.1


def make_mock_judge(position_bias: float = 0.0, noise: float = 0.5, seed: int = 0):
    """造一个模拟 judge。

    参数:
      position_bias: 对 A 位置的偏好强度。0 = 公平; 越大 = 越偏向排在前面的答案。
      noise:         随机扰动, 模拟 judge 不是每次都理性。
      seed:          固定随机种子, 保证每次跑结果可复现。
    返回: 一个 judge(prompt, answer_a, answer_b) 函数。
    """
    rng = random.Random(seed)

    def judge(prompt: str, answer_a: str, answer_b: str) -> str:
        score_a = _quality_proxy(prompt, answer_a) + rng.gauss(0, noise)
        score_b = _quality_proxy(prompt, answer_b) + rng.gauss(0, noise)
        # 关键: 给 A 位置无脑加一个 bias —— 这就是"位置偏置"的来源
        score_a += position_bias
        return "A" if score_a >= score_b else "B"

    return judge


def api_judge(prompt: str, answer_a: str, answer_b: str) -> str:
    """可选: 用真实大模型 API 当 judge (默认不被调用, 想试再说)。

    需要联网 + 在环境变量里设好 OPENAI_API_KEY / OPENAI_BASE_URL。
    这里用 OpenAI 兼容接口写法; 失败会抛错, 不影响 mock 流程。
    """
    import os

    from openai import OpenAI  # 延迟导入: 不用 API 时即使没装也不报错

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),  # 可指向 OpenRouter / GLM 等
    )
    system = "你是一个严格的评审。只回答字母 A 或 B, 表示哪个回答更好, 不要解释。"
    user = f"问题:\n{prompt}\n\n回答 A:\n{answer_a}\n\n回答 B:\n{answer_b}\n\n哪个更好? 只答 A 或 B。"
    resp = client.chat.completions.create(
        model=os.environ.get("JUDGE_MODEL", "gpt-4o-mini"),
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0,
    )
    text = resp.choices[0].message.content.strip().upper()
    return "A" if text.startswith("A") else "B"


# ============================================================================ #
# 第 3 步: 对一对答案做 swap-and-aggregate (正反各评一次)
# ============================================================================ #
def judge_one_pair(judge, prompt: str, good: str, bad: str) -> dict:
    """对一对 (好答案, 坏答案) 正反各问一次 judge, 返回这对的判定细节。"""
    # 顺序 1:  A = 好答案,  B = 坏答案
    verdict_1 = judge(prompt, good, bad)
    winner_1 = "good" if verdict_1 == "A" else "bad"  # 翻译成"内容谁赢了"

    # 顺序 2:  A = 坏答案,  B = 好答案  (把位置换过来)
    verdict_2 = judge(prompt, bad, good)
    winner_2 = "good" if verdict_2 == "B" else "bad"  # 注意: 这次好答案在 B 位置

    consistent = winner_1 == winner_2  # 两次结论是否一致
    return {
        "prompt": prompt[:30] + "...",
        "verdict_1": verdict_1,   # 第一次选的字母
        "verdict_2": verdict_2,   # 第二次选的字母
        "winner_1": winner_1,     # 第一次实际上选了好/坏
        "winner_2": winner_2,     # 第二次实际上选了好/坏
        "consistent": consistent, # 是否一致(可采纳)
        # 只有一致时才采纳判定; 采纳结果就是 winner_1
        "accepted_winner": winner_1 if consistent else None,
    }


# ============================================================================ #
# 第 4 步: 跑完所有样本, 汇总成三个指标
# ============================================================================ #
def run_judge_eval(judge, pairs: list[dict]) -> dict:
    """对所有 pair 跑 swap-and-aggregate, 返回 {明细, 指标}。"""
    rows = [judge_one_pair(judge, p["prompt"], p["good"], p["bad"]) for p in pairs]

    n = len(rows)
    n_flipped = sum(1 for r in rows if not r["consistent"])  # 换序后翻了的
    accepted = [r for r in rows if r["consistent"]]
    n_correct = sum(1 for r in accepted if r["accepted_winner"] == "good")

    metrics = {
        "n_pairs": n,
        # 位置翻转率: 换个顺序结论就变的比例 —— 这就是位置偏置的大小, 越低越好
        "position_flip_rate": round(n_flipped / n, 3) if n else 0.0,
        # 采纳率: 两次一致、可信可用的比例
        "accept_rate": round(len(accepted) / n, 3) if n else 0.0,
        # 采纳样本的准确率: 在可信判定里, judge 有没有真的挑出更好的答案
        "accuracy_on_accepted": round(n_correct / len(accepted), 3) if accepted else None,
    }
    return {"rows": rows, "metrics": metrics}


# ============================================================================ #
# 第 5 步: 写一份小报告 (交付物), 顺便对比"公平 judge"和"有偏置 judge"
# ============================================================================ #
def write_report(fair: dict, biased: dict, out_path: Path) -> None:
    lines = []
    lines.append("# LLM-as-a-Judge 偏置诊断报告\n")
    lines.append("> 由 `eval/judge.py` 生成。演示位置偏置如何被 swap-and-aggregate 发现。\n")
    lines.append("## 指标对比\n")
    lines.append("| judge | 位置翻转率 | 采纳率 | 采纳样本准确率 |")
    lines.append("|---|---|---|---|")
    for name, res in [("公平 judge (无偏置)", fair), ("有位置偏置的 judge", biased)]:
        m = res["metrics"]
        lines.append(
            f"| {name} | {m['position_flip_rate']} | {m['accept_rate']} | {m['accuracy_on_accepted']} |"
        )
    lines.append("\n## 怎么读这张表\n")
    lines.append("- **位置翻转率越低越好**: 它衡量 judge 受答案顺序影响的程度。")
    lines.append("- 有偏置的 judge 翻转率明显更高, 说明很多判定只是因为顺序不同而改变, 不可信。")
    lines.append("- **采纳率** = 两次一致、保留下来的比例; 翻转掉的判定被丢弃, 不进入训练标签 (防污染)。")
    lines.append("- **采纳样本准确率**: 在保留的判定里, judge 是否真的挑出了更好的答案。\n")
    lines.append("## 结论\n")
    lines.append("swap-and-aggregate 能把受位置影响的判定识别并剔除, 是 LLM-judge 可信度的基础防线。\n")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    pairs = DEMO_PAIRS

    # 造两个 judge: 一个公平(bias=0), 一个明显偏向 A 位置(bias 很大)
    fair_judge = make_mock_judge(position_bias=0.0, seed=42)
    biased_judge = make_mock_judge(position_bias=5.0, seed=42)

    fair = run_judge_eval(fair_judge, pairs)
    biased = run_judge_eval(biased_judge, pairs)

    # 终端打印
    print("\n===== 公平 judge (无位置偏置) =====")
    print(fair["metrics"])
    print("\n===== 有位置偏置的 judge =====")
    print(biased["metrics"])

    print("\n----- 有偏置 judge 的逐条明细 -----")
    print(f"{'第1次':<6}{'第2次':<6}{'是否一致':<8}问题")
    for r in biased["rows"]:
        flag = "一致" if r["consistent"] else "翻转!"
        print(f"{r['verdict_1']:<7}{r['verdict_2']:<7}{flag:<9}{r['prompt']}")

    # 写报告
    out = ROOT / "eval" / "judge_report.md"
    write_report(fair, biased, out)
    print(f"\n报告已写入: {out}")


if __name__ == "__main__":
    main()
