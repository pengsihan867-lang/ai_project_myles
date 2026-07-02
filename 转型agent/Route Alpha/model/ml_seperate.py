"""RouteAlpha M1/M2: 配置驱动的能力预测管线。

流程:
  1. 读取 config.yaml + peek.csv
  2. 每 model 独立 XGBoost: prompt → P(success)
  3. 特征: bge-small (frozen, 无穿越) 或 TF-IDF (按折仅在 train 上 fit)
  4. 扩张窗口 out-of-fold 回测 + 可选 Platt/Isotonic 校准
  5. 指标: accuracy / AUC / Brier / ECE

CLI:  python model/ml_seperate.py [config/config.yaml]
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import yaml
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
COST_SUFFIX = "|total_cost"


# --------------------------------------------------------------------------- #
# 配置 / 数据
# --------------------------------------------------------------------------- #
def load_config(path: str | Path = "config/config.yaml") -> dict:
    cfg_path = (ROOT / path) if not Path(path).is_absolute() else Path(path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"已加载配置: {cfg_path}")
    return cfg


@dataclass
class Dataset:
    prompts: list[str]
    sample_ids: list[str]
    eval_names: list[str]
    models: list[str]
    success: np.ndarray
    cost: np.ndarray
    frame: pd.DataFrame


def _resolve(path: str | Path) -> Path:
    return (ROOT / path) if not Path(path).is_absolute() else Path(path)


def load_data(cfg: dict) -> Dataset:
    csv_path = _resolve(cfg["data"]["csv_path"])
    max_samples = cfg["data"].get("max_samples")
    df = pd.read_csv(csv_path)
    if cfg["data"].get("shuffle"):
        df = df.sample(frac=1.0, random_state=int(cfg["data"].get("seed", 42))).reset_index(drop=True)
    if max_samples:
        df = df.head(int(max_samples)).reset_index(drop=True)

    models = list(cfg["models"])
    missing = [m for m in models if m not in df.columns]
    if missing:
        raise ValueError(f"以下选定模型在 CSV 里没有成功列: {missing}")

    thr = float(cfg["data"].get("success_threshold", 0.5))
    success = (df[models].astype(float).fillna(0.0).values >= thr).astype(int)

    cost_cols = {}
    for m in models:
        col = f"{m}{COST_SUFFIX}"
        if col in df.columns:
            cost_cols[m] = pd.to_numeric(df[col], errors="coerce")
        else:
            logger.warning(f"模型 {m} 缺少成本列 {col}, 用 0 兜底")
            cost_cols[m] = pd.Series(np.zeros(len(df)))
    cost_df = pd.DataFrame(cost_cols)
    cost_df = cost_df.fillna(cost_df.median(numeric_only=True)).fillna(0.0)
    cost = cost_df[models].values

    logger.info(
        f"数据: {len(df)} 条, 模型 {len(models)} 个; "
        f"平均成功率 {success.mean():.3f}, 平均成本 {cost.mean():.5f}"
    )
    return Dataset(
        prompts=df["prompt"].astype(str).tolist(),
        sample_ids=df["sample_id"].astype(str).tolist(),
        eval_names=df["eval_name"].astype(str).tolist(),
        models=models,
        success=success,
        cost=cost,
        frame=df,
    )


# --------------------------------------------------------------------------- #
# 可解释手工特征（逐行确定性，不 fit）
# --------------------------------------------------------------------------- #
_NUMERIC_EVALS = frozenset(
    {
        "grade-school-math",
        "mtbench-math",
        "gsm8k",
        "mmlu-elementary-mathematics",
        "mmlu-high-school-mathematics",
        "mmlu-college-mathematics",
    }
)
_CODE_EVALS = frozenset({"mbpp"})
_MC_EVALS = frozenset({"hellaswag", "winogrande", "arc-challenge"})
_OPEN_EVALS = frozenset(
    {"consensus_summary", "bias_detection", "abstract2title", "chinese_hard_translations"}
)

TASK_FAMILY_LABELS: list[tuple[str, str]] = [
    ("numeric_reasoning", "任务族_数值推理"),
    ("code", "任务族_代码"),
    ("multiple_choice", "任务族_选择题"),
    ("open_generation", "任务族_开放生成"),
    ("chinese_cultural", "任务族_中文文化"),
    ("other", "任务族_其他"),
]
TASK_FAMILY_KEYS = [k for k, _ in TASK_FAMILY_LABELS]


def _task_family_key(eval_name: str) -> str:
    """确定性任务族映射（复用 golden_set 思路，不 fit）。"""
    en_lower = eval_name.lower()
    if en_lower.startswith("chinese") or "chinese_" in en_lower or eval_name.startswith("Chinese"):
        return "chinese_cultural"
    if eval_name in _NUMERIC_EVALS:
        return "numeric_reasoning"
    if eval_name in _CODE_EVALS:
        return "code"
    if eval_name.startswith("mmlu-") or eval_name in _MC_EVALS:
        return "multiple_choice"
    if eval_name in _OPEN_EVALS:
        return "open_generation"
    return "other"


def _count_few_shot_examples(text: str) -> int:
    """启发式统计 few-shot 示例条数。"""
    patterns = [
        r"(?i)\bexample\s*\d*\s*:",
        r"(?i)\bfew-?shot\b",
        r"(?m)^\s*(?:Q|Question)\s*\d*\s*[:：]",
        r"(?m)^\s*\d+\.\s*(?:Q|Question)\b",
        r"(?m)^\s*输入\s*\d*\s*[:：]",
        r"(?m)^\s*示例\s*\d*\s*[:：]",
    ]
    return sum(len(re.findall(p, text)) for p in patterns)


def _count_abcd_options(text: str) -> int:
    """统计 A/B/C/D 选项标记出现次数。"""
    return len(re.findall(r"(?m)^\s*[ABCD][\.\)、:：]", text))


def _has_code_signal(text: str) -> int:
    signals = [
        r"\bdef\s+\w+\s*\(",
        r"\bimport\s+\w+",
        r"```",
        r"\{[\s\S]*\}",
        r"\bclass\s+\w+",
        r"\breturn\s+",
    ]
    return int(any(re.search(p, text) for p in signals))


def build_structural_features(
    prompts: list[str],
    eval_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """逐行确定性手工特征，中文列名；含任务族 one-hot。"""
    if len(prompts) != len(eval_names):
        raise ValueError("prompts 与 eval_names 长度必须一致")

    rows: list[list[float]] = []
    for prompt, eval_name in zip(prompts, eval_names):
        text = str(prompt)
        words = re.findall(r"\b\w+\b", text)
        n_chars = len(text)
        n_words = len(words)
        avg_word_len = (sum(len(w) for w in words) / n_words) if n_words else 0.0
        n_newlines = text.count("\n")
        n_digits = sum(ch.isdigit() for ch in text)
        n_questions = text.count("?") + text.count("？")
        alpha = [ch for ch in text if ch.isalpha()]
        upper_ratio = (sum(ch.isupper() for ch in alpha) / len(alpha)) if alpha else 0.0
        n_cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
        cjk_ratio = n_cjk / n_chars if n_chars else 0.0
        is_chinese = int(cjk_ratio >= 0.05 or bool(re.search(r"[\u4e00-\u9fff]", text)))

        has_code = float(_has_code_signal(text))
        n_options = float(_count_abcd_options(text))
        n_fewshot = float(_count_few_shot_examples(text))

        family = _task_family_key(eval_name)
        family_onehot = [1.0 if family == k else 0.0 for k in TASK_FAMILY_KEYS]
        is_numeric = 1.0 if family == "numeric_reasoning" else 0.0
        is_code = 1.0 if family == "code" else 0.0
        is_mc = 1.0 if family == "multiple_choice" else 0.0
        is_zh = 1.0 if family == "chinese_cultural" else 0.0

        # 交互特征：树模型对单特征平方/log 无感（阈值分裂不变），真正有增益的是「特征 × 特征」。
        # 这里造一组「数值特征 × 任务族」的领域交互，可解释、有业务含义。
        interactions = [
            float(n_digits) * is_numeric,   # 数字个数 × 数值推理：数学题里数字越多越难
            float(n_chars) * is_numeric,    # 字符数 × 数值推理：长应用题 vs 短算式
            n_options * is_mc,              # 选项个数 × 选择题：多选 vs 单选形态
            has_code * is_code,             # 是否含代码 × 代码任务：代码信号强化
            float(cjk_ratio) * is_zh,       # CJK比例 × 中文文化：中文任务专属
            float(n_chars) * float(n_newlines),  # 字符数 × 换行数：长且多段（few-shot/复杂指令）
        ]

        row = [
            float(n_chars),
            float(n_words),
            float(avg_word_len),
            float(n_newlines),
            float(n_digits),
            float(n_questions),
            float(upper_ratio),
            has_code,
            n_options,
            n_fewshot,
            float(cjk_ratio),
            float(is_chinese),
            *family_onehot,
            *interactions,
        ]
        rows.append(row)

    struct_names = [
        "字符数",
        "词数",
        "平均词长",
        "换行数",
        "数字个数",
        "问号个数",
        "大写字母比例",
        "是否含代码",
        "选项个数(ABCD)",
        "few_shot示例数",
        "CJK字符比例",
        "是否中文",
        *[label for _, label in TASK_FAMILY_LABELS],
        "数字个数x数值推理",
        "字符数x数值推理",
        "选项数x选择题",
        "代码信号x代码",
        "CJK比例x中文文化",
        "字符数x换行数",
    ]
    return np.asarray(rows, dtype=np.float32), struct_names


# --------------------------------------------------------------------------- #
# 特征: bge-small (frozen) / TF-IDF (按折 fit, 无穿越)
# --------------------------------------------------------------------------- #
class Featurizer:
    """prompt → 稠密特征。frozen 后端可全量缓存; fittable 后端每折只在 train 上 fit。"""

    def __init__(
        self,
        cfg: dict,
        prompts: list[str] | None = None,
        eval_names: list[str] | None = None,
    ):
        ec = cfg["embedding"]
        feat_cfg = cfg.get("features", {})
        self.use_structural = bool(feat_cfg.get("use_structural", True))
        self.use_embedding = bool(feat_cfg.get("use_embedding", True))
        if not self.use_structural and not self.use_embedding:
            raise ValueError("features 至少开启 use_structural 或 use_embedding 之一")

        self.backend = ec.get("backend", "auto")
        self.model_name = ec.get("model_name", "BAAI/bge-small-en-v1.5")
        self.batch_size = int(ec.get("batch_size", 64))
        self.tfidf_max_features = int(ec.get("tfidf_max_features", 512))
        self.cache_path = _resolve(ec["cache_path"]) if ec.get("cache_path") else None
        self.used_backend_: str | None = None
        self.is_fittable_: bool = False
        self._X_full: np.ndarray | None = None
        self._X_struct_full: np.ndarray | None = None
        self._struct_names: list[str] = []
        self.feature_names_: list[str] = []
        self._st_model = None
        self._prompts = prompts
        self._eval_names = eval_names

        if self.use_structural:
            if prompts is None or eval_names is None:
                raise ValueError("use_structural=True 时须在构造 Featurizer 时传入 prompts 与 eval_names")
            self._X_struct_full, self._struct_names = build_structural_features(prompts, eval_names)
            logger.info(f"结构特征: {self._X_struct_full.shape[1]} 维 (中文可解释)")

    @property
    def is_fittable(self) -> bool:
        return self.is_fittable_

    def _cache_key(self, prompts: list[str]) -> str:
        h = hashlib.sha1()
        h.update(f"{self.backend}|{self.model_name}|{len(prompts)}".encode())
        h.update("".join(prompts[:50]).encode("utf-8", "ignore"))
        return h.hexdigest()[:16]

    def _cache_acceptable(self, cached_backend: str) -> bool:
        if self.backend == "tfidf":
            return cached_backend == "tfidf"
        if self.backend in ("auto", "sentence_transformers"):
            return cached_backend.startswith("sentence_transformers:")
        return True

    def _ensure_backend(self, prompts: list[str]) -> None:
        """探测并固定 backend（首次调用）。"""
        if self.used_backend_ is not None:
            return
        if self.backend == "tfidf":
            self.used_backend_ = "tfidf"
            self.is_fittable_ = True
            return
        if self.backend in ("auto", "sentence_transformers"):
            try:
                os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
                from sentence_transformers import SentenceTransformer

                logger.info(f"用 sentence-transformers 编码: {self.model_name}")
                self._st_model = SentenceTransformer(self.model_name)
                self.used_backend_ = f"sentence_transformers:{self.model_name}"
                self.is_fittable_ = False
                return
            except Exception as e:  # noqa: BLE001
                if self.backend == "sentence_transformers":
                    raise
                logger.warning(f"bge-small 不可用({e}); 回退 TF-IDF")
        self.used_backend_ = "tfidf"
        self.is_fittable_ = True

    def encode_frozen(self, prompts: list[str]) -> np.ndarray:
        """frozen 后端: 全量编码一次（可缓存），按折切片无穿越。"""
        self._ensure_backend(prompts)
        if self.is_fittable_:
            raise RuntimeError("encode_frozen 仅用于 frozen 后端")

        if self.cache_path and self.cache_path.exists():
            try:
                cached = np.load(self.cache_path, allow_pickle=True)
                if (
                    str(cached.get("key")) == self._cache_key(prompts)
                    and self._cache_acceptable(str(cached.get("backend", "")))
                ):
                    self._X_full = cached["X"]
                    logger.info(f"特征缓存命中: {self.cache_path}")
                    return self._X_full
            except Exception as e:  # noqa: BLE001
                logger.warning(f"读取缓存失败: {e}")

        X = self._st_model.encode(
            prompts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        X = np.asarray(X, dtype=np.float32)
        self._X_full = X
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                self.cache_path, X=X, key=self._cache_key(prompts), backend=self.used_backend_
            )
            logger.info(f"特征已缓存: {self.cache_path}")
        return X

    def fit_transform_fold(
        self,
        prompts: list[str],
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """按折取特征。frozen→切片; fittable→仅 train fit；可选拼接结构特征。"""
        X_emb_train: np.ndarray | None = None
        X_emb_test: np.ndarray | None = None

        if self.use_embedding:
            self._ensure_backend(prompts)
            if not self.is_fittable_:
                if self._X_full is None:
                    self.encode_frozen(prompts)
                X_emb_train = self._X_full[train_idx]
                X_emb_test = self._X_full[test_idx]
            else:
                from sklearn.feature_extraction.text import TfidfVectorizer

                train_p = [prompts[i] for i in train_idx]
                test_p = [prompts[i] for i in test_idx]
                vec = TfidfVectorizer(
                    max_features=self.tfidf_max_features,
                    ngram_range=(1, 2),
                    analyzer="char_wb",
                )
                X_emb_train = vec.fit_transform(train_p).toarray().astype(np.float32)
                X_emb_test = vec.transform(test_p).toarray().astype(np.float32)

        if not self.use_structural:
            if X_emb_train is None or X_emb_test is None:
                raise ValueError("未启用结构特征时须启用 embedding")
            self.feature_names_ = self._embedding_feature_names(X_emb_train.shape[1])
            return X_emb_train, X_emb_test

        return self._concat_features(X_emb_train, X_emb_test, train_idx, test_idx)

    def transform(self, prompts: list[str]) -> np.ndarray:
        """兼容旧接口: frozen 全量编码; tfidf 全量 fit（仅用于非回测场景）。"""
        self._ensure_backend(prompts)
        if not self.is_fittable_:
            return self.encode_frozen(prompts)
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer(
            max_features=self.tfidf_max_features,
            ngram_range=(1, 2),
            analyzer="char_wb",
        )
        return vec.fit_transform(prompts).toarray().astype(np.float32)

    def _embedding_feature_names(self, n_dims: int) -> list[str]:
        return [f"emb_{i:03d}" for i in range(n_dims)]

    def _concat_features(
        self,
        X_emb_train: np.ndarray | None,
        X_emb_test: np.ndarray | None,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        parts_train: list[np.ndarray] = []
        parts_test: list[np.ndarray] = []
        names: list[str] = []

        if self.use_embedding:
            if X_emb_train is None or X_emb_test is None:
                raise ValueError("use_embedding=True 但未提供语义特征")
            parts_train.append(X_emb_train)
            parts_test.append(X_emb_test)
            names.extend(self._embedding_feature_names(X_emb_train.shape[1]))

        if self.use_structural:
            if self._X_struct_full is None:
                raise ValueError("use_structural=True 但结构特征未初始化")
            parts_train.append(self._X_struct_full[train_idx])
            parts_test.append(self._X_struct_full[test_idx])
            names.extend(self._struct_names)

        self.feature_names_ = names
        if not parts_train:
            raise ValueError("无可用特征块")
        return np.hstack(parts_train), np.hstack(parts_test)

    def build_full_matrix(self, prompts: list[str]) -> np.ndarray:
        """构建全量特征矩阵（诊断/展示用；fittable 后端会在全量上 fit）。"""
        emb: np.ndarray | None = None
        if self.use_embedding:
            emb = self.transform(prompts)
        n_emb = emb.shape[1] if emb is not None else 0
        n_struct = self._X_struct_full.shape[1] if self.use_structural else 0
        if emb is None and self._X_struct_full is None:
            raise ValueError("无可用特征")
        if emb is not None and self.use_structural:
            X = np.hstack([emb, self._X_struct_full])
            self.feature_names_ = self._embedding_feature_names(n_emb) + self._struct_names
        elif emb is not None:
            X = emb
            self.feature_names_ = self._embedding_feature_names(n_emb)
        else:
            X = self._X_struct_full
            self.feature_names_ = list(self._struct_names)
        return X


# --------------------------------------------------------------------------- #
# 校准
# --------------------------------------------------------------------------- #
def _fit_calibrator(method: str, y_cal: np.ndarray, p_cal: np.ndarray):
    if method == "none" or len(np.unique(y_cal)) < 2:
        return None
    if method == "sigmoid":
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(max_iter=1000)
        lr.fit(p_cal.reshape(-1, 1), y_cal)
        return ("sigmoid", lr)
    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_cal, y_cal)
        return ("isotonic", iso)
    raise ValueError(f"未知校准方法: {method}")


def _apply_calibrator(calibrator, p: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return p
    kind, model = calibrator
    if kind == "sigmoid":
        return model.predict_proba(p.reshape(-1, 1))[:, 1]
    return model.predict(p)


# --------------------------------------------------------------------------- #
# 滚动切分
# --------------------------------------------------------------------------- #
def make_folds(n: int, cfg: dict) -> Iterator[tuple[np.ndarray, np.ndarray, int]]:
    sp = cfg["split"]
    mode = sp.get("mode", "expanding")
    init = int(sp.get("initial_train", 100))
    step = int(sp.get("step", 100))

    if mode == "single":
        test_size = int(sp.get("test_size", step))
        train_idx = np.arange(0, min(init, n))
        test_idx = np.arange(init, min(init + test_size, n))
        if len(test_idx):
            yield train_idx, test_idx, 0
        return

    fold = 0
    start_test = init
    while start_test < n:
        end_test = min(start_test + step, n)
        test_idx = np.arange(start_test, end_test)
        if mode == "sliding":
            train_idx = np.arange(max(0, start_test - init), start_test)
        else:
            train_idx = np.arange(0, start_test)
        if len(train_idx) and len(test_idx):
            yield train_idx, test_idx, fold
        fold += 1
        start_test = end_test


def _split_train_calib(train_idx: np.ndarray, holdout_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(train_idx)
    if n < 4 or holdout_frac <= 0:
        return train_idx, np.array([], dtype=int)
    n_cal = max(1, int(n * holdout_frac))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cal_local = perm[:n_cal]
    proper_local = perm[n_cal:]
    if len(proper_local) == 0:
        proper_local = perm[n_cal - 1 :]
        cal_local = perm[: n_cal - 1]
    return train_idx[proper_local], train_idx[cal_local]


# --------------------------------------------------------------------------- #
# 滚动回测
# --------------------------------------------------------------------------- #
def _new_predictor(cfg: dict):
    from xgboost import XGBClassifier

    return XGBClassifier(**dict(cfg["predictor"]["params"]))


def _predict_proba(clf, X: np.ndarray, y_fallback: np.ndarray) -> np.ndarray:
    if len(np.unique(y_fallback)) < 2:
        return np.full(len(X), float(y_fallback.mean()) if len(y_fallback) else 0.0)
    return clf.predict_proba(X)[:, 1]


def rolling_backtest(cfg: dict, featurizer: Featurizer, data: Dataset) -> pd.DataFrame:
    """out-of-fold 回测; 特征按折无穿越; 可选校准。"""
    cal_cfg = cfg.get("predictor", {}).get("calibration", {})
    cal_method = cal_cfg.get("method", "isotonic")
    holdout_frac = float(cal_cfg.get("holdout_frac", 0.3))
    seed = int(cfg["data"].get("seed", 42))

    rows: list[dict] = []
    folds = list(make_folds(len(data.prompts), cfg))
    logger.info(
        f"out-of-fold 回测: {len(folds)} fold, backend={featurizer.backend}, "
        f"校准={cal_method}"
    )

    for train_idx, test_idx, fold in folds:
        X_train, X_test = featurizer.fit_transform_fold(data.prompts, train_idx, test_idx)
        train_proper, cal_idx = _split_train_calib(train_idx, holdout_frac, seed + fold)

        if len(cal_idx):
            proper_set = set(train_proper.tolist())
            cal_set = set(cal_idx.tolist())
            proper_mask = np.array([idx in proper_set for idx in train_idx])
            cal_mask = np.array([idx in cal_set for idx in train_idx])
            X_proper = X_train[proper_mask]
            X_cal = X_train[cal_mask]
        else:
            X_proper, X_cal = X_train, X_train[:0]

        for mi, model in enumerate(data.models):
            y_train = data.success[train_idx, mi]
            y_test = data.success[test_idx, mi]
            y_proper = data.success[train_proper, mi] if len(train_proper) else y_train
            y_cal = data.success[cal_idx, mi] if len(cal_idx) else np.array([])

            if len(np.unique(y_proper)) < 2:
                p_raw = np.full(len(test_idx), float(y_proper.mean()) if len(y_proper) else 0.0)
                p_cal = p_raw.copy()
            else:
                clf = _new_predictor(cfg)
                clf.fit(X_proper, y_proper)
                p_raw = clf.predict_proba(X_test)[:, 1]

                if len(cal_idx) and len(np.unique(y_cal)) >= 2:
                    p_cal_train = clf.predict_proba(X_cal)[:, 1]
                    calibrator = _fit_calibrator(cal_method, y_cal, p_cal_train)
                    p_cal = _apply_calibrator(calibrator, p_raw)
                else:
                    p_cal = p_raw.copy()

            for j, ti in enumerate(test_idx):
                rows.append(
                    {
                        "fold": fold,
                        "sample_id": data.sample_ids[ti],
                        "eval_name": data.eval_names[ti],
                        "model": model,
                        "y_true": int(y_test[j]),
                        "p_success_raw": float(p_raw[j]),
                        "p_success_cal": float(p_cal[j]),
                        "p_success": float(p_cal[j]),  # 默认用校准后
                        "cost": float(data.cost[ti, mi]),
                    }
                )

    pred_df = pd.DataFrame(rows)
    logger.info(f"回测产出 {len(pred_df)} 行 ({pred_df['sample_id'].nunique()} query)")
    return pred_df


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(y_prob, quantiles))
    if len(edges) < 2:
        return float("nan")
    ece = 0.0
    n = len(y_true)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_prob >= lo) & (y_prob <= hi if i == len(edges) - 2 else y_prob < hi)
        if not mask.any():
            continue
        ece += mask.sum() / n * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(ece)


def compute_metrics(
    pred_df: pd.DataFrame,
    prob_col: str = "p_success",
    n_bins: int = 10,
) -> pd.DataFrame:
    from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

    def _metrics(g: pd.DataFrame) -> pd.Series:
        y = g["y_true"].values
        p = g[prob_col].values
        acc = accuracy_score(y, (p >= 0.5).astype(int))
        try:
            auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan
        except ValueError:
            auc = np.nan
        brier = brier_score_loss(y, p) if len(np.unique(y)) > 1 else np.nan
        ece = compute_ece(y, p, n_bins=n_bins)
        return pd.Series(
            {"n": len(g), "pos_rate": y.mean(), "accuracy": acc, "auc": auc, "brier": brier, "ece": ece}
        )

    per_model = [
        pd.concat([pd.Series({"model": m, "prob_col": prob_col}), _metrics(g)])
        for m, g in pred_df.groupby("model")
    ]
    overall = pd.concat([pd.Series({"model": "__overall__", "prob_col": prob_col}), _metrics(pred_df)])
    return pd.DataFrame(per_model + [overall]).reset_index(drop=True)


def compare_calibration_metrics(pred_df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """校准前后 ECE/AUC 对比表。"""
    raw = compute_metrics(pred_df, prob_col="p_success_raw", n_bins=n_bins)
    cal = compute_metrics(pred_df, prob_col="p_success_cal", n_bins=n_bins)
    raw["variant"] = "raw"
    cal["variant"] = "calibrated"
    return pd.concat([raw, cal], ignore_index=True)


def feature_importance_report(
    cfg: dict,
    featurizer: Featurizer,
    data: Dataset,
    top_k: int = 25,
) -> pd.DataFrame:
    """诊断用特征重要度（全量 fit XGB，**非回测指标**）。

    结构特征逐项中文名；384 维 embedding 合并为「语义embedding(384维)合计」桶。
    """
    X = featurizer.build_full_matrix(data.prompts)
    names = list(featurizer.feature_names_)
    emb_prefix = "emb_"
    emb_indices = [i for i, n in enumerate(names) if n.startswith(emb_prefix)]

    importances = np.zeros(X.shape[1], dtype=np.float64)
    n_trained = 0
    for mi in range(len(data.models)):
        y = data.success[:, mi]
        if len(np.unique(y)) < 2:
            continue
        clf = _new_predictor(cfg)
        clf.fit(X, y)
        importances += clf.feature_importances_
        n_trained += 1

    if n_trained == 0:
        return pd.DataFrame(columns=["特征", "重要度", "备注"])

    importances /= n_trained
    rows: list[dict] = []

    if emb_indices:
        emb_imp = float(importances[emb_indices].sum())
        rows.append(
            {
                "特征": f"语义embedding({len(emb_indices)}维)合计",
                "重要度": emb_imp,
                "备注": "诊断用聚合桶，非单维解释",
            }
        )

    for i, name in enumerate(names):
        if i in emb_indices:
            continue
        rows.append({"特征": name, "重要度": float(importances[i]), "备注": "结构特征"})

    df = pd.DataFrame(rows).sort_values("重要度", ascending=False).reset_index(drop=True)
    df["用途"] = "诊断（全量fit，非回测）"
    return df.head(top_k)


# --------------------------------------------------------------------------- #
# 持久化 / 主流程
# --------------------------------------------------------------------------- #
def save_predictions(cfg: dict, pred_df: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    pred_path = _resolve(cfg["output"]["pred_path"])
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        pred_df.to_parquet(pred_path, index=False)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"写 parquet 失败({e}), 改写 csv")
        pred_path = pred_path.with_suffix(".csv")
        pred_df.to_csv(pred_path, index=False)
    logger.info(f"预测已保存: {pred_path}")

    if cfg["output"].get("metrics_path"):
        mp = _resolve(cfg["output"]["metrics_path"])
        metrics_df.to_csv(mp, index=False)
        logger.info(f"指标已保存: {mp}")


def run_pipeline(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = load_data(cfg)
    featurizer = Featurizer(cfg, prompts=data.prompts, eval_names=data.eval_names)
    if featurizer.use_embedding:
        featurizer._ensure_backend(data.prompts)
        logger.info(f"特征后端: {featurizer.used_backend_} (fittable={featurizer.is_fittable})")
    if featurizer.use_structural:
        logger.info(f"结构特征已拼接: {len(featurizer._struct_names)} 维")
    pred_df = rolling_backtest(cfg, featurizer, data)
    metrics_df = compute_metrics(pred_df, prob_col="p_success_cal")
    return pred_df, metrics_df


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    cfg_path = argv[0] if argv else "config/config.yaml"
    cfg = load_config(cfg_path)
    pred_df, metrics_df = run_pipeline(cfg)
    save_predictions(cfg, pred_df, metrics_df)
    cal_cmp = compare_calibration_metrics(pred_df)
    logger.info("\n校准后指标:\n" + metrics_df.to_string(index=False))
    logger.info("\n校准 ablation:\n" + cal_cmp.groupby(["model", "variant"])[["ece", "auc", "brier"]].first().to_string())


if __name__ == "__main__":
    main()
