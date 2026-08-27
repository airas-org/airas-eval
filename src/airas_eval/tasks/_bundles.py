"""Reusable metric bundles: the building blocks tasks are composed from.

A bundle is "all the standard metrics over one input shape". Bundles are
module constants — not registered, not evaluable on their own — so a caller
can only ever pick a task type, never a subset of it. Adapters that need
task-shape knowledge (binary-only metrics, top-5 needing > 5 classes, input
counts) live here as named module-level functions so they carry a stable
qualname into the task signature.
"""

import numpy as np

from airas_eval.exceptions import NotApplicable, UndefinedMetric
from airas_eval.metrics import classification as _cls
from airas_eval.metrics import pareto as _pareto
from airas_eval.metrics import regression as _reg
from airas_eval.metrics import search as _search
from airas_eval.metrics import selection as _sel
from airas_eval.spec import Bundle, MetricBinding
from airas_eval.tasks._inputs import (
    CandidateRankingInputs,
    ClassificationInputs,
    MultiobjectiveInputs,
    SearchInputs,
)

Probs = list[list[float]]
Labels = list[int]

# --- classification -----------------------------------------------------------


def _binary_positive_scores(
    probabilities: Probs, reference_labels: Labels
) -> tuple[list[float], list[int]]:
    probs = np.asarray(probabilities, dtype=float)
    reference = np.asarray(reference_labels)
    if probs.ndim != 2 or probs.shape[1] != 2:
        raise NotApplicable("binary-only metric: probabilities are not 2-class")
    if set(np.unique(reference).tolist()) != {0, 1}:
        raise UndefinedMetric("reference labels do not contain both classes 0 and 1")
    return probs[:, 1].tolist(), reference.tolist()


def auroc_binary(probabilities: Probs, reference_labels: Labels) -> float:
    return _cls.auroc(*_binary_positive_scores(probabilities, reference_labels))


def average_precision_binary(probabilities: Probs, reference_labels: Labels) -> float:
    return _cls.average_precision(
        *_binary_positive_scores(probabilities, reference_labels)
    )


def brier_score_binary(probabilities: Probs, reference_labels: Labels) -> float:
    return _cls.brier_score(*_binary_positive_scores(probabilities, reference_labels))


def n_examples(predicted_labels: Labels, reference_labels: Labels) -> float:
    if len(predicted_labels) != len(reference_labels):
        raise ValueError(
            f"length mismatch: {len(predicted_labels)} vs {len(reference_labels)}"
        )
    return float(len(reference_labels))


def per_example_correct(
    predicted_labels: Labels, reference_labels: Labels
) -> list[float]:
    predicted = np.asarray(predicted_labels)
    reference = np.asarray(reference_labels)
    if predicted.shape != reference.shape:
        raise ValueError(f"length mismatch: {predicted.shape} vs {reference.shape}")
    return [float(v) for v in (predicted == reference)]


_CORRECT = (
    MetricBinding(
        "correct",
        per_example_correct,
        ("predicted_labels", "reference_labels"),
        description="事例ごとの正誤(1/0)。2 システムのペア比較(compare)に使う。",
    ),
)


def n_classes(probabilities: Probs) -> float:
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2:
        raise ValueError(f"probabilities must be 2-dimensional, got {probs.shape}")
    return float(probs.shape[1])


def top_5_accuracy(probabilities: Probs, reference_labels: Labels) -> float:
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2 or probs.shape[1] <= 5:
        raise NotApplicable("top-5 accuracy requires more than 5 classes")
    return _cls.top_k_accuracy(probabilities, reference_labels, k=5)


_LABELS = ("predicted_labels", "reference_labels")
_PROBS = ("probabilities", "reference_labels")

CLASSIFICATION = Bundle(
    input_model=ClassificationInputs,
    per_example=_CORRECT,
    provenance_packages=("numpy", "scikit-learn"),
    notes=(
        "単一ラベルの多クラス分類。precision/recall/F1 はマクロ平均で zero_division=0"
        "(この設定ではマイクロ平均は accuracy に一致)。ECE は top-1 確信度を等幅 15 "
        "ビンに分けて計算"
    ),
    metrics=(
        MetricBinding(
            "accuracy",
            _cls.accuracy,
            _LABELS,
            description="正解率。予測ラベルが正解ラベルと一致した割合。",
        ),
        MetricBinding(
            "precision_macro",
            _cls.precision,
            _LABELS,
            {"average": "macro"},
            description="適合率(マクロ平均)。クラスごとの適合率を単純平均した値で、少数クラスも等しく重み付けされる。",
        ),
        MetricBinding(
            "recall_macro",
            _cls.recall,
            _LABELS,
            {"average": "macro"},
            description="再現率(マクロ平均)。クラスごとの再現率を単純平均した値。",
        ),
        MetricBinding(
            "f1_macro",
            _cls.f1,
            _LABELS,
            {"average": "macro"},
            description="F1 スコア(マクロ平均)。クラスごとの適合率と再現率の調和平均を単純平均した値。",
        ),
        MetricBinding(
            "balanced_accuracy",
            _cls.balanced_accuracy,
            _LABELS,
            description="均衡正解率。クラスごとの再現率の平均で、クラス不均衡の影響を受けない正解率。",
        ),
        MetricBinding(
            "matthews_corrcoef",
            _cls.matthews_corrcoef,
            _LABELS,
            description="Matthews 相関係数(MCC)。混同行列全体を使う相関係数で、不均衡データでも偶然の一致に惑わされにくい(−1〜1)。",
        ),
        MetricBinding(
            "log_loss",
            _cls.log_loss,
            _PROBS,
            description="対数損失(交差エントロピー)。正解クラスに割り当てた確率の負の対数の平均。確率の質を測り、低いほど良い。",
        ),
        MetricBinding(
            "expected_calibration_error",
            _cls.expected_calibration_error,
            _PROBS,
            {"n_bins": 15},
            description="期待較正誤差(ECE)。予測確信度をビン分けし、各ビンで確信度と実際の正解率のずれを重み付き平均したもの。低いほど確率が較正されている。",
        ),
        MetricBinding(
            "top_5_accuracy",
            top_5_accuracy,
            _PROBS,
            description="Top-5 正解率。確率上位 5 クラスの中に正解が含まれる割合(ImageNet 系の標準指標)。",
        ),
    ),
    summary=(
        MetricBinding(
            "n_examples", n_examples, _LABELS, description="評価に使われた事例数。"
        ),
        MetricBinding(
            "n_classes",
            n_classes,
            ("probabilities",),
            description="確率行列の列数(クラス数)。",
        ),
    ),
)

BINARY_CLASSIFICATION = Bundle(
    input_model=ClassificationInputs,
    per_example=_CORRECT,
    provenance_packages=("numpy", "scikit-learn"),
    notes=(
        "ラベルは {0, 1}。スコア系指標は probabilities[:, 1] を正例確率として読む。"
        "F1 は正例クラスについて計算。ECE は top-1 確信度を等幅 15 ビンに分けて計算"
    ),
    metrics=(
        MetricBinding(
            "accuracy",
            _cls.accuracy,
            _LABELS,
            description="正解率。予測ラベルが正解ラベルと一致した割合。",
        ),
        MetricBinding(
            "precision",
            _cls.precision,
            _LABELS,
            {"average": "binary"},
            description="適合率(正例クラス)。正例と予測したもののうち実際に正例だった割合。",
        ),
        MetricBinding(
            "recall",
            _cls.recall,
            _LABELS,
            {"average": "binary"},
            description="再現率(正例クラス)。実際の正例のうち正例と予測できた割合。",
        ),
        MetricBinding(
            "f1",
            _cls.f1,
            _LABELS,
            {"average": "binary"},
            description="F1 スコア(正例クラス)。適合率と再現率の調和平均。",
        ),
        MetricBinding(
            "balanced_accuracy",
            _cls.balanced_accuracy,
            _LABELS,
            description="均衡正解率。クラスごとの再現率の平均で、クラス不均衡の影響を受けない正解率。",
        ),
        MetricBinding(
            "matthews_corrcoef",
            _cls.matthews_corrcoef,
            _LABELS,
            description="Matthews 相関係数(MCC)。混同行列全体を使う相関係数で、不均衡データでも偶然の一致に惑わされにくい(−1〜1)。",
        ),
        MetricBinding(
            "auroc",
            auroc_binary,
            _PROBS,
            description="ROC 曲線下面積(AUROC)。正例と負例をランダムに選んだとき正例のスコアが高い確率。閾値に依存しない判別力(0.5 が偶然)。",
        ),
        MetricBinding(
            "average_precision",
            average_precision_binary,
            _PROBS,
            description="平均適合率(AP)。適合率–再現率曲線下の面積で、不均衡な正例検出の性能を測る。",
        ),
        MetricBinding(
            "log_loss",
            _cls.log_loss,
            _PROBS,
            description="対数損失(交差エントロピー)。正解クラスに割り当てた確率の負の対数の平均。確率の質を測り、低いほど良い。",
        ),
        MetricBinding(
            "brier_score",
            brier_score_binary,
            _PROBS,
            description="Brier スコア。正例確率と正解(0/1)の二乗誤差の平均。確率予測の精度と較正を同時に測り、低いほど良い。",
        ),
        MetricBinding(
            "expected_calibration_error",
            _cls.expected_calibration_error,
            _PROBS,
            {"n_bins": 15},
            description="期待較正誤差(ECE)。予測確信度をビン分けし、各ビンで確信度と実際の正解率のずれを重み付き平均したもの。低いほど確率が較正されている。",
        ),
    ),
    summary=(
        MetricBinding(
            "n_examples", n_examples, _LABELS, description="評価に使われた事例数。"
        ),
    ),
)

# --- search -------------------------------------------------------------------


def n_evaluations(evaluated_scores: list[float]) -> float:
    return float(len(evaluated_scores))


_SCORES = ("evaluated_scores",)
_WITH_ORACLE = ("evaluated_scores", "oracle_best")

SEARCH = Bundle(
    input_model=SearchInputs,
    provenance_packages=("numpy",),
    notes=(
        "スコアは高いほど良く、評価順に並ぶ。oracle_best は実験設計で固定する。"
        "スコアが oracle_best を超えた場合、リグレットは skip ではなくエラーになる"
    ),
    metrics=(
        MetricBinding(
            "best_score",
            _search.best_score,
            _SCORES,
            description="探索で評価した候補のうち最良のスコア。",
        ),
        MetricBinding(
            "final_regret",
            _search.final_regret,
            _WITH_ORACLE,
            description="最終リグレット。ベンチマークの既知最適値(oracle_best)と探索で見つけた最良スコアの差。0 が最適解到達。",
        ),
        MetricBinding(
            "mean_anytime_regret",
            _search.mean_anytime_regret,
            _WITH_ORACLE,
            description="平均 anytime リグレット。各評価時点での暫定最良(best-so-far)と最適値の差を全評価にわたって平均したもの。良い候補を早く見つけるほど小さい。",
        ),
        MetricBinding(
            "evaluations_to_best",
            _search.evaluations_to_best,
            _SCORES,
            description="最終的な最良スコアに初めて到達した評価回数(1 始まり)。",
        ),
        MetricBinding(
            "mean_evaluated_score",
            _search.mean_evaluated_score,
            _SCORES,
            description="評価した全候補のスコア平均。偶然見つけた最良値ではなく、探索が選んで評価した候補の質を表す。",
        ),
    ),
    curves=(
        MetricBinding(
            "best_so_far",
            _search.best_so_far,
            _SCORES,
            description="評価回数ごとの暫定最良スコアの推移(anytime 曲線)。",
        ),
    ),
    summary=(
        MetricBinding(
            "n_evaluations",
            n_evaluations,
            _SCORES,
            description="探索で評価した候補数(評価予算)。",
        ),
    ),
)

# --- candidate ranking --------------------------------------------------------


def n_candidates(predicted_scores: list[float], reference_scores: list[float]) -> float:
    if len(predicted_scores) != len(reference_scores):
        raise ValueError(
            f"length mismatch: {len(predicted_scores)} vs {len(reference_scores)}"
        )
    return float(len(reference_scores))


_PAIR = ("predicted_scores", "reference_scores")

CANDIDATE_RANKING = Bundle(
    input_model=CandidateRankingInputs,
    provenance_packages=("numpy", "scipy"),
    notes=(
        "スコアは高いほど良い。Kendall は tau-b。上位 k 集合は同点を安定な降順で"
        "解決する。順位は 1 始まり"
    ),
    metrics=(
        MetricBinding(
            "kendall_tau",
            _reg.kendall_tau,
            _PAIR,
            description="Kendall の τ(tau-b)。予測スコアと真のスコアの順位一致度。全候補ペアのうち順序が一致する割合に基づく(−1〜1)。",
        ),
        MetricBinding(
            "spearman_rho",
            _reg.spearman_rho,
            _PAIR,
            description="Spearman の ρ。予測スコアと真のスコアの順位相関係数(−1〜1)。",
        ),
        MetricBinding(
            "precision_at_top_10pct",
            _sel.precision_at_top_fraction,
            _PAIR,
            {"fraction": 0.10},
            description="上位 10% の一致率。予測で上位 10% とされた候補集合と、真の上位 10% の集合の重なりの割合。",
        ),
        MetricBinding(
            "selection_regret_at_1",
            _sel.selection_regret_at_k,
            _PAIR,
            {"k": 1},
            description="選択リグレット@1。予測で 1 位とした候補の真のスコアと、真の最良スコアとの差。予測器を信じて 1 つ選んだときの損失。",
        ),
        MetricBinding(
            "best_true_rank_in_top_10",
            _sel.best_true_rank_in_predicted_top_k,
            _PAIR,
            {"k": 10},
            description="予測上位 10 件の中に含まれる候補の真の順位の最良値(1 始まり)。1 なら真の最良候補が上位 10 件に入っている。",
        ),
    ),
    curves=(
        MetricBinding(
            "selection_regret_curve",
            _sel.selection_regret_curve,
            _PAIR,
            description="k = 1..n の各 k での選択リグレット。固定 k のスカラーを掃引の中で読むための曲線。",
        ),
        MetricBinding(
            "precision_at_top_k_curve",
            _sel.precision_at_top_k_curve,
            _PAIR,
            description="k = 1..n の各 k での上位 k 集合の一致率(|予測上位k ∩ 真の上位k| / k)。",
        ),
    ),
    summary=(
        MetricBinding(
            "n_candidates",
            n_candidates,
            _PAIR,
            description="順位付けの対象となった候補数。",
        ),
    ),
)

# --- multiobjective -----------------------------------------------------------


def n_points(points: list[list[float]]) -> float:
    return float(len(points))


def n_objectives(points: list[list[float]]) -> float:
    widths = {len(row) for row in points}
    if len(widths) != 1:
        raise ValueError("points must all have the same number of objectives")
    return float(widths.pop())


def pareto_front_size(points: list[list[float]]) -> float:
    return float(sum(_pareto.pareto_front_mask(points)))


_POINTS = ("points",)

MULTIOBJECTIVE = Bundle(
    input_model=MultiobjectiveInputs,
    provenance_packages=("numpy",),
    notes=(
        "全目的を最小化する。hypervolume は 2 目的の厳密計算。IGD/GD/spacing は"
        "正規化なしのユークリッド距離(呼び出し前に目的を正規化すること)"
    ),
    metrics=(
        MetricBinding(
            "pareto_front_size",
            pareto_front_size,
            _POINTS,
            description="Pareto フロント(非劣解)上の点の数。",
        ),
        MetricBinding(
            "hypervolume_2d",
            _pareto.hypervolume_2d,
            ("points", "reference_point"),
            description="ハイパーボリューム(2 目的)。Pareto フロントが参照点に対して支配する面積。フロントが良いほど大きい(全目的最小化)。",
        ),
        MetricBinding(
            "igd",
            _pareto.igd,
            ("points", "reference_front"),
            description="逆世代距離(IGD)。既知の参照フロントの各点から得られたフロントへの最近距離の平均。参照フロントをどれだけ広く近くカバーしたかを測り、低いほど良い。",
        ),
        MetricBinding(
            "gd",
            _pareto.gd,
            ("points", "reference_front"),
            description="世代距離(GD)。得られた各点から参照フロントへの最近距離の平均。収束度を測り、低いほど良い。",
        ),
        MetricBinding(
            "spacing",
            _pareto.spacing,
            _POINTS,
            description="Spacing(Schott)。フロント上の点の最近傍距離の標準偏差。0 に近いほど点が均等に分布している。",
        ),
    ),
    curves=(
        MetricBinding(
            "pareto_front",
            _pareto.pareto_front,
            _POINTS,
            description="非劣解の点集合(第 1 目的で昇順)。",
        ),
    ),
    summary=(
        MetricBinding(
            "n_points",
            n_points,
            _POINTS,
            description="評価対象の候補(目的ベクトル)の数。",
        ),
        MetricBinding(
            "n_objectives", n_objectives, _POINTS, description="目的関数の数。"
        ),
    ),
)
