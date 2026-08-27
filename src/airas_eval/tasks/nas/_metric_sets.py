"""NAS metric sets = the core set's bindings + NAS-specific ones.

The core part is *the same tuple object* as the generic set uses, so a
``nas_pre_training`` report contains everything a ``search`` report does, plus
what the NAS literature adds: the estimated-wall-clock axis (NAS-Bench-101 /
201), position in the search space and a closed-form random-search baseline
at equal budget (Lindauer & Hutter 2020; Yang et al. 2020), validation vs
test regret, and top-10 % rank correlation for predictors
(NAS-Bench-Suite-Zero).
"""

from airas_eval.metrics import classification as _cls
from airas_eval.metrics import population as _pop
from airas_eval.metrics import search as _search
from airas_eval.metrics import selection as _sel
from airas_eval.spec import MetricBinding, MetricSet
from airas_eval.tasks.generic import _metric_sets as core

# --- search --------------------------------------------------------------------

_SCORES = ("evaluated_scores",)
_WITH_COSTS = ("evaluated_scores", "evaluation_costs")
_WITH_SPACE = ("evaluated_scores", "search_space_scores")


def search_space_fraction_better(
    evaluated_scores: list[float], search_space_scores: list[float]
) -> float:
    return _pop.fraction_better(
        _search.best_score(evaluated_scores), search_space_scores
    )


def gain_over_random_search(
    evaluated_scores: list[float], search_space_scores: list[float]
) -> float:
    return _pop.gain_over_random_search(
        _search.best_score(evaluated_scores), search_space_scores, len(evaluated_scores)
    )


def relative_improvement_over_random(
    evaluated_scores: list[float], search_space_scores: list[float]
) -> float:
    return _pop.relative_improvement(
        _search.best_score(evaluated_scores), search_space_scores
    )


def n_search_space(search_space_scores: list[float]) -> float:
    return float(len(search_space_scores))


SEARCH = MetricSet(
    provenance_packages=core.SEARCH.provenance_packages,
    notes=(
        core.SEARCH.notes + "。コストはベンチマークが公表する各アーキテクチャの学習"
        "コストで、評価順に累積する。ランダム探索ベースラインは search_space_scores "
        "からの n 回一様抽出における最良値の厳密な期待値。相対改善は探索空間平均に"
        "対するもの(Yang et al. 2020)"
    ),
    metrics=core.SEARCH.metrics
    + (
        MetricBinding(
            "cost_to_best",
            _search.cost_to_best,
            _WITH_COSTS,
            description="最終的な最良スコアに初めて到達した時点までの累積コスト(学習時間など)。NAS-Bench の推定 wall-clock 軸。",
        ),
        MetricBinding(
            "search_space_fraction_better",
            search_space_fraction_better,
            _WITH_SPACE,
            description="探索空間全体のうち、見つけた最良候補より真に良い候補の割合。0 なら空間内最良、0.01 なら上位 1%。",
        ),
        MetricBinding(
            "gain_over_random_search",
            gain_over_random_search,
            _WITH_SPACE,
            description="同じ評価回数のランダム探索の期待最良値に対する、見つけた最良スコアの差。負ならランダム探索の方が期待値で優れる。",
        ),
        MetricBinding(
            "relative_improvement_over_random",
            relative_improvement_over_random,
            _WITH_SPACE,
            description="母集団平均に対する相対改善 (score − mean) / mean。search では探索空間全体、architecture では同一パイプラインで学習したランダムアーキテクチャ群が母集団(Yang et al. 2020)。",
        ),
    ),
    curves=core.SEARCH.curves
    + (
        MetricBinding(
            "best_so_far_vs_cost",
            _search.best_so_far_vs_cost,
            _WITH_COSTS,
            description="累積コストに対する暫定最良スコアの推移([累積コスト, best-so-far] の列)。",
        ),
    ),
    summary=core.SEARCH.summary
    + (
        MetricBinding(
            "total_cost",
            _search.total_cost,
            _WITH_COSTS,
            description="評価した全候補のコスト合計(消費した推定予算)。",
        ),
        MetricBinding(
            "n_search_space",
            n_search_space,
            ("search_space_scores",),
            description="探索空間に含まれる候補の総数。",
        ),
    ),
)

# --- predictor -----------------------------------------------------------------

_PAIR = ("predicted_scores", "reference_scores")

PREDICTOR = MetricSet(
    provenance_packages=core.CANDIDATE_RANKING.provenance_packages,
    notes=(
        core.CANDIDATE_RANKING.notes
        + "。上位 10% 相関は、真のスコアが上位 10% に入る候補だけで計算する"
        "(NAS-Bench-Suite-Zero のプロトコル)"
    ),
    metrics=core.CANDIDATE_RANKING.metrics
    + (
        MetricBinding(
            "kendall_tau_top_10pct",
            _sel.rank_correlation_top_fraction,
            _PAIR,
            {"fraction": 0.10, "method": "kendall"},
            description="真のスコアが上位 10% の候補だけに限定した Kendall の τ。空間全体では順位を当てても上位を取りこぼす予測器を検出する(NAS-Bench-Suite-Zero)。",
        ),
        MetricBinding(
            "spearman_rho_top_10pct",
            _sel.rank_correlation_top_fraction,
            _PAIR,
            {"fraction": 0.10, "method": "spearman"},
            description="真のスコアが上位 10% の候補だけに限定した Spearman の ρ。",
        ),
    ),
    curves=core.CANDIDATE_RANKING.curves,
    summary=core.CANDIDATE_RANKING.summary,
)

# --- architecture --------------------------------------------------------------

_LABELS = ("predicted_labels", "reference_labels")
_VS_RANDOM = ("predicted_labels", "reference_labels", "random_architecture_accuracies")


def relative_improvement_over_random_architectures(
    predicted_labels: list[int],
    reference_labels: list[int],
    random_architecture_accuracies: list[float],
) -> float:
    return _pop.relative_improvement(
        _cls.accuracy(predicted_labels, reference_labels),
        random_architecture_accuracies,
    )


def fraction_of_random_architectures_better(
    predicted_labels: list[int],
    reference_labels: list[int],
    random_architecture_accuracies: list[float],
) -> float:
    return _pop.fraction_better(
        _cls.accuracy(predicted_labels, reference_labels),
        random_architecture_accuracies,
    )


def n_random_architectures(random_architecture_accuracies: list[float]) -> float:
    return float(len(random_architecture_accuracies))


def test_regret(
    predicted_labels: list[int], reference_labels: list[int], oracle_test_best: float
) -> float:
    """Benchmark test optimum minus the trained architecture's top-1 accuracy,
    both as fractions in [0, 1]."""
    return _search.final_regret(
        [_cls.accuracy(predicted_labels, reference_labels)], oracle_test_best
    )


ARCHITECTURE = MetricSet(
    per_example=core.CLASSIFICATION.per_example,
    provenance_packages=core.CLASSIFICATION.provenance_packages,
    notes=(
        core.CLASSIFICATION.notes
        + "。ランダムアーキテクチャベースラインは、同じ探索空間から一様に抽出し同じ"
        "パイプラインで学習したアーキテクチャ群と top-1 正解率を比較する"
        "(Yang et al. 2020)。テストリグレットはベンチマークの公表テスト最適値との差で、"
        "いずれも 0〜1 の割合"
    ),
    metrics=core.CLASSIFICATION.metrics
    + (
        MetricBinding(
            "relative_improvement_over_random",
            relative_improvement_over_random_architectures,
            _VS_RANDOM,
            description="母集団平均に対する相対改善 (score − mean) / mean。search では探索空間全体、architecture では同一パイプラインで学習したランダムアーキテクチャ群が母集団(Yang et al. 2020)。",
        ),
        MetricBinding(
            "fraction_of_random_better",
            fraction_of_random_architectures_better,
            _VS_RANDOM,
            description="同一パイプラインで学習したランダムアーキテクチャのうち、最終アーキテクチャより精度が高いものの割合。",
        ),
        MetricBinding(
            "test_regret",
            test_regret,
            (*_LABELS, "oracle_test_best"),
            description="テストリグレット。ベンチマークの公表テスト最適値と、学習済み最終アーキテクチャのテスト正解率の差(いずれも 0〜1 の割合)。",
        ),
    ),
    summary=core.CLASSIFICATION.summary
    + (
        MetricBinding(
            "n_random_architectures",
            n_random_architectures,
            ("random_architecture_accuracies",),
            description="ベースラインとして与えられたランダムアーキテクチャの数。",
        ),
    ),
)
