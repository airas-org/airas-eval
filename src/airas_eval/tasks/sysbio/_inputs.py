"""Systems-biology input contracts: one flat model per task.

A reaction network inference run is scored per instance (one biochemical
model each) and the task reports averages over instances, so every field is
a list with one entry per instance, in the same order.
"""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from airas_eval.tasks.generic._inputs import TaskInputs, _check_finite


class Reaction(BaseModel):
    """A reaction as its participating species ids. Order and stoichiometry are
    ignored: two reactions match when their reactant sets and product sets
    (and, for the strict variant, modifier sets) are equal."""

    model_config = ConfigDict(extra="forbid")

    reactants: list[str] = Field(description="反応物の種 ID。")
    products: list[str] = Field(description="生成物の種 ID。")
    modifiers: list[str] = Field(
        default_factory=list, description="modifier(酵素・触媒など)の種 ID。"
    )


Trajectory = dict[str, list[float]]


class ReactionNetworkInferenceInputs(TaskInputs):
    """反応ネットワーク推定: インスタンス(生化学モデル)ごとに、提出モデルが追加した反応と
    真のモデルから取り除かれていた反応、および同じシミュレーション条件で得た提出モデルと
    真のモデルの時系列。4 つのリストは同じ長さ・順序。"""

    predicted_reactions: list[list[Reaction]] = Field(
        description="インスタンスごとに、提出モデルが追加した反応(与えられた不完全モデルに元からある反応は除く)。"
    )
    reference_reactions: list[list[Reaction]] = Field(
        description="インスタンスごとに、真のモデルから取り除かれていた反応。実験設計で固定された参照データ。"
    )
    predicted_trajectories: list[Trajectory] = Field(
        description="インスタンスごとに、提出モデルを真のモデルと同じ条件でシミュレートした時系列 {種 ID: 濃度列}。"
        "提出モデルが実行できない場合は、採点規約が定める代替モデル(不完全モデルなど)の時系列を与える。"
    )
    reference_trajectories: list[Trajectory] = Field(
        description="インスタンスごとに、真のモデルの時系列 {種 ID: 濃度列}。実験設計で固定された参照データ。"
    )

    @model_validator(mode="after")
    def _consistent(self) -> "ReactionNetworkInferenceInputs":
        lengths = {
            len(self.predicted_reactions),
            len(self.reference_reactions),
            len(self.predicted_trajectories),
            len(self.reference_trajectories),
        }
        if len(lengths) != 1:
            raise ValueError("all four lists must have one entry per instance")
        if not self.predicted_reactions:
            raise ValueError("at least one instance is required")
        for name in ("predicted_trajectories", "reference_trajectories"):
            for i, trajectory in enumerate(getattr(self, name)):
                for species, series in trajectory.items():
                    _check_finite(series, f"{name}[{i}][{species!r}]")
        return self
