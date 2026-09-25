"""SciGym-small inputs: the benchmark's own files per instance, plus the
submitted model. The reference files are pinned by sha256 (``small_manifest.json``,
generated from the official release), so a run cannot quietly score against
a different dataset."""

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from airas_eval.tasks.generic._inputs import TaskInputs

MANIFEST: dict[str, dict[str, str]] = json.loads(
    (Path(__file__).parent / "small_manifest.json").read_text()
)


class Instance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        description="BioModels ID(BIOMD0000000027 など)。SciGym-small の 137 件のいずれか。"
    )
    reference_sbml: str = Field(
        description="真のモデル(truth.xml の内容)。公式データと sha256 で照合される。"
    )
    incomplete_sbml: str = Field(
        description="反応を取り除いた不完全モデル(partial.xml の内容)。公式データと照合される。"
    )
    reference_sedml: str = Field(
        description="シミュレーション条件(truth.sedml の内容)。公式データと照合される。"
    )
    submitted_sbml: str | None = Field(
        default=None,
        description="エージェントが提出したモデル。無い場合や無効な場合は公式規約どおり不完全モデルで採点する。",
    )


class ScigymSmallInputs(TaskInputs):
    """SciGym-small: インスタンスごとに公式の 3 ファイルと提出モデル。ID は 137 件の中から
    重複なく、ファイル内容は公式リリース(h4duan/scigym-sbml の small split)と一致すること。"""

    instances: list[Instance] = Field(
        description="評価するインスタンス。全件(137)でなくてもよく、件数は n_instances で報告される。"
    )

    @model_validator(mode="after")
    def _official_files(self) -> "ScigymSmallInputs":
        if not self.instances:
            raise ValueError("at least one instance is required")
        ids = [i.id for i in self.instances]
        if len(set(ids)) != len(ids):
            raise ValueError("instance ids must be unique")
        for instance in self.instances:
            expected = MANIFEST.get(instance.id)
            if expected is None:
                raise ValueError(f"{instance.id} is not in SciGym-small")
            for field, name in (
                ("reference_sbml", "truth.xml"),
                ("incomplete_sbml", "partial.xml"),
                ("reference_sedml", "truth.sedml"),
            ):
                digest = hashlib.sha256(getattr(instance, field).encode()).hexdigest()
                if digest != expected[name]:
                    raise ValueError(
                        f"{instance.id}/{name} differs from the official release"
                    )
        return self
