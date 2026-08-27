# airas-eval

[AIRAS](https://github.com/airas-org/airas) のための信頼できる評価層。
エージェントはタスクタイプと生の予測結果を渡し、その研究種別で報告すべき標準的な
評価指標一式を固定された形で受け取る。エージェントは評価スクリプトを実装せず、
どの指標を(どの variant で)報告するかも選ばない。

## 構成

2 層構造で、その間に呼び出し側が選べるものはない。

1. **`metrics/`** — 評価指標の実装。入力の形ごとに 1 モジュールで、タスクの知識を
   持たない。標準実装(scikit-learn / scipy)があるものはそれに委譲し、存在しない
   もの(ECE、リグレット、上位 k 選択、2 目的ハイパーボリューム、IGD/GD/spacing)
   だけを自前で実装して variant を固定し、性質テストで検証している。
2. **`tasks/`** — エリアごとに 1 サブパッケージ、タスクタイプごとに 1 モジュール。
   タスクタイプとは「その種類の研究が報告すべき指標の全集合」で、1 つの検証済み
   入力ファイルから計算される(タスクあたり数十指標まで、スカラーまたは曲線、variant は
   すべて固定)。`generic/` が基本となる評価ファミリー、エリアパッケージ(`nas/`)
   はその上に積む — エリアの指標セットは *基本のバインディング + その分野の文献が
   追加するもの* であり、名前を変えた複製ではない。再利用部品
   (`tasks/generic/_metric_sets.py`, `tasks/nas/_metric_sets.py`)は内部の定数で、登録も
   公開も単独評価もされない。

```
tasks/
├── generic/   classification, binary_classification, search, candidate_ranking, multiobjective
└── nas/       nas_pre_training  = search + candidate_ranking の全指標 + NAS 追加分(wall-clock 軸、
               │                    探索空間内順位、ランダム探索比、上位 10% 相関)
               └── nas_post_training = classification + multiobjective の全指標 + NAS 追加分
                                    (ランダムアーキテクチャ比、テストリグレット)
```

NAS は「アーキテクチャの性能をいつ測るか」で 2 タスクに分かれる:

| タスクタイプ | 測るもの | 入力(1 ファイル) |
|---|---|---|
| `nas_pre_training` | 学習前のアーキテクチャ性能 — ベンチマーク参照による探索、性能予測器、ゼロコストプロキシ | 探索軌跡(`evaluated_scores` …)か予測器スコア(`predicted_scores` + `reference_scores`)の少なくとも一方。無い側の指標は skipped |
| `nas_post_training` | 学習後のアーキテクチャ性能 — 選ばれて学習された最終アーキテクチャ | 予測ラベル・正解ラベル(必須)、確率・ベースライン・テスト最適値・(誤り率, コスト) 点集合(任意) |

## 各タスクは何を返すか(指標の説明)

どちらも登録情報から導出されるので、実装と食い違うことがない:

```bash
airas-eval list                                    # タスクタイプの一覧(1 行ずつ: 名前、署名、概要、指標数)
airas-eval list nas_pre_training nas_post_training  # 指定タスクの詳細: 入力(型・値域)、指標(値域・方向・説明)
airas-eval list --all                              # 全タスクの詳細
airas-eval list --json [task ...]                  # 同じ内容を dict で(プログラムから読む契約)
airas-eval validate nas_post_training --inputs inputs.json   # 形式だけ検査(採点しない)
```

`list <task>` の入力欄(`name?: type` と説明、フィールド間の制約)と `list --json` の
`input_schema`(JSON Schema)は、入力検証に使うのと同じ pydantic モデルから生成される
ので、agent が読む契約と評価器が適用する検査が食い違わない。

および、エリアごとに生成される README(テストで同期を検証):

- [`tasks/generic/README.md`](src/airas_eval/tasks/generic/README.md) — 汎用の評価ファミリー
- [`tasks/nas/README.md`](src/airas_eval/tasks/nas/README.md) — NAS の 2 タスク

**各指標の説明(定義、読み方、高低どちらが良いか)はこれらの README の表に載っている。**
タスクやバンドルを変更したら `python -m airas_eval.tasks.readme` で再生成する。

## 使い方

```python
from airas_eval import evaluate

report = evaluate(
    "nas_post_training",
    {
        "predicted_labels": y_pred,
        "reference_labels": y_true,
        "probabilities": probs,  # 任意
        "oracle_test_best": 0.9437,  # 任意: ベンチマークのテスト最適値(0〜1)
        "points": [[error, macs], ...],  # 任意: 精度–効率トレードオフ(全目的を最小化)
        "reference_point": [0.2, 1e9],
    },
)
report.metrics  # スカラー指標: {"accuracy": ..., "test_regret": ..., "hypervolume_2d": ...}
report.curves  # 曲線指標: {"pareto_front": [...]}
report.inputs_summary  # 指標ではない — 入力サイズ: {"n_examples": 10000, ...}
report.skipped  # 計算できなかった指標(理由コード別): missing_optional_input は名前の一覧、他は名前 → 理由
report.omitted_optional_inputs  # 例: ["random_architecture_accuracies", "reference_front"]
report.provenance  # 導出されたタスク署名、依存パッケージの版、入力の SHA-256
```

```bash
airas-eval score nas_pre_training --inputs inputs.json --output evaluation.json
```

複数 seed と 2 システム比較も評価層が引き受ける(agent が統計を自前実装しないため):

```bash
# seed ごとの report を平均 ± 標準偏差に集約(同じタスク署名のものだけ。欠けている指標は incomplete に列挙)
airas-eval aggregate --reports evaluation_seed0.json evaluation_seed1.json evaluation_seed2.json

# 同じ参照データ上での 2 システムのペア比較(事例ごとの正誤に対する符号反転パーミュテーション検定)
airas-eval compare nas_post_training --a inputs_A.json --b inputs_B.json
```

`examples/` に NAS 各タスクの最小入力ファイルがあり、テストスイートがそれぞれを CLI
で採点する。NAS 固有の入力(`evaluation_costs`, `search_space_scores`,
`random_architecture_accuracies`, `oracle_test_best`, ...)は任意の参照データで、
省略するとそれを必要とする指標は skipped として報告され、省略自体も記録される。

入力は 1 タスク 1 ファイル(フラットな JSON オブジェクト)。呼び出し側が選べるのは
タスクタイプだけで、研究をどのタスクタイプで評価するかは研究計画側が決める
(評価ステップではない)。NAS では、探索ステージの後に `nas_pre_training`、
最終アーキテクチャの学習後に `nas_post_training` を呼ぶ、というパイプラインの段階が
それに対応する。

## 設計ルール

1. **指標を選ばせない。** タスクタイプは自分の指標を*すべて*計算する。集合は固定で、
   読み切れる大きさに保つ(長い一覧はそれ自体がチェリーピックの余地になる)。
   variant(平均方法、k、ビン数)はタスクごとに固定され、指標関数が取る全パラメータは
   明示的に固定される(テストで強制)。入力サイズ(`n_examples`, `n_evaluations`, ...)
   は指標とは別に `inputs_summary` に報告され、テスト集合の部分抽出や打ち切られた
   探索が見えるようにする。
2. **黙って消えるものはない。** 計算できなかった指標は `skipped` に理由コード別
   (`missing_optional_input`, `not_applicable`, `undefined_on_data`,
   `missing_dependency`)で現れる。任意入力の省略によるものは名前だけを列挙し
   (原因は `omitted_optional_inputs` と `list` の「必要な入力」で分かる)、それ以外は
   名前 → 理由を持つ。
   skip になるのはこれらの専用ケースだけで、不正な入力やライブラリのバグは
   「未定義」に隠れず例外で失敗する。
3. **来歴は手書きせず導出する。** タスク署名はタスク宣言(指標名、関数の識別子、
   固定 kwargs、入力フィールド)のハッシュなので、実装と乖離できない。エリアごとの
   README も同じ宣言から生成される。
4. **参照データは上流で固定する。** `reference_labels`、`oracle_best`、
   `oracle_test_best`、参照点・参照フロントは実験設計に属する。このライブラリはそれ
   らが本物であることを検証できず、エージェントが制御するプロセスの中では自衛でき
   ない — `airas-eval score` をエージェントが編集できない固定環境から実行すること。

## インストール

```bash
uv add "airas-eval==0.9.0"     # ライブラリ/CLI として
uvx airas-eval@0.9.0 list      # インストールせずに CLI だけ使う
```

評価層が研究の途中で変わらないように、必ずバージョンを固定する。依存: numpy,
scikit-learn, scipy, pydantic。

## 研究リポジトリからの呼び出し方

agent の実験コードは **airas_eval を import しない**。agent の成果物は評価の
**入力ファイル**(`evaluate` に渡す dict をそのまま JSON にしたもの)までで、評価は
固定版の CLI を別プロセスで走らせる。研究リポジトリには airas(オーケストレータ)の
テンプレート由来で次が置かれる:

```toml
# pyproject.toml — 依存として固定するが、実験コードからは import しない
[dependency-groups]
eval = ["airas-eval==0.9.0"]
```

```makefile
# Makefile — task_type と入出力先は研究計画から埋める
evaluate:
	uv run --group eval airas-eval score nas_pre_training \
	    --inputs artifacts/eval_inputs/nas_pre_training.json \
	    --output artifacts/evaluation/nas_pre_training.json
	uv run --group eval airas-eval score nas_post_training \
	    --inputs artifacts/eval_inputs/nas_post_training.json \
	    --output artifacts/evaluation/nas_post_training.json
```

agent は `make evaluate` を実行してスコアを確認しながら実験を進めてよい。ただし
Makefile は agent が書き換え得るので、**公式のスコアはオーケストレータが agent の
触れない環境で、同じ版の CLI を入力ファイルに直接かけて再計算したもの**とする。
report の `inputs_sha256` と入力ファイルの hash、`provenance.versions` の版を照合すれば、
どの入力をどの版で採点したかが確認できる。第三者は clone → `uv sync --group eval` →
`make evaluate` で同じ数字を再現できる(`uv.lock` が版を固定する)。

## リリース手順

公開は GitHub Actions の `publish.yml`(PyPI Trusted Publisher、API トークン不要)で行う。

初回のみ、PyPI 側の設定が必要:

1. https://pypi.org/manage/account/publishing/ で **pending publisher** を登録する —
   PyPI project name `airas-eval`、owner `airas-org`、repository `airas-eval`、
   workflow `publish.yml`、environment `pypi`。
2. GitHub リポジトリの Settings → Environments に `pypi` を作る(承認者を付けてもよい)。

毎回のリリース:

```bash
uv version --bump minor          # 版は pyproject.toml のみ(__version__ はメタデータから読む)
uv lock && uv run pytest -q
git commit -am "release: v0.9.0" && git tag v0.9.0 && git push --tags
```

タグの push で `Publish` ワークフローが起動し、HEAD にその版のタグが付いていることを
検証してから PyPI に公開する(Actions から手動実行も可能)。

## 開発

```bash
uv sync
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
```

タスクタイプの追加 = `tasks/<area>/` に既存バンドルから `TASK` を定義するモジュールを
1 つ置き、エリアの `TASKS` に 1 行足し、`python -m airas_eval.tasks.readme` を実行。
新しいエリア = 新しいサブパッケージ + `tasks.AREAS` に 1 行。バンドルの追加 =
`tasks/generic/_inputs.py` の入力モデルと `tasks/_bundles.py` の `Bundle`(`summary` の件数
付き)。各バインディングには日本語の `description` が必須(テストで強制)。タスクは
5〜10 指標程度に保つ: 標準的な variant のみ、パラメータごとに 1 つの固定値。

登録は意図的に明示的(entry point もスキャンもしない): 信頼層では、何が計算される
かがレビュー済みの差分に見えていなければならない。自前実装の指標を追加する場合は、
オラクル実装とのパリティテスト、または手計算ケース + 性質テストが必要。

## ライセンス

MIT
