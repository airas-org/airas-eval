# 入力例

NAS の各タスクタイプに対する最小の入力ファイル。`airas-eval score` が期待する形
(入力グループごとに 1 オブジェクト)になっている。利用者と同じ方法で実行できる:

```bash
uvx airas-eval score nas_post_training --inputs examples/nas_post_training.json
# チェックアウトから実行する場合:
uv run airas-eval score nas_post_training --inputs examples/nas_post_training.json
```

`tests/test_cli.py` は、ここにある全ファイルをインストール済み CLI に通して検証する。
