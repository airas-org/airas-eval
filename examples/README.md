# Examples

Minimal inputs for each NAS task, in the shape `airas-eval score` expects
(one object per input group). Run them as a consumer would:

```bash
uvx airas-eval score nas_search --inputs examples/nas_search.json
# or, from a checkout:
uv run airas-eval score nas_search --inputs examples/nas_search.json
```

`tests/test_cli.py` runs every file here through the installed CLI.
