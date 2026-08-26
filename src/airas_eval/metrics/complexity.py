"""Model complexity metrics (NAS reporting: the second axis next to accuracy).

These take a model, not (predictions, references), so they live apart from the
pure core. FLOPs counting is notoriously convention-dependent (FMA counted as
1 or 2 ops; per-operator coverage differs across counters), so results are
returned with the counter name and convention made explicit rather than as a
bare number.
"""

from typing import Any


def parameter_count(model: Any, trainable_only: bool = False) -> int:
    """Total parameter count of a torch.nn.Module. Requires torch."""
    params = model.parameters()
    if trainable_only:
        return int(sum(p.numel() for p in params if p.requires_grad))
    return int(sum(p.numel() for p in params))


def macs(model: Any, example_input: Any) -> dict[str, Any]:
    """Multiply-accumulate count via fvcore's FlopCountAnalysis.

    Requires torch and fvcore. Note: fvcore counts one fused multiply-add as a
    single operation — the returned number is MACs; multiply by 2 for the
    FLOPs=2*MACs convention. The counter identity is included in the result so
    reports can cite it (papers that just say "FLOPs" are ambiguous).
    """
    from fvcore.nn import FlopCountAnalysis  # noqa: PLC0415 - optional dependency

    analysis = FlopCountAnalysis(model, example_input)
    analysis.unsupported_ops_warnings(False)
    total_macs = int(analysis.total())
    return {
        "macs": total_macs,
        "flops_2x_macs": 2 * total_macs,
        "counter": "fvcore.nn.FlopCountAnalysis",
        "convention": "FMA counted as 1 op (MACs)",
        "unsupported_ops": {k: int(v) for k, v in analysis.unsupported_ops().items()},
    }
