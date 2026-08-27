from __future__ import annotations

from neural_a_share.gui import (
    NeuralAlphaApp,
    _model_is_degraded,
    _walk_forward_max_folds,
)


class _PipelineSpy:
    def __init__(self) -> None:
        self.train_mode: bool | None = None
        self.walk_forward_mode: bool | None = None
        self.max_folds: int | None = None
        self.resume: bool | None = None

    def train(self, allow_degraded_survivorship: bool = False) -> str:
        self.train_mode = allow_degraded_survivorship
        return "trained"

    def walk_forward(
        self,
        max_folds: int | None = None,
        allow_degraded_survivorship: bool = False,
        resume: bool = True,
    ) -> str:
        self.max_folds = max_folds
        self.walk_forward_mode = allow_degraded_survivorship
        self.resume = resume
        return "walked"


def test_gui_recovers_degraded_mode_from_registry_or_version() -> None:
    assert _model_is_degraded("model-v1", {"survivorship_status": "DEGRADED"})
    assert _model_is_degraded("model-v1-degraded", {})
    assert not _model_is_degraded("model-v1", {"survivorship_status": "PASS"})


def test_gui_passes_explicit_degraded_mode_to_train_and_walk_forward() -> None:
    app = NeuralAlphaApp.__new__(NeuralAlphaApp)
    app.allow_degraded_survivorship = True
    app.walk_forward_max_folds = 5
    pipeline = _PipelineSpy()
    app._pipeline = lambda: pipeline  # type: ignore[method-assign]

    assert app._train_model() == "trained"
    assert app._walk_forward() == "walked"
    assert pipeline.train_mode is True
    assert pipeline.walk_forward_mode is True
    assert pipeline.max_folds == 5
    assert pipeline.resume is True


def test_gui_keeps_strict_mode_when_override_is_disabled() -> None:
    app = NeuralAlphaApp.__new__(NeuralAlphaApp)
    app.allow_degraded_survivorship = False
    pipeline = _PipelineSpy()
    app._pipeline = lambda: pipeline  # type: ignore[method-assign]

    app._walk_forward()
    assert pipeline.walk_forward_mode is False
    assert pipeline.max_folds == 2


def test_gui_walk_forward_scope_is_explicit() -> None:
    assert _walk_forward_max_folds("最近 2 折（快速）") == 2
    assert _walk_forward_max_folds("最近 10 折") == 10
    assert _walk_forward_max_folds("全部折（完整范围）") is None
