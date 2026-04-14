"""Session auto-tune for Tag all: bounded coordinate descent on batch size, Hydrus parallelism, and optionally ORT intra-op threads.

Strategy (§4.2 / §4.4): warm-up (default **three** outer batches) → sweep batch sizes → sweep ``hydrus_download_parallel`` at best batch
→ optionally sweep ``cpu_intra_op_threads`` with ``inter_op=1`` (each new value may require ONNX reload).
Supervised mode requires ``tuning_ack`` when any knob in the proposal differs from the completed batch.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Literal

from backend.config import AppConfig

log = logging.getLogger(__name__)

DEFAULT_WARM_UP_BATCHES = 3

TuningControlMode = Literal["supervised", "auto_lucky"]


@dataclass(frozen=True)
class ResolvedBounds:
    """Clamped search rectangle within global AppConfig limits."""

    batch_min: int
    batch_max: int
    dlp_min: int
    dlp_max: int


@dataclass
class AfterBatchResult:
    """Returned after each outer batch when session auto-tune is active."""

    next_batch_size: int
    next_download_parallel: int
    next_ort_intra: int
    next_ort_inter: int
    tuning_state: dict
    require_ack_before_next: bool


def _triplet(lo: int, hi: int) -> list[int]:
    """Up to three trial points: lo, mid, hi (deduplicated, sorted)."""
    if lo > hi:
        lo, hi = hi, lo
    if lo == hi:
        return [lo]
    mid = (lo + hi) // 2
    return sorted({lo, mid, hi})


def _default_intra_upper_bound(config: AppConfig) -> int:
    """Cap intra-op search upper bound by CPU count (§4.4)."""
    try:
        n = len(os.sched_getaffinity(0))
    except Exception:
        n = os.cpu_count() or 8
    return max(1, min(64, int(n)))


def normalize_tuning_control_mode(raw: object | None) -> tuple[TuningControlMode, bool]:
    """Return (mode, clamped_invalid). Omitted → auto_lucky (§4.1)."""
    if raw is None:
        return "auto_lucky", False
    s = str(raw).strip().lower()
    if s in ("supervised",):
        return "supervised", False
    if s in ("auto_lucky", "im_feeling_lucky", "lucky"):
        return "auto_lucky", False
    return "auto_lucky", True


def resolve_tuning_bounds(config: AppConfig, raw: object | None) -> tuple[ResolvedBounds, list[str]]:
    """Merge client ``tuning_bounds`` with global limits; collect clamp warnings."""
    warnings: list[str] = []
    g_bs_lo, g_bs_hi = 1, 256
    g_dlp_lo, g_dlp_hi = 1, 32

    b_min, b_max = g_bs_lo, g_bs_hi
    dlp_min, dlp_max = g_dlp_lo, g_dlp_hi

    if isinstance(raw, dict):
        bs_o = raw.get("batch_size")
        if isinstance(bs_o, dict):
            for key, target in (("min", b_min), ("max", b_max)):
                v = bs_o.get(key)
                if v is not None:
                    try:
                        iv = int(v)
                        if key == "min":
                            b_min = iv
                        else:
                            b_max = iv
                    except (TypeError, ValueError):
                        warnings.append(f"ignored invalid tuning_bounds.batch_size.{key}")
        dlp_o = raw.get("hydrus_download_parallel")
        if isinstance(dlp_o, dict):
            for key, target in (("min", dlp_min), ("max", dlp_max)):
                v = dlp_o.get(key)
                if v is not None:
                    try:
                        iv = int(v)
                        if key == "min":
                            dlp_min = iv
                        else:
                            dlp_max = iv
                    except (TypeError, ValueError):
                        warnings.append(f"ignored invalid tuning_bounds.hydrus_download_parallel.{key}")

    o_b_min, o_b_max = b_min, b_max
    o_d_min, o_d_max = dlp_min, dlp_max

    b_min = max(g_bs_lo, min(b_min, g_bs_hi))
    b_max = max(g_bs_lo, min(b_max, g_bs_hi))
    if b_min > b_max:
        warnings.append("tuning_bounds batch_size min>max; swapped")
        b_min, b_max = b_max, b_min

    dlp_min = max(g_dlp_lo, min(dlp_min, g_dlp_hi))
    dlp_max = max(g_dlp_lo, min(dlp_max, g_dlp_hi))
    if dlp_min > dlp_max:
        warnings.append("tuning_bounds hydrus_download_parallel min>max; swapped")
        dlp_min, dlp_max = dlp_max, dlp_min

    if (o_b_min, o_b_max) != (b_min, b_max) and isinstance(raw, dict) and raw.get("batch_size"):
        warnings.append("tuning_bounds.batch_size clamped to 1–256")
    if (o_d_min, o_d_max) != (dlp_min, dlp_max) and isinstance(raw, dict) and raw.get("hydrus_download_parallel"):
        warnings.append("tuning_bounds.hydrus_download_parallel clamped to 1–32")

    return ResolvedBounds(batch_min=b_min, batch_max=b_max, dlp_min=dlp_min, dlp_max=dlp_max), warnings


def resolve_intra_thread_bounds(
    config: AppConfig,
    raw: object | None,
    *,
    default_hi: int | None = None,
) -> tuple[int, int, list[str]]:
    """Return (intra_min, intra_max) clamped to 1–64; optional ``tuning_bounds.cpu_intra_op_threads``."""
    warnings: list[str] = []
    hi_cap = default_hi if default_hi is not None else _default_intra_upper_bound(config)
    lo, hi = 1, hi_cap

    if isinstance(raw, dict):
        t_o = raw.get("cpu_intra_op_threads")
        if isinstance(t_o, dict):
            for key, attr in (("min", "lo"), ("max", "hi")):
                v = t_o.get(key)
                if v is not None:
                    try:
                        iv = int(v)
                        if key == "min":
                            lo = iv
                        else:
                            hi = iv
                    except (TypeError, ValueError):
                        warnings.append(f"ignored invalid tuning_bounds.cpu_intra_op_threads.{key}")

    lo = max(1, min(64, lo))
    hi = max(1, min(64, hi))
    if lo > hi:
        warnings.append("tuning_bounds.cpu_intra_op_threads min>max; swapped")
        lo, hi = hi, lo

    return lo, hi, warnings


def clamp_supervised_timeout_s(raw: object | None) -> float | None:
    """Optional timeout for supervised gates; None = wait indefinitely."""
    if raw is None:
        return None
    try:
        t = float(raw)
    except (TypeError, ValueError):
        return None
    if t <= 0:
        return None
    return max(30.0, min(3600.0, t))


class SessionAutoTune:
    """Coordinate-descent tuner with optional supervised acknowledgements and optional ORT intra sweep."""

    def __init__(
        self,
        *,
        mode: TuningControlMode,
        baseline_batch: int,
        baseline_dlp: int,
        bounds: ResolvedBounds,
        warm_up_batches: int = DEFAULT_WARM_UP_BATCHES,
        tune_threads: bool = False,
        baseline_ort_intra: int = 8,
        baseline_ort_inter: int = 1,
        intra_bounds: tuple[int, int] | None = None,
    ) -> None:
        self._mode = mode
        self._baseline_bs = baseline_batch
        self._baseline_dlp = baseline_dlp
        self._bounds = bounds
        self._warm_up = max(1, int(warm_up_batches))
        self._tune_threads = tune_threads
        self._baseline_intra = max(1, min(64, int(baseline_ort_intra)))
        self._baseline_inter = max(1, min(16, int(baseline_ort_inter)))

        self._bs_cand = _triplet(bounds.batch_min, bounds.batch_max)
        self._dlp_cand = _triplet(bounds.dlp_min, bounds.dlp_max)

        if tune_threads and intra_bounds is not None:
            ilo, ihi = intra_bounds
            self._intra_cand = _triplet(ilo, ihi)
        else:
            self._intra_cand = []

        self._phase: Literal[
            "warm_up",
            "explore_bs",
            "explore_dlp",
            "explore_intra",
            "hold",
        ] = "warm_up"
        self._batches_done = 0
        self._i_bs = 0
        self._i_dlp = 0
        self._i_intra = 0
        self._scores_bs: list[tuple[float, int]] = []
        self._scores_dlp: list[tuple[float, int]] = []
        self._scores_intra: list[tuple[float, int]] = []

        self._best_bs = baseline_batch
        self._best_dlp = baseline_dlp
        self._best_intra = self._baseline_intra

    def after_batch(self, row: dict) -> AfterBatchResult:
        """Update state from one performance row (merged batch metrics + WS fields)."""
        self._batches_done += 1

        cur_bs = int(row.get("effective_batch") or self._baseline_bs)
        cur_dlp = int(row.get("download_parallel") or self._baseline_dlp)
        cur_intra = int(row.get("ort_intra_op_threads") or self._baseline_intra)
        cur_inter = int(row.get("ort_inter_op_threads") or self._baseline_inter)

        wall = (
            float(row.get("fetch_s") or 0)
            + float(row.get("predict_s") or 0)
            + float(row.get("hydrus_apply_batch_s") or 0)
        )
        files = int(row.get("files_in_batch") or row.get("predict_queue") or 0)
        score = (files / wall) if wall > 0 else 0.0

        next_bs = cur_bs
        next_dlp = cur_dlp
        next_intra = cur_intra
        next_inter = cur_inter

        if not self._tune_threads:
            next_intra, next_inter = self._baseline_intra, self._baseline_inter

        if self._phase == "warm_up":
            if self._batches_done < self._warm_up:
                next_bs, next_dlp = self._baseline_bs, self._baseline_dlp
                if self._tune_threads:
                    next_intra, next_inter = self._baseline_intra, self._baseline_inter
            elif self._batches_done == self._warm_up:
                self._phase = "explore_bs"
                self._i_bs = 0
                next_bs = self._bs_cand[0]
                next_dlp = self._baseline_dlp
                if self._tune_threads:
                    next_intra, next_inter = self._baseline_intra, self._baseline_inter
            else:
                next_bs, next_dlp = self._baseline_bs, self._baseline_dlp
                if self._tune_threads:
                    next_intra, next_inter = self._baseline_intra, self._baseline_inter

        elif self._phase == "explore_bs":
            idx = self._i_bs
            self._scores_bs.append((score, self._bs_cand[idx]))
            self._i_bs += 1
            if self._i_bs < len(self._bs_cand):
                next_bs = self._bs_cand[self._i_bs]
                next_dlp = self._baseline_dlp
            else:
                self._best_bs = max(self._scores_bs, key=lambda x: x[0])[1]
                self._phase = "explore_dlp"
                self._i_dlp = 0
                next_bs = self._best_bs
                next_dlp = self._dlp_cand[0]
            if self._tune_threads:
                next_intra, next_inter = self._baseline_intra, self._baseline_inter

        elif self._phase == "explore_dlp":
            idx = self._i_dlp
            self._scores_dlp.append((score, self._dlp_cand[idx]))
            self._i_dlp += 1
            if self._i_dlp < len(self._dlp_cand):
                next_bs = self._best_bs
                next_dlp = self._dlp_cand[self._i_dlp]
            else:
                self._best_dlp = max(self._scores_dlp, key=lambda x: x[0])[1]
                if self._tune_threads and self._intra_cand:
                    self._phase = "explore_intra"
                    self._i_intra = 0
                    next_bs = self._best_bs
                    next_dlp = self._best_dlp
                    next_intra = self._intra_cand[0]
                    next_inter = 1
                else:
                    self._phase = "hold"
                    next_bs, next_dlp = self._best_bs, self._best_dlp
                    if self._tune_threads:
                        next_intra, next_inter = self._best_intra, 1
                    else:
                        next_intra, next_inter = self._baseline_intra, self._baseline_inter

        elif self._phase == "explore_intra":
            idx = self._i_intra
            self._scores_intra.append((score, self._intra_cand[idx]))
            self._i_intra += 1
            if self._i_intra < len(self._intra_cand):
                next_bs = self._best_bs
                next_dlp = self._best_dlp
                next_intra = self._intra_cand[self._i_intra]
                next_inter = 1
            else:
                self._best_intra = max(self._scores_intra, key=lambda x: x[0])[1]
                self._phase = "hold"
                next_bs, next_dlp = self._best_bs, self._best_dlp
                next_intra, next_inter = self._best_intra, 1
        else:
            next_bs, next_dlp = self._best_bs, self._best_dlp
            if self._tune_threads:
                next_intra, next_inter = self._best_intra, 1
            else:
                next_intra, next_inter = self._baseline_intra, self._baseline_inter

        require_ack = self._mode == "supervised" and (
            (next_bs, next_dlp, next_intra, next_inter)
            != (cur_bs, cur_dlp, cur_intra, cur_inter)
        )

        log.debug(
            "session_autotune batch_done=%s phase=%s score=%.6f files=%s wall_s=%.4f "
            "cur_bs=%s cur_dlp=%s next_bs=%s next_dlp=%s",
            self._batches_done,
            self._phase,
            score,
            files,
            wall,
            cur_bs,
            cur_dlp,
            next_bs,
            next_dlp,
        )

        tuning_state = self._tuning_state_dict(
            next_bs=next_bs,
            next_dlp=next_dlp,
            next_intra=next_intra,
            next_inter=next_inter,
            awaiting_approval=require_ack,
        )

        return AfterBatchResult(
            next_batch_size=next_bs,
            next_download_parallel=next_dlp,
            next_ort_intra=next_intra,
            next_ort_inter=next_inter,
            tuning_state=tuning_state,
            require_ack_before_next=require_ack,
        )

    def _phase_ui_labels(
        self,
        next_bs: int,
        next_dlp: int,
        next_intra: int,
        next_inter: int,
        *,
        awaiting_approval: bool,
    ) -> dict[str, str]:
        """Human-readable tuning step for WebSocket / UI (sequential phases)."""
        if awaiting_approval:
            return {
                "phase_title": "Supervised tuning — approval required",
                "phase_detail": (
                    f"Next proposal: inference batch {next_bs}, Hydrus parallel {next_dlp}, "
                    f"ORT intra-op {next_intra}, inter-op {next_inter}. "
                    "Use the tab that started Tag all and click “Approve tuning step” "
                    "(read-only progress views cannot send approval)."
                ),
            }
        p = self._phase
        if p == "warm_up":
            step = min(self._batches_done, self._warm_up)
            return {
                "phase_title": "Warm-up — baseline measurement",
                "phase_detail": (
                    f"Sequential step {step} of {self._warm_up}: fixed baseline — batch {self._baseline_bs}, "
                    f"Hydrus parallel {self._baseline_dlp}, ORT intra {self._baseline_intra} "
                    f"(inter {self._baseline_inter})."
                ),
            }
        if p == "explore_bs":
            n = len(self._bs_cand)
            k = len(self._scores_bs)
            return {
                "phase_title": "Search — inference batch size (one candidate per batch)",
                "phase_detail": (
                    f"Trial {k} of {n} completed; Hydrus download parallel fixed at {self._baseline_dlp}. "
                    f"Next batch uses inference batch {next_bs}."
                ),
            }
        if p == "explore_dlp":
            n = len(self._dlp_cand)
            k = len(self._scores_dlp)
            return {
                "phase_title": "Search — Hydrus download parallelism",
                "phase_detail": (
                    f"Trial {k} of {n} completed; inference batch fixed at best so far ({self._best_bs}). "
                    f"Next batch uses parallel {next_dlp} concurrent downloads."
                ),
            }
        if p == "explore_intra":
            n = len(self._intra_cand) if self._intra_cand else 0
            k = len(self._scores_intra)
            return {
                "phase_title": "Search — ONNX Runtime CPU intra-op threads",
                "phase_detail": (
                    f"Trial {k} of {n} completed (ORT session may reload each trial). "
                    f"Next batch: intra-op {next_intra}, inter-op {next_inter}. "
                    f"Batch {self._best_bs}, Hydrus parallel {self._best_dlp}."
                ),
            }
        # hold
        return {
            "phase_title": "Hold — best settings for remainder of run",
            "phase_detail": (
                f"Using batch {self._best_bs}, Hydrus parallel {self._best_dlp}, "
                f"ORT intra {self._best_intra} / inter {self._baseline_inter}."
            ),
        }

    def _tuning_state_dict(
        self,
        *,
        next_bs: int,
        next_dlp: int,
        next_intra: int,
        next_inter: int,
        awaiting_approval: bool,
    ) -> dict:
        phase = self._phase
        if awaiting_approval:
            phase = "awaiting_approval"
        ui = self._phase_ui_labels(
            next_bs, next_dlp, next_intra, next_inter, awaiting_approval=awaiting_approval
        )
        out: dict = {
            "phase": phase,
            "underlying_phase": self._phase,
            "phase_title": ui["phase_title"],
            "phase_detail": ui["phase_detail"],
            "warm_up_batches": self._warm_up,
            "batches_completed": self._batches_done,
            "next_batch_size": next_bs,
            "next_download_parallel": next_dlp,
            "next_ort_intra_op_threads": next_intra,
            "next_ort_inter_op_threads": next_inter,
            "best_batch_size": self._best_bs,
            "best_download_parallel": self._best_dlp,
            "best_ort_intra_op_threads": self._best_intra,
            "baseline_batch_size": self._baseline_bs,
            "baseline_download_parallel": self._baseline_dlp,
            "baseline_ort_intra_op_threads": self._baseline_intra,
            "baseline_ort_inter_op_threads": self._baseline_inter,
            "session_auto_tune_threads": self._tune_threads,
            "awaiting_approval": awaiting_approval,
            "proposal": {
                "batch_size": next_bs,
                "hydrus_download_parallel": next_dlp,
                "cpu_intra_op_threads": next_intra,
                "cpu_inter_op_threads": next_inter,
            },
            "control_mode": self._mode,
        }
        return out

    def tuning_search_batches_planned(self) -> int:
        """Outer batches dedicated to warm-up + coordinate search (before **hold**)."""
        n = self._warm_up + len(self._bs_cand) + len(self._dlp_cand)
        if self._tune_threads and self._intra_cand:
            n += len(self._intra_cand)
        return max(1, n)

    def tuning_search_batches_done(self) -> int:
        """Completed search batches (same units as :meth:`tuning_search_batches_planned`)."""
        total = self.tuning_search_batches_planned()
        if self._phase == "hold":
            return total
        if self._phase == "warm_up":
            return min(self._batches_done, self._warm_up)
        done = self._warm_up
        if self._phase == "explore_bs":
            return min(done + len(self._scores_bs), total)
        done += len(self._bs_cand)
        if self._phase == "explore_dlp":
            return min(done + len(self._scores_dlp), total)
        done += len(self._dlp_cand)
        if self._phase == "explore_intra":
            return min(done + len(self._scores_intra), total)
        return min(self._batches_done, total)

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def best_pair(self) -> tuple[int, int]:
        return self._best_bs, self._best_dlp

    @property
    def best_ort_threads(self) -> tuple[int, int]:
        if self._tune_threads:
            return self._best_intra, 1
        return self._baseline_intra, self._baseline_inter

    def summary_for_report(self) -> dict:
        """Structured summary for ``tuning_report.autotune`` (end of run)."""
        out: dict = {
            "phase": self._phase,
            "best_batch_size": self._best_bs,
            "best_download_parallel": self._best_dlp,
            "explore_bs_scores": [{"score": s, "batch_size": b} for s, b in self._scores_bs],
            "explore_dlp_scores": [{"score": s, "download_parallel": d} for s, d in self._scores_dlp],
            "session_auto_tune_threads": self._tune_threads,
        }
        if self._tune_threads:
            out["best_ort_intra_op_threads"] = self._best_intra
            out["best_ort_inter_op_threads"] = 1
            out["explore_intra_scores"] = [{"score": s, "intra": i} for s, i in self._scores_intra]
        return out

    def ui_snapshot_commit_phase(self) -> dict:
        """Stable WebSocket payload for learning **commit** segment: knobs locked, no ``after_batch`` advances."""
        bi, bo = self.best_ort_threads
        return {
            "phase": "commit_apply",
            "underlying_phase": "commit_apply",
            "phase_title": "Commit segment — apply tags with locked best settings",
            "phase_detail": (
                f"Inference batch {self._best_bs}, Hydrus parallel {self._best_dlp}, "
                f"ORT intra-op {bi}, inter-op {bo}. Incremental Hydrus writes follow your Tag all rules."
            ),
            "calibration_segment": "commit",
            "warm_up_batches": self._warm_up,
            "batches_completed": self._batches_done,
            "next_batch_size": self._best_bs,
            "next_download_parallel": self._best_dlp,
            "next_ort_intra_op_threads": bi,
            "next_ort_inter_op_threads": bo,
            "best_batch_size": self._best_bs,
            "best_download_parallel": self._best_dlp,
            "best_ort_intra_op_threads": self._best_intra,
            "baseline_batch_size": self._baseline_bs,
            "baseline_download_parallel": self._baseline_dlp,
            "baseline_ort_intra_op_threads": self._baseline_intra,
            "baseline_ort_inter_op_threads": self._baseline_inter,
            "session_auto_tune_threads": self._tune_threads,
            "awaiting_approval": False,
            "proposal": {
                "batch_size": self._best_bs,
                "hydrus_download_parallel": self._best_dlp,
                "cpu_intra_op_threads": bi,
                "cpu_inter_op_threads": bo,
            },
            "control_mode": self._mode,
        }

    def merge_progress_ui_fields(
        self,
        tuning_state: dict,
        perf_series: list[dict],
        *,
        commit_segment: bool,
    ) -> dict:
        """Add tuning search progress and rough ETA for WebSocket / UI (not persisted in YAML)."""
        out = dict(tuning_state)
        phase = str(out.get("phase") or "")
        if commit_segment or phase == "commit_apply":
            total = self.tuning_search_batches_planned()
            out["tuning_search_done"] = total
            out["tuning_search_total"] = total
            out["tuning_search_complete"] = True
            out["tuning_eta_seconds"] = None
            out["tuning_avg_batch_wall_s"] = None
            return out
        total = self.tuning_search_batches_planned()
        done = self.tuning_search_batches_done()
        avg = None
        if perf_series:
            tail = perf_series[-min(3, len(perf_series)) :]
            walls: list[float] = []
            for r in tail:
                w = float(r.get("fetch_s") or 0) + float(r.get("predict_s") or 0) + float(
                    r.get("hydrus_apply_batch_s") or 0
                )
                walls.append(w)
            if walls:
                avg = sum(walls) / len(walls)
        rem = max(0, total - done)
        eta = round(avg * rem, 1) if avg is not None and rem > 0 else None
        if out.get("awaiting_approval"):
            eta = None
        out["tuning_search_done"] = done
        out["tuning_search_total"] = total
        out["tuning_search_complete"] = self._phase == "hold"
        out["tuning_eta_seconds"] = eta
        out["tuning_avg_batch_wall_s"] = round(avg, 4) if avg else None
        return out
