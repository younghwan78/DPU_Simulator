# DPU IB Simulator Implementation Goal

## Objective

Implement the DPU IB simulator from `C:/Users/user/Downloads/DPU_IB_implementation.md` as a uv-managed Python project, centered on a single `dpu_ib_sim.py` program and verified through TDD.

## Original Request

`/goal 이 문서를 목표로 구현을 하려고 하는데 자세하게 검토해서 목표 설정하고 구현해줘. uv 가상환경에서 TDD형태로 진행해줘`

## Intake Summary

- Input shape: `existing_plan`
- Audience: local engineering user who needs a runnable simulator and regression harness.
- Authority: `requested`
- Proof type: `test`
- Completion proof: `uv run pytest` passes and `uv run python dpu_ib_sim.py --sweep examples/golden.yaml --out <csv>` writes a CSV from the golden config.
- Goal oracle: the implementation must reproduce the spec's golden `IB_outfifo_preload`, `IB_streaming`, and final binding term, while surfacing F1-F5 flagged assumptions and gracefully degrading rotation preload when `DPU_ACLK_MHz` is absent.
- Likely misfire: building UI scaffolding without a trustworthy pure calculation engine or missing the F3 `DPU_ACLK` blocking behavior.
- Blind spots considered: the supplied spec intentionally leaves rotation preload underdetermined because `DPU_ACLK_MHz` is absent; the first tranche must make that ambiguity visible instead of hiding it behind a magic default.
- Existing plan facts: single-file Python program, pure engine separate from UI imports, YAML config, pytest golden regression, PySide6 GUI path, matplotlib sweep path, headless `--sweep` CSV path.

## Goal Oracle

The oracle for this goal is:

`uv run pytest` passes, and a headless sweep from `examples/golden.yaml` writes rows whose golden baseline binds `IB_outfifo_preload` at about `3735.58 MB/s`.

The PM must keep comparing task receipts to this oracle. Planning, discovery, a passing tiny slice, or a clean-looking board is not enough. The goal finishes only when a final Judge/PM audit maps receipts and verification back to this oracle and records `full_outcome_complete: true`.

## Goal Kind

`existing_plan`

## Current Tranche

Build the first complete local implementation slice: uv project metadata, golden YAML, pure engine, CLI sweep CSV, lazy GUI entrypoint, and pytest coverage for the acceptance-critical engine paths.

## Non-Negotiable Constraints

- Use uv for environment and verification commands.
- Follow TDD: failing tests first, then implementation.
- Keep calculation logic UI-free and importable without PySide6 or matplotlib.
- Do not invent a hidden `DPU_ACLK_MHz` default; when it is missing, rotation preload is `N/A` and F3 is flagged.
- Use the provided document as the implementation contract.

## Stop Rule

Stop only when a final audit proves the full original outcome is complete.

Do not stop after planning, discovery, or task selection while a safe Worker task can be activated.

## Canonical Board

Machine truth lives at:

`docs/goals/dpu-ib-simulator/state.yaml`

If this charter and `state.yaml` disagree, `state.yaml` wins for task status, active task, receipts, verification freshness, and completion truth.

## Run Command

```text
/goal Follow docs/goals/dpu-ib-simulator/goal.md.
```
