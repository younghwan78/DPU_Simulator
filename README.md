# DPU IB Simulator

Single-file Python simulator for estimating DPU input bandwidth (IB) from panel,
layer, and system configuration. The implementation follows the consolidated
`DPU_IB_implementation.md` contract and keeps the calculation engine independent
from the GUI so the golden regression can run quickly in CI or a local shell.

## What It Does

- Computes the three DPU IB terms:
  - `IB_rotation_preload`
  - `IB_outfifo_preload`
  - `IB_streaming`
- Reports `DPU_IB = max(rotation_preload, outfifo_preload, streaming)`.
- Exposes intermediate values for timing, per-layer data, shared preload data,
  streaming, rotation latency, and final term selection.
- Provides a PySide6 desktop GUI with:
  - editable panel/system/layer inputs
  - resizable sidebar and editable layer table
  - highlighted result card showing all three terms and the max binding term
  - grouped/color-coded breakdown table with units
  - formula tab with MathJax-rendered formulae and live-value tables
  - 1D and 2D sweep plotting
- Provides headless CLI modes for:
  - actual-case batch CSV import/export
  - 1D and 2D sweep CSV output
- Uses a uv-managed Python environment and pytest golden regression tests.

## Project Layout

```text
.
|-- dpu_ib_sim.py                  # single-file engine, CLI, and GUI
|-- pyproject.toml                 # uv/project metadata
|-- uv.lock                        # resolved dependency lockfile
|-- examples/
|   |-- golden.yaml                # spec golden example config
|   |-- batch_cases.csv            # sample batch input
|   `-- batch_result.csv           # sample batch output
|-- tests/
|   `-- test_engine_golden.py      # golden engine, CLI, and GUI-structure tests
`-- docs/goals/dpu-ib-simulator/   # GoalBuddy charter/state for this build
```

## Requirements

- Python 3.10+
- uv
- Optional GUI dependencies:
  - PySide6
  - matplotlib

The base dependency set is intentionally small: `PyYAML` for config loading and
`pytest` for development tests. GUI dependencies are optional and installed only
when `--extra gui` is requested.

## Quick Start

From PowerShell:

```powershell
uv run pytest
```

Launch the GUI:

```powershell
uv run --extra gui python dpu_ib_sim.py
```

Launch the GUI with the golden config:

```powershell
uv run --extra gui python dpu_ib_sim.py --config examples/golden.yaml
```

## Headless Sweep

Run a default 1D sweep and write CSV:

```powershell
uv run python dpu_ib_sim.py --sweep examples/golden.yaml --out sweep.csv
```

Run the golden acceptance point:

```powershell
uv run python dpu_ib_sim.py --sweep examples/golden.yaml `
  --param panel.fps --start 120 --stop 120 --step 1 `
  --out sweep_120.csv
```

Expected golden row:

```text
DPU_IB_MBps = 3735.5845959183675
binding_term = IB_outfifo_preload
IB_streaming_MBps = 3598.227359999999
```

Run a 2D sweep:

```powershell
uv run python dpu_ib_sim.py --sweep examples/golden.yaml `
  --param panel.fps --start 60 --stop 120 --step 60 `
  --param2 layers[0].src_h --start2 1080 --stop2 2160 --step2 1080 `
  --out sweep_2d.csv
```

Parameter paths use dot notation for `panel` and `system`, and indexed notation
for layers:

```text
panel.fps
panel.panel_w
panel.panel_h
system.PTW
system.MO_derating
layers[0].src_w
layers[0].src_h
layers[0].scale_v
layers[0].stream_coeff
```

## Headless Batch

Use batch mode when each CSV batch column is a real case to calculate. The
command reads a base YAML config, applies non-empty batch cells as overrides,
calculates each batch independently, and writes a result CSV. Batch input and
output use the same UI-like matrix shape: `section/group/name/note` on the Y
axis and batch numbers on the X axis.

```powershell
uv run python dpu_ib_sim.py --batch examples/batch_cases.csv `
  --base examples/golden.yaml `
  --out examples/batch_result.csv
```

Input shape:

```csv
section,group,name,note,batch_1,batch_2,batch_3
input,meta,case_id,,golden_120,system_layer_variant,fps_144_rotation
input,meta,description,,Baseline golden case,System and layer override case,144 Hz with layer overrides
input,panel,panel.panel_w,px,1080,1440,1080
input,panel,panel.fps,Hz,120,120,144
input,system,system.PTW,ratio,0.3,0.45,0.3
input,system,system.max_bus_port_BW_MBs,MB/s,5500,9000,5500
input,layers[0],layers[0].fmt,,YUV_8B,RGB_4B,YUV_8B
input,layers[0],layers[0].rotation,,True,False,True
input,layers[1],layers[1].hdr,,False,True,False
```

Output shape:

```csv
section,group,name,note,batch_1,batch_2,batch_3
input,meta,case_id,,golden_120,system_layer_variant,fps_144_rotation
summary,term,DPU_IB_MBps,MB/s,3735.5845959183675,5273.690111999998,4317.872831999999
summary,term,binding_term,,IB_outfifo_preload,IB_streaming,IB_streaming
summary,status,status,,ok,ok,ok
breakdown,timing,T_line_ns,ns,3472.22,3170.98,2893.52
breakdown,shared,MO_buf_bytes,B,8192,6144,8192
breakdown,clock,DPU_ACLK_MHz,MHz; F6/F7; binding,245.54,375.00,245.54
breakdown,term,IB_streaming_MBps,MB/s,3598.23,5273.69,4317.87
```

Sample files:

```text
examples/batch_cases.csv
examples/batch_result.csv
```

Input rules:

- Each batch column is one case.
- Only rows with `section=input` are used as batch inputs. The output preserves
  those input rows, then appends `summary` and `breakdown` sections.
- Metadata names such as `case_id` and `description` are preserved as input rows.
- Override names use the same parameter path syntax as sweep. Supported
  groups include `panel.*`, `system.*`, and existing base-config layer indices
  such as `layers[0].src_w`, `layers[1].fmt`, and `layers[2].rotation`.
- Empty cells are ignored, so the base YAML value is kept. For optional
  resolution fields this means the existing `panel default` fallback still
  applies.
- Layer rows override the existing layers from the base YAML. To compare cases
  with different layer settings, put the maximum layer set in the base YAML and
  override the fields needed per batch column.
- Invalid batch columns do not stop the batch. The batch column is written with
  `status=error` and an `error` message, then the next row continues.
- Output `breakdown` rows follow the same model values as the GUI Breakdown tab:
  timing, shared, streaming, clock, rotation, per-layer rows, and final term rows.

## GUI Usage

The GUI is split into an editable left panel and read-only result tabs on the
right.

- Drag the splitter between the left input panel and right output area to resize
  the sidebar.
- Edit panel/system fields or layer rows, then wait for debounce or press
  `Calculate`.
- The result card shows all three IB terms and highlights the `MAX` term that
  becomes `DPU_IB`.
- The Breakdown tab groups rows by timing/shared/streaming/rotation/layer/term,
  color-codes those groups, and displays units in the note column.
- The Formula tab renders each formula and shows live substituted values in
  tables.
- The Sweep tab supports:
  - 1D line plots
  - 2D heatmaps when `2D` is checked
  - CSV export for the computed grid

## Configuration

Config files are YAML with three top-level sections:

```yaml
system:
  PTW: 0.3
  MO_derating: 0.3
  MO_entries: 64
  MO_entry_bytes: 64
  OF_lines: 2
  margin: 1.13
  bus_width_B: 32
  bus_util: 0.70
  max_bus_port_BW_MBs: 5500
  ppc_comp: 4
  ppc_scaler: 4
  ppc_outfifo: 2
  ppc_pbld: 2
  ppc_ai_scaler: 2
  ppc_wb: 2
  vtap_scaler: 4
  vtap_outfifo: 2
  stream_clk_overhead: 1.13
  ptw_group_mode: dpuf0
  ai_scaler_enabled: false
  wb_enabled: false
  dpuf_xres: null
  dpuf_yres: null
  dpu_xres: null
  dpu_yres: null
  wb_xres: 0
  wb_yres: 0
  overlay_count_overrides: {}

panel:
  panel_w: 1080
  panel_h: 2340
  fps: 120
  disp_bpp: 30
  vbp: 28
  vfp: 28
  vsa: 4
  dsc_mode: 2slice_dual
  dst_y: 0

layers:
  - name: L1_Camera
    fmt: YUV_8B
    src_w: 3840
    src_h: 2160
    dst_w: null
    dst_h: null
    compressed: false
    compression_mode: sajc
    hdr: false
    scaling: false
    scale_v: 1.0
    rotation: true
    dpuf: 0
    stream_coeff: 3.0
```

When `dpuf_xres/dpuf_yres` or `dpu_xres/dpu_yres` are `null`, the simulator uses
`panel_w/panel_h` as the effective resolution. The Breakdown and Formula tabs
show this as `panel default` so the assumption is visible.

See `examples/golden.yaml` for the complete golden input.

## Formula Notes

Units:

- lengths: px
- data: bytes
- time: ns
- bandwidth: MB/s

Clock selection:

```text
DPU_ACLK = max(ACLK1, ACLK2, ACLK3, ACLK4, ACLK5, ACLK6)
```

Final selection:

```text
DPU_IB = max(IB_rotation_preload, IB_outfifo_preload, IB_streaming)
```

Important flagged assumptions from the source specification:

- `F1`: streaming uses a single effective per-layer `stream_coeff`.
- `F2`: PTW group mode is configurable; default is `dpuf0` to match the golden
  example.
- `F4`: `YUV_10B` bpp is modeled as 15 until packing is confirmed.
- `F5`: rotation preload reuses outfifo `rot_init_data`; the source document
  notes a small unresolved mismatch.
- `F6`: `max_bus_port_BW_MBs` defaults to 5500 MB/s until the real silicon value
  is confirmed.
- `F7`: overlay counts are auto-derived per DPUF unless
  `overlay_count_overrides` is provided.

## Verification

Run the test suite:

```powershell
uv run pytest
```

Current verified coverage includes:

- golden timing/intermediate values
- golden computed `DPU_ACLK` ACLK1..6 and binding row
- golden outfifo preload
- golden streaming
- computed-clock rotation preload path
- 1D and 2D sweep behavior
- YAML loader and CLI CSV output
- key GUI structure checks for sidebar, result card, table formatting, formula
  live-value tables, and sweep controls

## Packaging

The project is structured as a single executable script. A one-file build can be
created with PyInstaller after installing the build extra:

```powershell
uv run --extra gui --extra build pyinstaller --onefile dpu_ib_sim.py
```

The GUI build can be large because PySide6 and Qt WebEngine are substantial
runtime dependencies.

## Development Notes

- Keep calculation logic UI-free. `solve()` and `sweep()` must not import Qt or
  matplotlib.
- Add regression tests before changing engine behavior.
- Do not add hidden silicon constants. Put constants in `SystemConfig` or YAML.
- Keep generated files such as `.venv`, caches, `dist`, `build`, and sweep CSVs
  out of git.
