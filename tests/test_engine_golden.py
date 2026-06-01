import csv
import subprocess
import sys
from pathlib import Path

import pytest

import inspect

import dpu_ib_sim
from dpu_ib_sim import breakdown_rows, default_golden_config, load_config, solve, sweep, term_summary_rows


def test_golden_outfifo_streaming_and_binding_match_spec():
    result = solve(default_golden_config())

    assert result.timing["V_total"] == 2400
    assert result.timing["T_line_ns"] == pytest.approx(3472.2222, abs=0.0001)
    assert result.timing["V_blank_time_ns"] == pytest.approx(194444.44, abs=0.01)

    assert result.shared["MO_buf_bytes"] == 8192
    assert result.per_layer["L1_Camera"]["rot_init_data_bytes"] == pytest.approx(296228.57, abs=0.01)
    assert result.per_layer["L9_PIP"]["rot_init_data_bytes"] == pytest.approx(197485.71, abs=0.01)
    assert result.per_layer["L5_UI"]["comp_data_bytes"] == pytest.approx(34560, abs=1)
    assert result.shared["DSC_data_bytes"] == pytest.approx(10125, abs=1)
    assert result.shared["DSIM_data_bytes"] == pytest.approx(4050, abs=1)
    assert result.shared["OUTFIFO_data_bytes"] == pytest.approx(8100, abs=1)
    assert result.shared["Total_preload_data_bytes"] == pytest.approx(558741, abs=1)

    assert result.terms["IB_outfifo_preload_MBps"] == pytest.approx(3735.58, abs=0.5)
    assert result.streaming["dpuf_clk_MHz"] == pytest.approx(342.69, abs=0.01)
    assert result.per_layer["L1_Camera"]["streaming_MBps"] == pytest.approx(1028.06, abs=0.01)
    assert result.per_layer["L5_UI"]["streaming_MBps"] == pytest.approx(685.38, abs=0.01)
    assert result.per_layer["L9_PIP"]["streaming_MBps"] == pytest.approx(1370.75, abs=0.01)
    assert result.streaming["PTW_MBps"] == pytest.approx(514.03, abs=0.01)
    assert result.terms["IB_streaming_MBps"] == pytest.approx(3598.23, abs=0.5)

    assert result.terms["IB_rotation_preload_MBps"] is None
    assert "F3" in result.flags
    assert result.dpu_ib_MBps == pytest.approx(3735.58, abs=0.5)
    assert result.binding_term == "IB_outfifo_preload"


def test_rotation_preload_uses_configured_aclk_without_hidden_default():
    cfg = default_golden_config()
    cfg.system.DPU_ACLK_MHz = 600.0

    result = solve(cfg)

    expected_latency_ns = (4680 + 1350 + 540 + 1080) * 1000 / 600.0
    expected_tx_allow_ns = 194444.44444444444 - expected_latency_ns
    expected_rot_data = 296228.5714285714 + 197485.7142857143
    expected_rotation = expected_rot_data / expected_tx_allow_ns * 1000 * 1.3

    assert result.rotation["pipeline_latency_cycles"] == pytest.approx(7650)
    assert result.rotation["tx_allow_time_ns"] == pytest.approx(expected_tx_allow_ns, abs=0.01)
    assert result.terms["IB_rotation_preload_MBps"] == pytest.approx(expected_rotation, abs=0.01)
    assert "F3" not in result.flags
    assert "F5" in result.flags


def test_sweep_varies_only_selected_parameter_and_keeps_baseline_unchanged():
    cfg = default_golden_config()

    rows = sweep(cfg, "panel.fps", 60, 120, 60)

    assert [row["panel.fps"] for row in rows] == [60, 120]
    assert cfg.panel.fps == 120
    assert rows[0]["DPU_IB_MBps"] != rows[1]["DPU_IB_MBps"]
    assert rows[1]["binding_term"] == "IB_outfifo_preload"


def test_2d_sweep_varies_parameter_pair_and_keeps_baseline_unchanged():
    cfg = default_golden_config()

    rows = sweep(cfg, "panel.fps", 60, 120, 60, "layers[0].src_h", 1080, 2160, 1080)

    assert len(rows) == 4
    assert {(row["panel.fps"], row["layers[0].src_h"]) for row in rows} == {
        (60, 1080),
        (60, 2160),
        (120, 1080),
        (120, 2160),
    }
    assert cfg.panel.fps == 120
    assert cfg.layers[0].src_h == 2160
    assert all("DPU_IB_MBps" in row for row in rows)
    assert all("binding_term" in row for row in rows)


def test_yaml_loader_and_cli_sweep_write_csv(tmp_path):
    cfg = load_config(Path("examples/golden.yaml"))
    assert cfg.panel.panel_w == 1080
    assert len(cfg.layers) == 3

    out_path = tmp_path / "sweep.csv"
    completed = subprocess.run(
        [
            sys.executable,
            "dpu_ib_sim.py",
            "--sweep",
            "examples/golden.yaml",
            "--param",
            "panel.fps",
            "--start",
            "120",
            "--stop",
            "120",
            "--step",
            "1",
            "--out",
            str(out_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "wrote" in completed.stdout
    with out_path.open(newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    assert len(rows) == 1
    assert rows[0]["binding_term"] == "IB_outfifo_preload"


def test_breakdown_rows_include_group_colors_and_units_in_notes():
    result = solve(default_golden_config())

    rows = breakdown_rows(result)

    assert rows
    assert {row["group"] for row in rows} >= {"timing", "shared", "streaming", "rotation", "term"}
    assert all(row["background"] for row in rows)
    assert next(row for row in rows if row["name"] == "T_line_ns")["note"] == "ns"
    assert next(row for row in rows if row["name"] == "MO_buf_bytes")["note"] == "B"
    assert "MB/s" in next(row for row in rows if row["name"] == "IB_outfifo_preload_MBps")["note"]
    assert "F1/F2" in next(row for row in rows if row["name"] == "PTW_MBps")["note"]


def test_gui_uses_splitter_for_resizable_sidebar():
    source = inspect.getsource(dpu_ib_sim.run_gui)

    assert "QSplitter" in source
    assert "setStretchFactor" in source


def test_term_summary_marks_all_terms_and_the_max_binding():
    result = solve(default_golden_config())

    terms = term_summary_rows(result)

    assert [term["key"] for term in terms] == [
        "IB_rotation_preload",
        "IB_outfifo_preload",
        "IB_streaming",
    ]
    assert [term["is_binding"] for term in terms] == [False, True, False]
    assert next(term for term in terms if term["is_binding"])["label"] == "MAX"


def test_breakdown_header_uses_interactive_resize_mode():
    source = inspect.getsource(dpu_ib_sim.run_gui)

    assert "setSectionResizeMode(QHeaderView.Interactive)" in source
    assert "self.breakdown.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)" not in source


def test_gui_centers_breakdown_cells_and_spaces_summary_label():
    source = inspect.getsource(dpu_ib_sim.run_gui)

    assert "item.setTextAlignment(Qt.AlignCenter)" in source
    assert "&nbsp;&nbsp;&nbsp;max term:" in source


def test_formula_live_values_render_as_table_rows():
    html = dpu_ib_sim.build_formula_html(solve(default_golden_config()))

    assert "<h4>Live values</h4>" in html
    assert "class='live-values'" in html
    assert "<td>MO_buf_bytes</td>" in html
    assert "<td>8192</td>" in html
    assert "<td>B</td>" in html
    assert "<td>IB_outfifo_preload</td>" in html
    assert "<td>3735.58</td>" in html
    assert "MO_buf_bytes=8192, Total_pipeline_data_bytes" not in html


def test_gui_exposes_optional_2d_sweep_controls_and_heatmap():
    source = inspect.getsource(dpu_ib_sim.run_gui)

    assert "self.sweep_2d" in source
    assert "self.sweep_param2" in source
    assert "param2 if self.sweep_2d.isChecked() else None" in source
    assert "imshow(" in source
