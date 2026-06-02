import csv
import subprocess
import sys
from pathlib import Path

import pytest

import inspect

import dpu_ib_sim
from dpu_ib_sim import (
    breakdown_rows,
    default_golden_config,
    dpu_aclk,
    load_config,
    result_delta_summary,
    solve,
    sweep,
    term_summary_rows,
)


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

    assert result.terms["IB_rotation_preload_MBps"] == pytest.approx(3698.0, abs=25.0)
    assert "F3" not in result.flags
    assert "F6" in result.flags
    assert "F7" in result.flags
    assert result.dpu_ib_MBps == pytest.approx(3735.58, abs=0.5)
    assert result.binding_term == "IB_outfifo_preload"


def test_dpu_aclk_golden_breakdown_and_binding():
    cfg = default_golden_config()

    clock = dpu_aclk(cfg)
    result = solve(cfg)

    assert clock["ACLK1_MHz"] == pytest.approx(85.7, abs=0.2)
    assert clock["ACLK2_MHz"] == pytest.approx(88.8, abs=0.5)
    assert clock["ACLK3_MHz"] == pytest.approx(245.5, abs=1.0)
    assert clock["ACLK4_MHz"] == pytest.approx(171.3, abs=0.5)
    assert clock["ACLK5_MHz"] == 0
    assert clock["ACLK6_MHz"] == 0
    assert clock["DPU_ACLK_MHz"] == pytest.approx(245.5, abs=1.0)
    assert clock["aclk_binding"] == "ACLK3"
    assert result.clock["DPU_ACLK_MHz"] == pytest.approx(clock["DPU_ACLK_MHz"], abs=0.01)
    assert clock["resolution_source"]["dpuf"] == "panel default"
    assert clock["resolution_source"]["dpu"] == "panel default"
    assert result.rotation["pipeline_latency_cycles"] == pytest.approx(5130)
    assert result.rotation["pipeline_latency_lines"] == pytest.approx(6.0, abs=0.1)
    assert result.rotation["tx_allow_lines"] == pytest.approx(50.0, abs=0.2)
    assert result.terms["IB_rotation_preload_MBps"] is not None


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


def test_batch_calculate_applies_csv_overrides_and_skips_blank_cells(tmp_path):
    csv_path = tmp_path / "cases.csv"
    csv_path.write_text(
        "\n".join(
            [
                "case_id,panel.fps,system.dpuf_xres,system.dpuf_yres,system.dpu_xres,system.dpu_yres,layers[0].src_w",
                "baseline,120,,,,,",
                "small-dpuf,144,640,480,640,480,1920",
            ]
        ),
        encoding="utf-8",
    )
    base = default_golden_config()

    rows = dpu_ib_sim.batch_calculate(base, csv_path)

    assert len(rows) == 2
    assert rows[0]["case_id"] == "baseline"
    assert rows[0]["status"] == "ok"
    assert rows[0]["panel.fps"] == "120"
    assert rows[0]["system.dpuf_xres"] == ""
    assert rows[0]["DPU_IB_MBps"] == pytest.approx(3735.5846, abs=0.001)
    assert rows[0]["DPU_ACLK_MHz"] == pytest.approx(245.5357, abs=0.001)
    assert rows[0]["binding_term"] == "IB_outfifo_preload"
    assert rows[0]["error"] == ""
    assert rows[1]["case_id"] == "small-dpuf"
    assert rows[1]["status"] == "ok"
    assert rows[1]["panel.fps"] == "144"
    assert rows[1]["layers[0].src_w"] == "1920"
    assert rows[1]["DPU_IB_MBps"] != pytest.approx(rows[0]["DPU_IB_MBps"], abs=0.001)
    assert base.layers[0].src_w == 3840


def test_batch_calculate_accepts_transposed_input_layout(tmp_path):
    csv_path = tmp_path / "cases.csv"
    csv_path.write_text(
        "\n".join(
            [
                "section,group,name,note,batch_1,batch_2",
                "input,meta,case_id,,baseline,fps_144",
                "input,meta,description,,Golden baseline,High refresh",
                "input,panel,panel.fps,,120,144",
                "input,system,system.dpuf_xres,panel default,,640",
                "input,system,system.dpuf_yres,panel default,,480",
                "input,system,system.dpu_xres,panel default,,640",
                "input,system,system.dpu_yres,panel default,,480",
            ]
        ),
        encoding="utf-8",
    )

    rows = dpu_ib_sim.batch_calculate(default_golden_config(), csv_path)

    assert len(rows) == 2
    assert rows[0]["case_id"] == "baseline"
    assert rows[0]["description"] == "Golden baseline"
    assert rows[0]["panel.fps"] == "120"
    assert rows[0]["system.dpuf_xres"] == ""
    assert rows[0]["status"] == "ok"
    assert rows[0]["DPU_IB_MBps"] == pytest.approx(3735.5846, abs=0.001)
    assert rows[1]["case_id"] == "fps_144"
    assert rows[1]["description"] == "High refresh"
    assert rows[1]["panel.fps"] == "144"
    assert rows[1]["system.dpuf_xres"] == "640"
    assert rows[1]["binding_term"] == "IB_rotation_preload"


def test_batch_calculate_varies_system_and_layer_configuration_per_batch(tmp_path):
    csv_path = tmp_path / "cases.csv"
    csv_path.write_text(
        "\n".join(
            [
                "section,group,name,note,batch_1,batch_2",
                "input,meta,case_id,,golden,layer_system_variant",
                "input,panel,panel.fps,,120,120",
                "input,system,system.PTW,,0.3,0.45",
                "input,system,system.MO_entries,,64,96",
                "input,system,system.max_bus_port_BW_MBs,MB/s,5500,9000",
                "input,system,system.ai_scaler_enabled,,False,True",
                "input,system,system.wb_enabled,,False,True",
                "input,system,system.wb_xres,px,0,1920",
                "input,system,system.wb_yres,px,0,1080",
                "input,layers[0],layers[0].name,,L1_Camera,L1_Camera_1080p",
                "input,layers[0],layers[0].fmt,,YUV_8B,RGB_4B",
                "input,layers[0],layers[0].src_w,px,3840,1920",
                "input,layers[0],layers[0].src_h,px,2160,1080",
                "input,layers[0],layers[0].compressed,,False,True",
                "input,layers[0],layers[0].scaling,,False,True",
                "input,layers[0],layers[0].scale_v,,1.0,1.5",
                "input,layers[0],layers[0].rotation,,True,False",
                "input,layers[0],layers[0].dpuf,,0,1",
                "input,layers[0],layers[0].stream_coeff,,3.0,1.5",
                "input,layers[1],layers[1].name,,L5_UI,L5_UI_HDR",
                "input,layers[1],layers[1].fmt,,RGB_4B,RGB_8B_FP16",
                "input,layers[1],layers[1].src_w,px,1080,1440",
                "input,layers[1],layers[1].src_h,px,2340,2560",
                "input,layers[1],layers[1].hdr,,False,True",
                "input,layers[1],layers[1].stream_coeff,,2.0,2.5",
            ]
        ),
        encoding="utf-8",
    )

    rows = dpu_ib_sim.batch_calculate(default_golden_config(), csv_path)

    assert [row["case_id"] for row in rows] == ["golden", "layer_system_variant"]
    assert rows[0]["status"] == "ok"
    assert rows[1]["status"] == "ok"
    assert rows[1]["system.PTW"] == "0.45"
    assert rows[1]["system.max_bus_port_BW_MBs"] == "9000"
    assert rows[1]["system.ai_scaler_enabled"] == "True"
    assert rows[1]["system.wb_enabled"] == "True"
    assert rows[1]["layers[0].name"] == "L1_Camera_1080p"
    assert rows[1]["layers[0].fmt"] == "RGB_4B"
    assert rows[1]["layers[0].rotation"] == "False"
    assert rows[1]["layers[1].hdr"] == "True"
    assert rows[1]["DPU_IB_MBps"] != pytest.approx(rows[0]["DPU_IB_MBps"], abs=0.001)
    assert rows[1]["DPU_ACLK_MHz"] != pytest.approx(rows[0]["DPU_ACLK_MHz"], abs=0.001)


def test_batch_calculate_marks_row_errors_and_continues(tmp_path):
    csv_path = tmp_path / "cases.csv"
    csv_path.write_text(
        "\n".join(
            [
                "case_id,panel.fps,panel.dsc_mode",
                "bad,120,unknown_mode",
                "good,120,2slice_dual",
            ]
        ),
        encoding="utf-8",
    )

    rows = dpu_ib_sim.batch_calculate(default_golden_config(), csv_path)

    assert [row["case_id"] for row in rows] == ["bad", "good"]
    assert rows[0]["status"] == "error"
    assert "unknown dsc_mode" in rows[0]["error"]
    assert rows[0]["DPU_IB_MBps"] == ""
    assert rows[1]["status"] == "ok"
    assert rows[1]["error"] == ""
    assert rows[1]["DPU_IB_MBps"] == pytest.approx(3735.5846, abs=0.001)


def test_write_batch_csv_matches_ui_breakdown_layout(tmp_path):
    in_path = tmp_path / "cases.csv"
    in_path.write_text(
        "\n".join(
            [
                "section,group,name,note,batch_1,batch_2",
                "input,meta,case_id,,baseline,fps_144",
                "input,panel,panel.fps,,120,144",
                "input,system,system.PTW,,0.3,0.3",
            ]
        ),
        encoding="utf-8",
    )
    rows = dpu_ib_sim.batch_calculate(default_golden_config(), in_path)
    out_path = tmp_path / "batch_result.csv"

    dpu_ib_sim.write_batch_csv(rows, out_path)

    with out_path.open(newline="", encoding="utf-8") as fp:
        csv_rows = list(csv.DictReader(fp))
    assert csv_rows[0] == {
        "section": "input",
        "group": "meta",
        "name": "case_id",
        "note": "",
        "batch_1": "baseline",
        "batch_2": "fps_144",
    }
    by_row = {(row["section"], row["group"], row["name"]): row for row in csv_rows}
    assert by_row[("summary", "term", "DPU_IB_MBps")]["note"] == "MB/s"
    assert float(by_row[("summary", "term", "DPU_IB_MBps")]["batch_1"]) == pytest.approx(3735.58, abs=0.01)
    assert by_row[("summary", "term", "binding_term")]["batch_1"] == "IB_outfifo_preload"
    assert by_row[("summary", "status", "status")]["batch_2"] == "ok"
    assert by_row[("breakdown", "timing", "T_line_ns")]["note"] == "ns"
    assert by_row[("breakdown", "shared", "MO_buf_bytes")]["note"] == "B"
    assert by_row[("breakdown", "clock", "DPU_ACLK_MHz")]["note"] == "MHz; F6/F7; binding"
    assert by_row[("breakdown", "term", "IB_streaming_MBps")]["note"] == "MB/s"


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


def test_cli_batch_writes_transposed_csv_from_imported_cases(tmp_path):
    in_path = tmp_path / "cases.csv"
    in_path.write_text(
        "\n".join(
            [
                "section,group,name,note,batch_1,batch_2",
                "input,meta,case_id,,baseline,small-dpuf",
                "input,panel,panel.fps,,120,120",
                "input,system,system.dpuf_xres,panel default,,640",
                "input,system,system.dpuf_yres,panel default,,480",
                "input,system,system.dpu_xres,panel default,,640",
                "input,system,system.dpu_yres,panel default,,480",
            ]
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "batch_result.csv"

    completed = subprocess.run(
        [
            sys.executable,
            "dpu_ib_sim.py",
            "--batch",
            str(in_path),
            "--base",
            "examples/golden.yaml",
            "--out",
            str(out_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "wrote 2 batch rows" in completed.stdout
    with out_path.open(newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    matrix = {(row["section"], row["group"], row["name"]): row for row in rows}
    assert rows[0] == {
        "section": "input",
        "group": "meta",
        "name": "case_id",
        "note": "",
        "batch_1": "baseline",
        "batch_2": "small-dpuf",
    }
    assert matrix[("summary", "status", "status")]["batch_1"] == "ok"
    assert matrix[("summary", "term", "binding_term")]["batch_1"] == "IB_outfifo_preload"
    assert matrix[("breakdown", "clock", "DPU_ACLK_MHz")]["batch_2"] == matrix[
        ("breakdown", "clock", "DPU_ACLK_MHz")
    ]["batch_1"]


def test_breakdown_rows_include_group_colors_and_units_in_notes():
    result = solve(default_golden_config())

    rows = breakdown_rows(result)

    assert rows
    assert {row["group"] for row in rows} >= {"timing", "shared", "streaming", "clock", "rotation", "term"}
    assert all(row["background"] for row in rows)
    assert next(row for row in rows if row["name"] == "DPU_ACLK_MHz")["group"] == "clock"
    assert "panel default" in next(row for row in rows if row["name"] == "dpuf_resolution_source")["note"]
    assert "panel default" in next(row for row in rows if row["name"] == "dpu_resolution_source")["note"]
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
    assert "<td>DPU_ACLK</td>" in html
    assert "<td>245.54</td>" in html
    assert "MO_buf_bytes=8192, Total_pipeline_data_bytes" not in html


def test_gui_system_panel_uses_computed_aclk_fields():
    source = inspect.getsource(dpu_ib_sim.run_gui)

    assert '"DPU_ACLK_MHz",' not in source
    assert '"max_bus_port_BW_MBs"' in source
    assert '"bus_width_B"' in source
    assert '"dpuf_xres"' in source


def test_gui_optional_fields_show_fallback_placeholder_and_tooltip():
    source = inspect.getsource(dpu_ib_sim.run_gui)

    assert "setPlaceholderText(\"panel default\")" in source
    assert "setToolTip(\"Blank uses panel_w/panel_h fallback\")" in source
    assert "auto default" in source


def test_result_delta_summary_shows_unchanged_final_with_changed_clock_terms():
    previous = solve(default_golden_config())
    cfg = default_golden_config()
    cfg.system.dpuf_xres = 640
    cfg.system.dpuf_yres = 480
    cfg.system.dpu_xres = 640
    cfg.system.dpu_yres = 480
    current = solve(cfg)

    summary = result_delta_summary(current, previous)
    rows = breakdown_rows(current, previous)
    terms = term_summary_rows(current, previous)

    assert summary["dpu_ib_delta"] == pytest.approx(0)
    assert summary["dpu_aclk_delta"] == pytest.approx(0)
    assert summary["binding_changed"] is False
    assert {"ACLK1", "ACLK4"} <= set(summary["changed_clock_terms"])
    assert summary["headline"] == "DPU_IB unchanged; DPU_ACLK unchanged; changed clock terms: ACLK1, ACLK4"
    assert next(row for row in rows if row["name"] == "ACLK1_MHz")["delta"].startswith("-75.26")
    assert next(row for row in rows if row["name"] == "DPU_ACLK_MHz")["delta"] == "0.00"
    assert next(term for term in terms if term["key"] == "IB_outfifo_preload")["delta_display"] == "0.00"


def test_gui_summary_and_breakdown_include_baseline_delta_fields():
    source = inspect.getsource(dpu_ib_sim.run_gui)

    assert "self.baseline_result" in source
    assert "self.delta_label" in source
    assert "Set baseline" in source
    assert "self.set_baseline" in source
    assert "[\"group\", \"name\", \"model\", \"delta\", \"measured\", \"note\"]" in source


def test_gui_bolds_nonzero_delta_cells_only():
    source = inspect.getsource(dpu_ib_sim.run_gui)

    assert "item.setFont(delta_font)" in source
    assert "text not in {\"\", \"0.00\", \"same\"}" in source


def test_gui_exposes_optional_2d_sweep_controls_and_heatmap():
    source = inspect.getsource(dpu_ib_sim.run_gui)

    assert "self.sweep_2d" in source
    assert "self.sweep_param2" in source
    assert "param2 if self.sweep_2d.isChecked() else None" in source
    assert "imshow(" in source
