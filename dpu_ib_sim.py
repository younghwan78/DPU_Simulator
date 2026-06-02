from __future__ import annotations

import argparse
import copy
import csv
import html
import math
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only outside the uv env
    yaml = None


@dataclass(frozen=True)
class FormatInfo:
    bpp: float
    read_line_uncomp: float
    read_line_comp: float
    lmc: float


@dataclass(frozen=True)
class DscInfo:
    slice_factor: float
    ppc_dsc: float
    ppc_dsim: float


FORMAT_LUT: dict[str, FormatInfo] = {
    "RGB_8B_FP16": FormatInfo(64, 8, 8, 4),
    "RGB_4B": FormatInfo(32, 16, 8, 8),
    "RGB_2B": FormatInfo(16, 32, 16, 8),
    "YUV_8B": FormatInfo(12, 64, 32, 4),
    "YUV_10B": FormatInfo(15, 32, 32, 4),
}

DSC_LUT: dict[str, DscInfo] = {
    "1slice_single": DscInfo(1.0, 1, 1),
    "2slice_single": DscInfo(2.0, 1, 1),
    "2slice_dual": DscInfo(2.5, 2, 2),
    "4slice_dual": DscInfo(3.5, 2, 2),
}


@dataclass
class LayerConfig:
    name: str
    src_w: int
    src_h: int
    fmt: str
    dst_w: int | None = None
    dst_h: int | None = None
    compressed: bool = False
    compression_mode: str = "sajc"
    hdr: bool = False
    scaling: bool = False
    scale_v: float = 1.0
    rotation: bool = False
    dpuf: int = 0
    stream_coeff: float = 1.0


@dataclass
class PanelConfig:
    panel_w: int
    panel_h: int
    fps: float
    disp_bpp: float
    vbp: int
    vfp: int
    vsa: int
    dsc_mode: str
    dst_y: int = 0


@dataclass
class SystemConfig:
    PTW: float = 0.3
    MO_derating: float = 0.3
    MO_entries: int = 64
    MO_entry_bytes: int = 64
    OF_lines: int = 2
    margin: float = 1.13
    bus_width_B: float = 32
    bus_util: float = 0.70
    max_bus_port_BW_MBs: float = 5500
    ppc_comp: float = 4
    ppc_scaler: float = 4
    ppc_outfifo: float = 2
    ppc_pbld: float = 2
    ppc_ai_scaler: float = 2
    ppc_wb: float = 2
    vtap_scaler: float = 4
    vtap_outfifo: float = 2
    stream_clk_overhead: float = 1.13
    ptw_group_mode: str = "dpuf0"
    ai_scaler_enabled: bool = False
    wb_enabled: bool = False
    dpuf_xres: int | None = None
    dpuf_yres: int | None = None
    dpu_xres: int | None = None
    dpu_yres: int | None = None
    wb_xres: int = 0
    wb_yres: int = 0
    overlay_count_overrides: dict[Any, Any] = field(default_factory=dict)


@dataclass
class SimulatorConfig:
    system: SystemConfig
    panel: PanelConfig
    layers: list[LayerConfig] = field(default_factory=list)


@dataclass
class Result:
    timing: dict[str, float]
    per_layer: dict[str, dict[str, float | str | bool | None]]
    shared: dict[str, float]
    streaming: dict[str, float | str]
    clock: dict[str, float | str | dict[str, Any]]
    rotation: dict[str, float | str | None]
    terms: dict[str, float | None]
    dpu_ib_MBps: float
    binding_term: str | None
    flags: dict[str, str]
    formula_values: dict[str, float | str | None]


FORMULAE: list[dict[str, Any]] = [
    {
        "key": "DPU_ACLK",
        "title": "DPU ACLK",
        "latex": r"DPU\_ACLK = \max(ACLK_1, ACLK_2, ACLK_3, ACLK_4, ACLK_5, ACLK_6)",
        "symbols": {
            "ACLK1": ("per-layer throughput condition", "MHz", "dpu_aclk()"),
            "ACLK2": ("latency condition", "MHz", "dpu_aclk()"),
            "ACLK3": ("bus throughput condition", "MHz", "SystemConfig.max_bus_port_BW_MBs"),
            "ACLK4": ("PBLD condition", "MHz", "SystemConfig.ppc_pbld"),
            "ACLK5": ("AI scaler condition", "MHz", "SystemConfig.ai_scaler_enabled"),
            "ACLK6": ("writeback condition", "MHz", "SystemConfig.wb_enabled"),
        },
        "bind_keys": [
            "ACLK1_MHz",
            "ACLK2_MHz",
            "ACLK3_MHz",
            "ACLK4_MHz",
            "ACLK5_MHz",
            "ACLK6_MHz",
            "DPU_ACLK_MHz",
            "aclk_binding",
            "dpuf_xres_effective",
            "dpuf_yres_effective",
            "dpuf_resolution_source",
            "dpu_xres_effective",
            "dpu_yres_effective",
            "dpu_resolution_source",
        ],
    },
    {
        "key": "IB_outfifo_preload",
        "title": "Outfifo preload",
        "latex": r"IB_{outfifo} = \frac{MO_{buf} + \sum_i P_i + S}{T_{vblank}} \cdot (1 + PTW)",
        "symbols": {
            "MO_buf": ("active DPUF memory outstanding buffer", "B", "SystemConfig.MO_entries"),
            "P_i": ("per-layer pipeline preload data", "B", "LayerConfig + HW LUT"),
            "S": ("DSC + DSIM + OUTFIFO shared data", "B", "PanelConfig + SystemConfig"),
            "T_vblank": ("vertical blank time", "ns", "PanelConfig"),
            "PTW": ("page-table-walk ratio", "ratio", "SystemConfig.PTW"),
        },
        "bind_keys": ["MO_buf_bytes", "Total_pipeline_data_bytes", "Shared_data_bytes", "V_blank_time_ns", "PTW"],
    },
    {
        "key": "IB_streaming",
        "title": "Streaming",
        "latex": r"IB_{streaming} = \sum_i BW_i + PTW \cdot \sum_{i \in group} BW_i",
        "symbols": {
            "BW_i": ("per-layer streaming bandwidth", "MB/s", "LayerConfig.stream_coeff"),
            "group": ("PTW group", "set", "SystemConfig.ptw_group_mode"),
            "PTW": ("page-table-walk ratio", "ratio", "SystemConfig.PTW"),
        },
        "bind_keys": ["dpuf_clk_MHz", "stream_sum_MBps", "stream_ptw_MBps", "ptw_group_mode"],
    },
    {
        "key": "IB_rotation_preload",
        "title": "Rotation preload",
        "latex": r"IB_{rot} = \sum_{dpuf}\left(\sum_i \frac{rot\_init_i}{T_{allow}}\right)\cdot(1 + PTW)",
        "symbols": {
            "rot_init_i": ("per-rotation-layer initial data", "B", "LayerConfig + HW LUT"),
            "T_allow": ("available time minus pipeline latency", "ns", "dpu_aclk()"),
            "PTW": ("page-table-walk ratio", "ratio", "SystemConfig.PTW"),
        },
        "bind_keys": ["pipeline_latency_cycles", "tx_allow_time_ns", "rotation_data_bytes", "PTW"],
    },
]


def _format_info(fmt: str) -> FormatInfo:
    try:
        return FORMAT_LUT[fmt]
    except KeyError as exc:
        raise ValueError(f"unknown format: {fmt}") from exc


def _dsc_info(mode: str) -> DscInfo:
    try:
        return DSC_LUT[mode]
    except KeyError as exc:
        raise ValueError(f"unknown dsc_mode: {mode}") from exc


def default_golden_config() -> SimulatorConfig:
    return SimulatorConfig(
        system=SystemConfig(),
        panel=PanelConfig(
            panel_w=1080,
            panel_h=2340,
            fps=120,
            disp_bpp=30,
            vbp=28,
            vfp=28,
            vsa=4,
            dsc_mode="2slice_dual",
            dst_y=0,
        ),
        layers=[
            LayerConfig(
                name="L1_Camera",
                fmt="YUV_8B",
                src_w=3840,
                src_h=2160,
                compressed=False,
                scaling=False,
                scale_v=1.0,
                rotation=True,
                dpuf=0,
                stream_coeff=3.0,
            ),
            LayerConfig(
                name="L5_UI",
                fmt="RGB_4B",
                src_w=1080,
                src_h=2340,
                compressed=True,
                scaling=False,
                scale_v=1.0,
                rotation=False,
                dpuf=0,
                stream_coeff=2.0,
            ),
            LayerConfig(
                name="L9_PIP",
                fmt="YUV_8B",
                src_w=2560,
                src_h=1440,
                compressed=False,
                scaling=False,
                scale_v=1.0,
                rotation=True,
                dpuf=1,
                stream_coeff=4.0,
            ),
        ],
    )


def load_config(path: str | Path) -> SimulatorConfig:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load config files. Run `uv sync` first.")
    with Path(path).open(encoding="utf-8") as fp:
        raw = yaml.safe_load(fp) or {}
    return config_from_dict(raw)


def save_config(config: SimulatorConfig, path: str | Path) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required to save config files. Run `uv sync` first.")
    data = {
        "system": asdict(config.system),
        "panel": asdict(config.panel),
        "layers": [asdict(layer) for layer in config.layers],
    }
    with Path(path).open("w", encoding="utf-8") as fp:
        yaml.safe_dump(data, fp, sort_keys=False)


def config_from_dict(raw: dict[str, Any]) -> SimulatorConfig:
    system_data = dict(raw.get("system") or {})
    system_data.pop("DPU_ACLK_MHz", None)
    panel_data = raw.get("panel") or {}
    layers_data = raw.get("layers") or []
    system = SystemConfig(**system_data)
    panel = PanelConfig(**panel_data)
    layers = [LayerConfig(**layer) for layer in layers_data]
    return SimulatorConfig(system=system, panel=panel, layers=layers)


def timing(panel: PanelConfig) -> dict[str, float]:
    v_total = panel.panel_h + panel.vbp + panel.vfp + panel.vsa
    t_line_ns = 1e9 / (v_total * panel.fps)
    v_blank_time_ns = (panel.vbp + panel.vfp) * t_line_ns
    start_time_ns = panel.dst_y * t_line_ns
    return {
        "V_total": v_total,
        "T_line_ns": t_line_ns,
        "V_blank_time_ns": v_blank_time_ns,
        "start_time_ns": start_time_ns,
        "available_time_ns": v_blank_time_ns + start_time_ns,
    }


def _read_line(layer: LayerConfig, info: FormatInfo) -> float:
    return info.read_line_comp if layer.compressed else info.read_line_uncomp


def _per_layer_data(layer: LayerConfig, system: SystemConfig) -> dict[str, float | str | bool | None]:
    info = _format_info(layer.fmt)
    read_line = _read_line(layer, info)
    bytes_per_px = info.bpp / 8
    if system.MO_derating >= 1:
        raise ValueError("MO_derating must be less than 1.0")

    rot_init_data = 0.0
    if layer.rotation:
        rot_init_data = (layer.src_h * read_line * bytes_per_px) / (1 - system.MO_derating)

    comp_data = 0.0
    if layer.compressed:
        comp_data = info.lmc * layer.src_w * bytes_per_px

    scaler_data = 0.0
    if layer.scaling:
        scaler_data = system.vtap_scaler * layer.src_w * bytes_per_px

    return {
        "name": layer.name,
        "fmt": layer.fmt,
        "dst_w": layer.dst_w,
        "dst_h": layer.dst_h,
        "bpp": info.bpp,
        "read_line": read_line,
        "lmc": info.lmc,
        "compressed": layer.compressed,
        "compression_mode": layer.compression_mode,
        "hdr": layer.hdr,
        "scaling": layer.scaling,
        "rotation": layer.rotation,
        "dpuf": layer.dpuf,
        "scale_v": layer.scale_v,
        "stream_coeff": layer.stream_coeff,
        "rot_init_data_bytes": rot_init_data,
        "comp_data_bytes": comp_data,
        "scaler_data_bytes": scaler_data,
        "pipeline_data_bytes": rot_init_data + comp_data + scaler_data,
        "streaming_MBps": None,
        "rotation_demanded_MBps": None,
    }


def dpu_aclk(config: SimulatorConfig) -> dict[str, Any]:
    system = config.system
    panel = config.panel
    dsc = _dsc_info(panel.dsc_mode)
    time = timing(panel)
    per_layer = {layer.name: _per_layer_data(layer, system) for layer in config.layers}

    dpuf_uses_panel_default = system.dpuf_xres is None or system.dpuf_yres is None
    dpu_uses_panel_default = system.dpu_xres is None or system.dpu_yres is None
    dpuf_xres = system.dpuf_xres if system.dpuf_xres is not None else panel.panel_w
    dpuf_yres = system.dpuf_yres if system.dpuf_yres is not None else panel.panel_h
    dpu_xres = system.dpu_xres if system.dpu_xres is not None else panel.panel_w
    dpu_yres = system.dpu_yres if system.dpu_yres is not None else panel.panel_h
    dpuf_resol_clk = _resol_clk(dpuf_xres, dpuf_yres, panel.fps, system.margin)
    dpu_resol_clk = _resol_clk(dpu_xres, dpu_yres, panel.fps, system.margin)
    panel_resol_clk = _resol_clk(panel.panel_w, panel.panel_h, panel.fps, system.margin)
    wb_resol_clk = _resol_clk(system.wb_xres, system.wb_yres, panel.fps, system.margin)

    overlay_counts = _overlay_counts_by_dpuf(config)
    aclk1_layers: dict[str, float] = {}
    for layer in config.layers:
        h_ratio, v_ratio = _layer_scale_ratios(layer)
        ppc_tdm = _ppc_tdm(overlay_counts.get(layer.dpuf, _empty_overlay_counts()))
        value = dpuf_resol_clk * h_ratio * v_ratio / ppc_tdm
        aclk1_layers[layer.name] = value
    aclk1 = max(aclk1_layers.values(), default=0.0)

    aclk2_cycles = _aclk2_latency_cycles(config, per_layer, dsc)
    aclk2 = aclk2_cycles / time["available_time_ns"] * 1000 if time["available_time_ns"] > 0 else math.inf
    aclk3 = system.max_bus_port_BW_MBs / system.bus_width_B / system.bus_util
    aclk4 = dpu_resol_clk / system.ppc_pbld
    aclk5 = panel_resol_clk / system.ppc_ai_scaler if system.ai_scaler_enabled else 0.0
    aclk6 = wb_resol_clk / system.ppc_wb if system.wb_enabled and system.wb_xres and system.wb_yres else 0.0
    candidates = {
        "ACLK1": aclk1,
        "ACLK2": aclk2,
        "ACLK3": aclk3,
        "ACLK4": aclk4,
        "ACLK5": aclk5,
        "ACLK6": aclk6,
    }
    binding = max(candidates, key=lambda key: candidates[key])
    return {
        "ACLK1_MHz": aclk1,
        "ACLK2_MHz": aclk2,
        "ACLK3_MHz": aclk3,
        "ACLK4_MHz": aclk4,
        "ACLK5_MHz": aclk5,
        "ACLK6_MHz": aclk6,
        "DPU_ACLK_MHz": candidates[binding],
        "aclk_binding": binding,
        "dpuf_resol_clk_MHz": dpuf_resol_clk,
        "dpu_resol_clk_MHz": dpu_resol_clk,
        "panel_resol_clk_MHz": panel_resol_clk,
        "wb_resol_clk_MHz": wb_resol_clk,
        "ACLK1_layer_MHz": aclk1_layers,
        "ACLK2_latency_cycles": aclk2_cycles,
        "overlay_counts": overlay_counts,
        "dpuf_xres_effective": dpuf_xres,
        "dpuf_yres_effective": dpuf_yres,
        "dpu_xres_effective": dpu_xres,
        "dpu_yres_effective": dpu_yres,
        "resolution_source": {
            "dpuf": "panel default" if dpuf_uses_panel_default else "explicit config",
            "dpu": "panel default" if dpu_uses_panel_default else "explicit config",
        },
        "dpuf_resolution_source": "panel default" if dpuf_uses_panel_default else "explicit config",
        "dpu_resolution_source": "panel default" if dpu_uses_panel_default else "explicit config",
    }


def _resol_clk(xres: int | float, yres: int | float, fps: float, margin: float) -> float:
    return float(xres) * float(yres) * fps * margin / 1e6


def _layer_scale_ratios(layer: LayerConfig) -> tuple[float, float]:
    default_dst_w = layer.src_h if layer.rotation else layer.src_w
    default_dst_h = layer.src_w if layer.rotation else layer.src_h
    dst_w = layer.dst_w or default_dst_w
    dst_h = layer.dst_h or default_dst_h
    if dst_w <= 0 or dst_h <= 0:
        raise ValueError(f"layer {layer.name} dst_w/dst_h must be positive")
    if layer.rotation:
        return layer.src_h / dst_w, layer.src_w / dst_h
    return layer.src_w / dst_w, layer.src_h / dst_h


def _empty_overlay_counts() -> dict[str, int]:
    return {"hdr": 0, "sajc": 0, "sbwc": 0}


def _overlay_counts_by_dpuf(config: SimulatorConfig) -> dict[int, dict[str, int]]:
    counts: dict[int, dict[str, int]] = {}
    for layer in config.layers:
        dpuf_counts = counts.setdefault(layer.dpuf, _empty_overlay_counts())
        if layer.hdr:
            dpuf_counts["hdr"] += 1
        if layer.compressed:
            mode = layer.compression_mode.lower()
            if mode == "sbwc":
                dpuf_counts["sbwc"] += 1
            else:
                dpuf_counts["sajc"] += 1

    for raw_dpuf, override in config.system.overlay_count_overrides.items():
        dpuf = int(raw_dpuf)
        dpuf_counts = counts.setdefault(dpuf, _empty_overlay_counts())
        for key in ["hdr", "sajc", "sbwc"]:
            if key in override and override[key] is not None:
                dpuf_counts[key] = int(override[key])
    return counts


def _ppc_tdm(counts: dict[str, int]) -> float:
    hdr_ppc = 4 / counts["hdr"] if counts["hdr"] else 4
    sajc_count = counts["sajc"]
    if sajc_count in {0, 1, 2}:
        sajc_ppc = 4
    elif sajc_count == 3:
        sajc_ppc = 2.6
    else:
        sajc_ppc = 2
    sbwc_count = counts["sbwc"]
    if sbwc_count in {0, 1}:
        sbwc_ppc = 4
    else:
        sbwc_ppc = 2
    return min(hdr_ppc, sajc_ppc, sbwc_ppc)


def _latency_layers(config: SimulatorConfig) -> list[LayerConfig]:
    rotation_layers = [layer for layer in config.layers if layer.rotation]
    return rotation_layers or config.layers


def _aclk2_latency_cycles(
    config: SimulatorConfig,
    per_layer: dict[str, dict[str, float | str | bool | None]],
    dsc: DscInfo,
) -> float:
    system = config.system
    panel = config.panel
    rotator_latency = 0.0
    compression_latency = 0.0
    scaler_latency = 0.0
    for layer in _latency_layers(config):
        info = _format_info(layer.fmt)
        src_width_rot = layer.src_h
        if layer.rotation:
            rotator_latency = max(
                rotator_latency,
                float(per_layer[layer.name]["rot_init_data_bytes"]) / (system.bus_width_B * system.bus_util),
            )
        if layer.compressed:
            compression_latency = max(compression_latency, src_width_rot * info.lmc / system.ppc_comp)
        scaler_latency = max(scaler_latency, src_width_rot * system.vtap_scaler / system.ppc_scaler)

    dsc_cycles = panel.panel_w * dsc.slice_factor / dsc.ppc_dsc
    dsim_cycles = panel.panel_w / dsc.ppc_dsim
    return max(rotator_latency, compression_latency) + scaler_latency + dsc_cycles + dsim_cycles


def solve(config: SimulatorConfig) -> Result:
    if not config.layers:
        raise ValueError("at least one layer is required")

    system = config.system
    panel = config.panel
    dsc = _dsc_info(panel.dsc_mode)
    time = timing(panel)
    flags: dict[str, str] = {
        "F1": "Streaming uses a single effective per-layer stream_coeff.",
        "F2": f"Streaming PTW group mode is configurable; current mode is {system.ptw_group_mode}.",
        "F6": "max_bus_port_BW_MBs default is 5500 MB/s until the real silicon value is confirmed.",
        "F7": "Overlay counts are auto-derived by DPUF unless overlay_count_overrides is provided.",
    }

    per_layer = {layer.name: _per_layer_data(layer, system) for layer in config.layers}
    if any(layer.fmt == "YUV_10B" for layer in config.layers):
        flags["F4"] = "YUV_10B bpp is modeled as 15 until packing is confirmed."
    if any(layer.rotation for layer in config.layers):
        flags["F5"] = "Rotation path reuses outfifo rot_init_data; source document notes a data mismatch."

    active_dpufs = {layer.dpuf for layer in config.layers}
    mo_buf = len(active_dpufs) * system.MO_entries * system.MO_entry_bytes
    dsc_data = panel.panel_w * dsc.slice_factor * panel.disp_bpp / 8
    dsim_data = panel.panel_w * panel.disp_bpp / 8
    outfifo_data = system.OF_lines * panel.panel_w * panel.disp_bpp / 8
    shared_data = dsc_data + dsim_data + outfifo_data
    total_pipeline_data = sum(float(values["pipeline_data_bytes"]) for values in per_layer.values())
    total_preload_data = mo_buf + total_pipeline_data + shared_data
    outfifo_demanded = total_preload_data / time["V_blank_time_ns"] * 1000
    ib_outfifo = outfifo_demanded * (1 + system.PTW)

    dpuf_clk = panel.panel_w * panel.panel_h * panel.fps * system.stream_clk_overhead / 1e6
    stream_sum = 0.0
    stream_ptw_base = 0.0
    for layer in config.layers:
        layer_bw = dpuf_clk * layer.stream_coeff * layer.scale_v
        per_layer[layer.name]["streaming_MBps"] = layer_bw
        stream_sum += layer_bw
        if _layer_is_in_ptw_group(layer, system.ptw_group_mode):
            stream_ptw_base += layer_bw
    stream_ptw = system.PTW * stream_ptw_base
    ib_streaming = stream_sum + stream_ptw

    clock = dpu_aclk(config)
    rotation = _rotation_preload(config, per_layer, time, dsc, clock, flags)
    ib_rotation = rotation["IB_rotation_preload_MBps"]

    terms = {
        "IB_rotation_preload_MBps": _maybe_float(ib_rotation),
        "IB_outfifo_preload_MBps": ib_outfifo,
        "IB_streaming_MBps": ib_streaming,
    }
    binding_term, dpu_ib = _binding_term(terms)

    shared = {
        "MO_buf_bytes": mo_buf,
        "DSC_data_bytes": dsc_data,
        "DSIM_data_bytes": dsim_data,
        "OUTFIFO_data_bytes": outfifo_data,
        "Shared_data_bytes": shared_data,
        "Total_pipeline_data_bytes": total_pipeline_data,
        "Total_preload_data_bytes": total_preload_data,
        "outfifo_demanded_MBps": outfifo_demanded,
    }
    streaming = {
        "dpuf_clk_MHz": dpuf_clk,
        "stream_sum_MBps": stream_sum,
        "PTW_base_MBps": stream_ptw_base,
        "PTW_MBps": stream_ptw,
        "ptw_group_mode": system.ptw_group_mode,
    }
    formula_values: dict[str, float | str | None] = {
        "PTW": system.PTW,
        "MO_buf_bytes": mo_buf,
        "Total_pipeline_data_bytes": total_pipeline_data,
        "Shared_data_bytes": shared_data,
        "V_blank_time_ns": time["V_blank_time_ns"],
        "dpuf_clk_MHz": dpuf_clk,
        "stream_sum_MBps": stream_sum,
        "stream_ptw_MBps": stream_ptw,
        "ptw_group_mode": system.ptw_group_mode,
        "pipeline_latency_cycles": rotation["pipeline_latency_cycles"],
        "pipeline_latency_lines": rotation["pipeline_latency_lines"],
        "tx_allow_time_ns": rotation["tx_allow_time_ns"],
        "tx_allow_lines": rotation["tx_allow_lines"],
        "rotation_data_bytes": rotation["rotation_data_bytes"],
    }
    for key in [
        "ACLK1_MHz",
        "ACLK2_MHz",
        "ACLK3_MHz",
        "ACLK4_MHz",
        "ACLK5_MHz",
        "ACLK6_MHz",
        "DPU_ACLK_MHz",
        "aclk_binding",
        "dpuf_xres_effective",
        "dpuf_yres_effective",
        "dpuf_resolution_source",
        "dpu_xres_effective",
        "dpu_yres_effective",
        "dpu_resolution_source",
    ]:
        formula_values[key] = clock[key]

    return Result(
        timing=time,
        per_layer=per_layer,
        shared=shared,
        streaming=streaming,
        clock=clock,
        rotation=rotation,
        terms=terms,
        dpu_ib_MBps=dpu_ib,
        binding_term=binding_term,
        flags=flags,
        formula_values=formula_values,
    )


def _layer_is_in_ptw_group(layer: LayerConfig, mode: str) -> bool:
    if mode == "dpuf0":
        return layer.dpuf == 0
    if mode == "rotation_layers":
        return layer.rotation
    raise ValueError("ptw_group_mode must be 'dpuf0' or 'rotation_layers'")


def _rotation_preload(
    config: SimulatorConfig,
    per_layer: dict[str, dict[str, float | str | bool | None]],
    time: dict[str, float],
    dsc: DscInfo,
    clock: dict[str, Any],
    flags: dict[str, str],
) -> dict[str, float | str | None]:
    system = config.system
    panel = config.panel

    compression_cycles = 0.0
    scaler_cycles = 0.0
    rotation_data = 0.0
    for layer in _latency_layers(config):
        info = _format_info(layer.fmt)
        if layer.compressed:
            compression_cycles = max(compression_cycles, layer.src_h * info.lmc / system.ppc_comp)
        scaler_cycles = max(scaler_cycles, layer.src_h * system.vtap_scaler / system.ppc_scaler)
    for layer in config.layers:
        if layer.rotation:
            rotation_data += float(per_layer[layer.name]["rot_init_data_bytes"])

    dsc_cycles = panel.panel_w * dsc.slice_factor / dsc.ppc_dsc
    dsim_cycles = panel.panel_w / dsc.ppc_dsim
    outfifo_cycles = panel.panel_w * system.vtap_outfifo / system.ppc_outfifo
    pipeline_cycles = compression_cycles + scaler_cycles + dsc_cycles + dsim_cycles + outfifo_cycles

    base = {
        "compression_latency_cycles": compression_cycles,
        "scaler_latency_cycles": scaler_cycles,
        "DSC_latency_cycles": dsc_cycles,
        "DSIM_latency_cycles": dsim_cycles,
        "Outfifo_latency_cycles": outfifo_cycles,
        "pipeline_latency_cycles": pipeline_cycles,
        "rotation_data_bytes": rotation_data,
        "pipeline_latency_time_ns": None,
        "pipeline_latency_lines": None,
        "tx_allow_time_ns": None,
        "tx_allow_lines": None,
        "IB_rotation_preload_MBps": None,
        "status": "N/A",
    }

    if rotation_data == 0:
        base["status"] = "N/A (no rotation layers)"
        return base
    dpu_aclk_mhz = float(clock["DPU_ACLK_MHz"])
    if dpu_aclk_mhz <= 0:
        base["status"] = "N/A (computed DPU_ACLK_MHz is not positive)"
        return base

    latency_time = pipeline_cycles * 1000 / dpu_aclk_mhz
    tx_allow = time["available_time_ns"] - latency_time
    latency_lines = latency_time / time["T_line_ns"]
    tx_allow_lines = tx_allow / time["T_line_ns"]
    if tx_allow <= 0:
        base["pipeline_latency_time_ns"] = latency_time
        base["pipeline_latency_lines"] = latency_lines
        base["tx_allow_time_ns"] = tx_allow
        base["tx_allow_lines"] = tx_allow_lines
        base["status"] = "N/A (pipeline latency exceeds available time)"
        return base

    demanded_sum = 0.0
    for layer in config.layers:
        if layer.rotation:
            demanded = float(per_layer[layer.name]["rot_init_data_bytes"]) / tx_allow * 1000
            per_layer[layer.name]["rotation_demanded_MBps"] = demanded
            demanded_sum += demanded

    base["pipeline_latency_time_ns"] = latency_time
    base["pipeline_latency_lines"] = latency_lines
    base["tx_allow_time_ns"] = tx_allow
    base["tx_allow_lines"] = tx_allow_lines
    base["IB_rotation_preload_MBps"] = demanded_sum * (1 + system.PTW)
    base["status"] = "computed"
    return base


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _binding_term(terms: dict[str, float | None]) -> tuple[str | None, float]:
    finite = {key: value for key, value in terms.items() if value is not None and math.isfinite(value)}
    if not finite:
        return None, math.nan
    key = max(finite, key=lambda name: float(finite[name]))
    return key.removesuffix("_MBps"), float(finite[key])


def sweep(
    config: SimulatorConfig,
    param: str,
    start: float,
    stop: float,
    step: float,
    param2: str | None = None,
    start2: float | None = None,
    stop2: float | None = None,
    step2: float | None = None,
) -> list[dict[str, Any]]:
    values1 = list(_inclusive_range(start, stop, step))
    values2 = [None]
    if param2:
        if start2 is None or stop2 is None or step2 is None:
            raise ValueError("start2, stop2, and step2 are required when param2 is set")
        values2 = list(_inclusive_range(start2, stop2, step2))

    rows: list[dict[str, Any]] = []
    for value1 in values1:
        for value2 in values2:
            scenario = copy.deepcopy(config)
            assigned1 = set_path(scenario, param, value1)
            row: dict[str, Any] = {param: assigned1}
            if param2:
                assigned2 = set_path(scenario, param2, value2)
                row[param2] = assigned2
            result = solve(scenario)
            row.update(
                {
                    "DPU_IB_MBps": result.dpu_ib_MBps,
                    "binding_term": result.binding_term,
                    "IB_rotation_preload_MBps": result.terms["IB_rotation_preload_MBps"],
                    "IB_outfifo_preload_MBps": result.terms["IB_outfifo_preload_MBps"],
                    "IB_streaming_MBps": result.terms["IB_streaming_MBps"],
                }
            )
            rows.append(row)
    return rows


BATCH_RESULT_COLUMNS = [
    "DPU_IB_MBps",
    "binding_term",
    "IB_rotation_preload_MBps",
    "IB_outfifo_preload_MBps",
    "IB_streaming_MBps",
    "DPU_ACLK_MHz",
    "aclk_binding",
    "status",
    "error",
]


@dataclass
class BatchInput:
    columns: list[str]
    rows: list[dict[str, str | None]]
    items: list[dict[str, str]]


class BatchRows(list[dict[str, Any]]):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        input_items: list[dict[str, str]],
        breakdowns: list[list[dict[str, Any]] | None],
    ) -> None:
        super().__init__(rows)
        self.input_items = input_items
        self.breakdowns = breakdowns


def batch_calculate(base_config: SimulatorConfig, csv_path: str | Path) -> list[dict[str, Any]]:
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is None:
            return []
        fieldnames = list(reader.fieldnames)
        raw_rows = list(reader)
    batch_input = _normalize_batch_input(fieldnames, raw_rows)
    rows: list[dict[str, Any]] = []
    breakdowns: list[list[dict[str, Any]] | None] = []
    for row in batch_input.rows:
        batch_row, batch_breakdown = _batch_row(base_config, batch_input.columns, row)
        rows.append(batch_row)
        breakdowns.append(batch_breakdown)
    return BatchRows(rows, batch_input.items, breakdowns)


def _normalize_batch_input(
    fieldnames: list[str],
    raw_rows: list[dict[str, str | None]],
) -> BatchInput:
    if _is_ui_matrix_batch_input(fieldnames):
        return _ui_matrix_batch_input(fieldnames, raw_rows)
    if _is_item_matrix_batch_input(fieldnames):
        return _transpose_batch_input(fieldnames, raw_rows)
    return BatchInput(fieldnames, raw_rows, [_input_item_for_name(column) for column in fieldnames])


def _is_ui_matrix_batch_input(fieldnames: list[str]) -> bool:
    return len(fieldnames) >= 3 and fieldnames[:3] == ["section", "group", "name"]


def _is_item_matrix_batch_input(fieldnames: list[str]) -> bool:
    return bool(fieldnames) and fieldnames[0] == "item"


def _ui_matrix_batch_input(
    fieldnames: list[str],
    raw_rows: list[dict[str, str | None]],
) -> BatchInput:
    try:
        batch_start = fieldnames.index("note") + 1
    except ValueError:
        batch_start = fieldnames.index("name") + 1
    batch_columns = fieldnames[batch_start:]
    input_rows = [
        row
        for row in raw_rows
        if str(row.get("section", "") or "").strip() == "input" and str(row.get("name", "") or "").strip()
    ]
    input_items = [
        {
            "section": "input",
            "group": str(row.get("group", "") or "").strip(),
            "name": str(row.get("name", "") or "").strip(),
            "note": str(row.get("note", "") or "").strip(),
        }
        for row in input_rows
    ]
    input_columns = [item["name"] for item in input_items]
    case_rows: list[dict[str, str | None]] = []
    for batch_column in batch_columns:
        case: dict[str, str | None] = {}
        for row, item in zip(input_rows, input_items):
            case[item["name"]] = row.get(batch_column, "")
        case_rows.append(case)
    return BatchInput(input_columns, case_rows, input_items)


def _transpose_batch_input(
    fieldnames: list[str],
    raw_rows: list[dict[str, str | None]],
) -> BatchInput:
    batch_columns = fieldnames[1:]
    input_columns = [str(row.get("item", "") or "").strip() for row in raw_rows]
    input_columns = [column for column in input_columns if column]
    input_items = [_input_item_for_name(column) for column in input_columns]
    case_rows: list[dict[str, str | None]] = []
    for batch_column in batch_columns:
        case: dict[str, str | None] = {}
        for row in raw_rows:
            item = str(row.get("item", "") or "").strip()
            if item:
                case[item] = row.get(batch_column, "")
        case_rows.append(case)
    return BatchInput(input_columns, case_rows, input_items)


def _input_item_for_name(name: str) -> dict[str, str]:
    group = "meta"
    layer_match = _LAYER_PATH.match(name)
    if name.startswith("panel."):
        group = "panel"
    elif name.startswith("system."):
        group = "system"
    elif layer_match:
        group = f"layers[{layer_match.group(1)}]"
    return {"section": "input", "group": group, "name": name, "note": ""}


def _batch_row(
    base_config: SimulatorConfig,
    input_columns: list[str],
    input_row: dict[str, str | None],
) -> tuple[dict[str, Any], list[dict[str, Any]] | None]:
    output: dict[str, Any] = {column: input_row.get(column, "") or "" for column in input_columns}
    scenario = copy.deepcopy(base_config)
    try:
        for column in input_columns:
            value = input_row.get(column)
            if not _is_batch_override_column(column) or value is None or value.strip() == "":
                continue
            set_path(scenario, column, value.strip())
        result = solve(scenario)
        output.update(_batch_result_values(result))
        return output, breakdown_rows(result)
    except Exception as exc:
        output.update({column: "" for column in BATCH_RESULT_COLUMNS})
        output["status"] = "error"
        output["error"] = str(exc)
        return output, None


def _is_batch_override_column(column: str) -> bool:
    return column.startswith(("panel.", "system.")) or _LAYER_PATH.match(column) is not None


def _batch_result_values(result: Result) -> dict[str, Any]:
    return {
        "DPU_IB_MBps": result.dpu_ib_MBps,
        "binding_term": result.binding_term or "",
        "IB_rotation_preload_MBps": _blank_none(result.terms["IB_rotation_preload_MBps"]),
        "IB_outfifo_preload_MBps": _blank_none(result.terms["IB_outfifo_preload_MBps"]),
        "IB_streaming_MBps": _blank_none(result.terms["IB_streaming_MBps"]),
        "DPU_ACLK_MHz": result.clock["DPU_ACLK_MHz"],
        "aclk_binding": result.clock["aclk_binding"],
        "status": "ok",
        "error": "",
    }


def _blank_none(value: Any) -> Any:
    return "" if value is None else value


def _inclusive_range(start: float, stop: float, step: float) -> list[float]:
    if step == 0:
        raise ValueError("step must not be zero")
    if (stop - start) * step < 0:
        raise ValueError("step sign does not move from start toward stop")

    values: list[float] = []
    index = 0
    epsilon = abs(step) / 1_000_000
    while True:
        value = round(start + index * step, 10)
        if step > 0 and value > stop + epsilon:
            break
        if step < 0 and value < stop - epsilon:
            break
        values.append(value)
        index += 1
    return values


_LAYER_PATH = re.compile(r"^layers\[(\d+)\]\.([A-Za-z_][A-Za-z0-9_]*)$")


def get_path(config: SimulatorConfig, path: str) -> Any:
    layer_match = _LAYER_PATH.match(path)
    if layer_match:
        index = int(layer_match.group(1))
        field_name = layer_match.group(2)
        return getattr(config.layers[index], field_name)
    section, _, field_name = path.partition(".")
    if not field_name:
        raise ValueError(f"invalid path: {path}")
    if section == "panel":
        return getattr(config.panel, field_name)
    if section == "system":
        return getattr(config.system, field_name)
    raise ValueError(f"invalid path: {path}")


def set_path(config: SimulatorConfig, path: str, value: Any) -> Any:
    current = get_path(config, path)
    coerced = _coerce_for_path(path, current, value)
    layer_match = _LAYER_PATH.match(path)
    if layer_match:
        index = int(layer_match.group(1))
        field_name = layer_match.group(2)
        setattr(config.layers[index], field_name, coerced)
        return coerced
    section, _, field_name = path.partition(".")
    target = config.panel if section == "panel" else config.system
    setattr(target, field_name, coerced)
    return coerced


OPTIONAL_INT_PATHS = {
    "system.dpuf_xres",
    "system.dpuf_yres",
    "system.dpu_xres",
    "system.dpu_yres",
}
OPTIONAL_INT_LAYER_FIELDS = {"dst_w", "dst_h"}


def _coerce_for_path(path: str, current: Any, value: Any) -> Any:
    layer_match = _LAYER_PATH.match(path)
    if path in OPTIONAL_INT_PATHS or (layer_match and layer_match.group(2) in OPTIONAL_INT_LAYER_FIELDS):
        return _optional_int(value)
    return _coerce_like(current, value)


def _coerce_like(current: Any, value: Any) -> Any:
    if current is None:
        return None if value in ("", None) else float(value)
    if isinstance(current, bool):
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if isinstance(current, int) and not isinstance(current, bool):
        return int(round(float(value)))
    if isinstance(current, float):
        return float(value)
    return str(value)


def _optional_int(value: Any) -> int | None:
    text = "" if value is None else str(value).strip()
    if text.lower() in {"", "none", "null", "panel default", "(panel default)", "auto default", "(auto default)"}:
        return None
    return int(float(text))


GROUP_BACKGROUNDS: dict[str, str] = {
    "timing": "#eef6ff",
    "shared": "#f2f8ec",
    "streaming": "#fff4df",
    "clock": "#fff8c5",
    "rotation": "#f8eefc",
    "term": "#fff0f0",
}
LAYER_BACKGROUNDS = ["#f7f7f7", "#f1f5f9", "#f8faf0", "#fdf6f0"]


def breakdown_rows(result: Result, previous: Result | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, value in result.timing.items():
        rows.append(_breakdown_row("timing", key, value))
    for key, value in result.shared.items():
        rows.append(_breakdown_row("shared", key, value))
    for key, value in result.streaming.items():
        extra = "F1/F2" if key != "dpuf_clk_MHz" else ""
        rows.append(_breakdown_row("streaming", key, value, extra))
    for key in [
        "ACLK1_MHz",
        "ACLK2_MHz",
        "ACLK3_MHz",
        "ACLK4_MHz",
        "ACLK5_MHz",
        "ACLK6_MHz",
        "DPU_ACLK_MHz",
        "aclk_binding",
        "dpuf_xres_effective",
        "dpuf_yres_effective",
        "dpuf_resolution_source",
        "dpu_xres_effective",
        "dpu_yres_effective",
        "dpu_resolution_source",
    ]:
        if key == "DPU_ACLK_MHz":
            extra = "F6/F7; binding"
        elif key == "aclk_binding":
            extra = "binding"
        elif key in {"dpuf_resolution_source", "dpu_resolution_source"}:
            extra = f"F7; {result.clock[key]}"
        elif key.endswith("_effective"):
            extra = "F7; effective value"
        else:
            extra = "F6/F7"
        rows.append(_breakdown_row("clock", key, result.clock[key], extra))
    for key, value in result.rotation.items():
        extra = "F5" if key != "status" else str(value)
        rows.append(_breakdown_row("rotation", key, value, extra))
    for index, (layer, values) in enumerate(result.per_layer.items()):
        background = LAYER_BACKGROUNDS[index % len(LAYER_BACKGROUNDS)]
        for key, value in values.items():
            rows.append(_breakdown_row(layer, key, value, background=background))
    for key, value in result.terms.items():
        extra = "binding" if result.binding_term == key.removesuffix("_MBps") else ""
        rows.append(_breakdown_row("term", key, value, extra))
    if previous is not None:
        previous_rows = breakdown_rows(previous)
        previous_by_key = {(row["group"], row["name"]): row for row in previous_rows}
        for row in rows:
            previous_row = previous_by_key.get((row["group"], row["name"]))
            row["delta"] = _delta_display(row.get("_raw"), previous_row.get("_raw") if previous_row else None)
    else:
        for row in rows:
            row["delta"] = ""
    return rows


def _breakdown_row(
    group: str,
    name: str,
    value: Any,
    extra_note: str = "",
    background: str | None = None,
) -> dict[str, Any]:
    unit = _unit_for(name)
    note_parts = [part for part in [unit, extra_note] if part and part != "None"]
    return {
        "group": group,
        "name": name,
        "model": _fmt(value),
        "_raw": value,
        "delta": "",
        "measured": "",
        "note": "; ".join(note_parts),
        "background": background or GROUP_BACKGROUNDS.get(group, "#ffffff"),
    }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _value_changed(current: Any, previous: Any) -> bool:
    if _is_number(current) and _is_number(previous):
        return abs(float(current) - float(previous)) > 1e-9
    return current != previous


def _delta_display(current: Any, previous: Any) -> str:
    if previous is None:
        return ""
    if _is_number(current) and _is_number(previous):
        delta = float(current) - float(previous)
        if abs(delta) < 0.005:
            return "0.00"
        return f"{delta:+.2f}"
    if current == previous:
        return "same"
    return "changed"


def _unit_for(name: str) -> str:
    if name.endswith("_bytes"):
        return "B"
    if name.endswith("_ns"):
        return "ns"
    if name.endswith("_cycles"):
        return "cycles"
    if name.endswith("_MHz"):
        return "MHz"
    if name.endswith("_MBps"):
        return "MB/s"
    if name == "bpp":
        return "bits/pixel"
    if name in {"read_line", "lmc"}:
        return "lines"
    if name in {"scale_v", "stream_coeff", "PTW_base_MBps"}:
        return "ratio" if name != "PTW_base_MBps" else "MB/s"
    if name == "dpuf":
        return "index"
    if name.endswith("_xres_effective") or name.endswith("_yres_effective"):
        return "px"
    if name.endswith("_resolution_source"):
        return "source"
    if name == "aclk_binding":
        return "tag"
    return ""


TERM_ORDER = [
    ("IB_rotation_preload", "Rotation preload", "IB_rotation_preload_MBps"),
    ("IB_outfifo_preload", "Outfifo preload", "IB_outfifo_preload_MBps"),
    ("IB_streaming", "Streaming", "IB_streaming_MBps"),
]


def term_summary_rows(result: Result, previous: Result | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, title, term_key in TERM_ORDER:
        is_binding = result.binding_term == key
        previous_value = previous.terms.get(term_key) if previous is not None else None
        rows.append(
            {
                "key": key,
                "title": title,
                "value": result.terms.get(term_key),
                "display": _fmt(result.terms.get(term_key)),
                "delta_display": _delta_display(result.terms.get(term_key), previous_value) if previous is not None else "",
                "unit": "MB/s",
                "is_binding": is_binding,
                "label": "MAX" if is_binding else "",
            }
        )
    return rows


def result_delta_summary(result: Result, previous: Result | None = None) -> dict[str, Any]:
    if previous is None:
        return {
            "dpu_ib_delta": None,
            "dpu_aclk_delta": None,
            "binding_changed": False,
            "changed_clock_terms": [],
            "headline": "No previous result yet",
        }
    dpu_ib_delta = result.dpu_ib_MBps - previous.dpu_ib_MBps
    dpu_aclk_delta = float(result.clock["DPU_ACLK_MHz"]) - float(previous.clock["DPU_ACLK_MHz"])
    binding_changed = result.binding_term != previous.binding_term
    changed_clock_terms = [
        label
        for label, key in [
            ("ACLK1", "ACLK1_MHz"),
            ("ACLK2", "ACLK2_MHz"),
            ("ACLK3", "ACLK3_MHz"),
            ("ACLK4", "ACLK4_MHz"),
            ("ACLK5", "ACLK5_MHz"),
            ("ACLK6", "ACLK6_MHz"),
        ]
        if _value_changed(result.clock[key], previous.clock[key])
    ]
    parts = [
        f"DPU_IB {'unchanged' if abs(dpu_ib_delta) < 0.005 else 'changed ' + _delta_display(result.dpu_ib_MBps, previous.dpu_ib_MBps) + ' MB/s'}",
        f"DPU_ACLK {'unchanged' if abs(dpu_aclk_delta) < 0.005 else 'changed ' + _delta_display(result.clock['DPU_ACLK_MHz'], previous.clock['DPU_ACLK_MHz']) + ' MHz'}",
    ]
    if binding_changed:
        parts.append(f"binding changed: {previous.binding_term} -> {result.binding_term}")
    if changed_clock_terms:
        parts.append("changed clock terms: " + ", ".join(changed_clock_terms))
    return {
        "dpu_ib_delta": dpu_ib_delta,
        "dpu_aclk_delta": dpu_aclk_delta,
        "binding_changed": binding_changed,
        "changed_clock_terms": changed_clock_terms,
        "headline": "; ".join(parts),
    }


def build_formula_html(result: Result) -> str:
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<script id='MathJax-script' async src='https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'></script>",
        "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:18px;line-height:1.45}"
        ".term{border:1px solid #d0d7de;border-radius:6px;padding:12px;margin:0 0 12px}"
        ".bind{border-color:#0969da;background:#f6f8fa}.flag{color:#9a6700;font-weight:600}"
        "table{border-collapse:collapse;margin:8px 0 12px}"
        "td,th{border:1px solid #d0d7de;padding:5px 8px;text-align:left}"
        ".live-values th{background:#f6f8fa}.live-values td:nth-child(2){font-weight:600;text-align:right}"
        ".live-values td:nth-child(3){color:#57606a}</style>",
        "</head><body>",
    ]
    parts.append(f"<h2>DPU_IB = {_fmt(result.dpu_ib_MBps)} MB/s</h2>")
    parts.append(f"<p>Binding term: <b>{html.escape(str(result.binding_term))}</b></p>")
    for formula in FORMULAE:
        key = formula["key"]
        is_binding = result.binding_term == key
        css = "term bind" if is_binding else "term"
        parts.append(f"<section class='{css}'>")
        parts.append(f"<h3>{html.escape(formula['title'])}</h3>")
        parts.append(f"<p>\\[{formula['latex']}\\]</p>")
        parts.append(_formula_value_table(formula, result))
        parts.append("<table><tr><th>Symbol</th><th>Description</th><th>Unit</th><th>Source</th></tr>")
        for symbol, details in formula["symbols"].items():
            desc, unit, source = details
            parts.append(
                "<tr>"
                f"<td>{html.escape(symbol)}</td>"
                f"<td>{html.escape(desc)}</td>"
                f"<td>{html.escape(unit)}</td>"
                f"<td>{html.escape(source)}</td>"
                "</tr>"
            )
        parts.append("</table>")
        if key in {"DPU_ACLK", "IB_streaming", "IB_rotation_preload"}:
            if key == "DPU_ACLK":
                flag_ids = ["F6", "F7"]
            elif key == "IB_streaming":
                flag_ids = ["F1", "F2"]
            else:
                flag_ids = ["F5"]
            active = [f"{flag}: {result.flags[flag]}" for flag in flag_ids if flag in result.flags]
            if active:
                parts.append("<p class='flag'>" + html.escape(" | ".join(active)) + "</p>")
        parts.append("</section>")
    parts.append("</body></html>")
    return "".join(parts)


def _formula_value_table(formula: dict[str, Any], result: Result) -> str:
    rows = ["<h4>Live values</h4><table class='live-values'><tr><th>Name</th><th>Value</th><th>Unit</th></tr>"]
    for key in formula["bind_keys"]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(key)}</td>"
            f"<td>{html.escape(_fmt(result.formula_values.get(key)))}</td>"
            f"<td>{html.escape(_unit_for(key))}</td>"
            "</tr>"
        )
    if formula["key"] == "DPU_ACLK":
        term_value = result.formula_values.get("DPU_ACLK_MHz")
        term_unit = "MHz"
    else:
        term_value = result.terms.get(f"{formula['key']}_MBps")
        term_unit = "MB/s"
    rows.append(
        "<tr>"
        f"<td>{html.escape(formula['key'])}</td>"
        f"<td>{html.escape(_fmt(term_value))}</td>"
        f"<td>{html.escape(term_unit)}</td>"
        "</tr>"
    )
    rows.append("</table>")
    return "".join(rows)


def _formula_value_line(formula: dict[str, Any], result: Result) -> str:
    values = []
    for key in formula["bind_keys"]:
        values.append(f"{key}={_fmt(result.formula_values.get(key))}")
    term_value = result.terms.get(f"{formula['key']}_MBps")
    values.append(f"{formula['key']}={_fmt(term_value)} MB/s")
    return ", ".join(values)


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if math.isnan(value):
            return "N/A"
        return f"{value:.2f}"
    return str(value)


def write_sweep_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    if not rows:
        raise ValueError("no sweep rows to write")
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_batch_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    write_sweep_csv(_ui_batch_matrix_rows(rows), path)


def _ui_batch_matrix_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("no batch rows to write")
    matrix: list[dict[str, Any]] = []
    input_items = getattr(rows, "input_items", None) or _fallback_input_items(rows)
    for item in input_items:
        matrix.append(_batch_matrix_row(item["section"], item["group"], item["name"], item["note"], rows, item["name"]))
    matrix.extend(_batch_summary_rows(rows))
    matrix.extend(_batch_breakdown_matrix_rows(rows))
    return matrix


def _fallback_input_items(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    result_columns = set(BATCH_RESULT_COLUMNS)
    return [_input_item_for_name(column) for column in rows[0] if column not in result_columns]


def _batch_matrix_row(
    section: str,
    group: str,
    name: str,
    note: str,
    rows: list[dict[str, Any]],
    source_key: str,
) -> dict[str, Any]:
    row = _empty_batch_matrix_row(section, group, name, note, len(rows))
    for index, source in enumerate(rows, start=1):
        row[f"batch_{index}"] = source.get(source_key, "")
    return row


def _empty_batch_matrix_row(section: str, group: str, name: str, note: str, batch_count: int) -> dict[str, Any]:
    row: dict[str, Any] = {"section": section, "group": group, "name": name, "note": note}
    for index in range(1, batch_count + 1):
        row[f"batch_{index}"] = ""
    return row


def _batch_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        _batch_matrix_row("summary", "term", "DPU_IB_MBps", "MB/s", rows, "DPU_IB_MBps"),
        _batch_matrix_row("summary", "term", "binding_term", "", rows, "binding_term"),
        _batch_matrix_row("summary", "status", "status", "", rows, "status"),
        _batch_matrix_row("summary", "status", "error", "", rows, "error"),
    ]


def _batch_breakdown_matrix_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    breakdowns: list[list[dict[str, Any]] | None] = getattr(rows, "breakdowns", [None for _ in rows])
    ordered_keys: list[tuple[str, str]] = []
    row_meta: dict[tuple[str, str], dict[str, Any]] = {}
    for breakdown in breakdowns:
        if not breakdown:
            continue
        for breakdown_row in breakdown:
            key = (str(breakdown_row["group"]), str(breakdown_row["name"]))
            if key not in row_meta:
                ordered_keys.append(key)
                row_meta[key] = breakdown_row

    matrix_rows: list[dict[str, Any]] = []
    for key in ordered_keys:
        meta = row_meta[key]
        row = _empty_batch_matrix_row("breakdown", str(meta["group"]), str(meta["name"]), str(meta["note"]), len(rows))
        for index, breakdown in enumerate(breakdowns, start=1):
            source_row = _find_breakdown_row(breakdown, key)
            row[f"batch_{index}"] = source_row.get("model", "") if source_row else ""
        matrix_rows.append(row)
    return matrix_rows


def _find_breakdown_row(
    breakdown: list[dict[str, Any]] | None,
    key: tuple[str, str],
) -> dict[str, Any] | None:
    if not breakdown:
        return None
    for row in breakdown:
        if (str(row["group"]), str(row["name"])) == key:
            return row
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DPU IB simulator")
    parser.add_argument("--gui", action="store_true", help="launch the PySide6 GUI (default when --sweep is not used)")
    parser.add_argument("--config", help="YAML config to open in GUI")
    parser.add_argument("--sweep", metavar="CFG", help="run a headless sweep from a YAML config")
    parser.add_argument("--batch", metavar="CSV", help="run headless batch cases from a CSV file")
    parser.add_argument("--base", metavar="CFG", help="base YAML config for --batch; defaults to the built-in golden config")
    parser.add_argument("--param", default="panel.fps", help="1D sweep parameter path")
    parser.add_argument("--start", type=float, default=60.0)
    parser.add_argument("--stop", type=float, default=144.0)
    parser.add_argument("--step", type=float, default=12.0)
    parser.add_argument("--param2", help="optional 2D sweep parameter path")
    parser.add_argument("--start2", type=float)
    parser.add_argument("--stop2", type=float)
    parser.add_argument("--step2", type=float)
    parser.add_argument("--out", default="sweep.csv", help="CSV output path for --sweep or --batch")
    args = parser.parse_args(argv)

    if args.batch:
        cfg = load_config(args.base) if args.base else default_golden_config()
        rows = batch_calculate(cfg, args.batch)
        write_batch_csv(rows, args.out)
        print(f"wrote {len(rows)} batch rows to {args.out}")
        return 0

    if args.sweep:
        cfg = load_config(args.sweep)
        rows = sweep(
            cfg,
            args.param,
            args.start,
            args.stop,
            args.step,
            args.param2,
            args.start2,
            args.stop2,
            args.step2,
        )
        write_sweep_csv(rows, args.out)
        print(f"wrote {len(rows)} rows to {args.out}")
        return 0

    cfg = load_config(args.config) if args.config else default_golden_config()
    return run_gui(cfg)


def run_gui(config: SimulatorConfig) -> int:  # pragma: no cover - GUI smoke is manual
    try:
        from PySide6.QtCore import Qt, QTimer
        from PySide6.QtGui import QColor, QFont
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QFileDialog,
            QFormLayout,
            QFrame,
            QGridLayout,
            QHBoxLayout,
            QHeaderView,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QScrollArea,
            QSplitter,
            QTableWidget,
            QTableWidgetItem,
            QTabWidget,
            QTextBrowser,
            QVBoxLayout,
            QWidget,
        )
    except ImportError:
        print("PySide6 is required for the GUI. Install with: uv sync --extra gui", file=sys.stderr)
        return 2

    try:
        from PySide6.QtWebEngineWidgets import QWebEngineView
    except ImportError:
        QWebEngineView = None

    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
    except ImportError:
        FigureCanvas = None
        Figure = None

    class SimulatorWindow(QMainWindow):
        def __init__(self, initial: SimulatorConfig) -> None:
            super().__init__()
            self.setWindowTitle("DPU IB Simulator")
            self.config = copy.deepcopy(initial)
            self.baseline_result: Result | None = None
            self.last_result: Result | None = None
            self.debounce = QTimer(self)
            self.debounce.setSingleShot(True)
            self.debounce.setInterval(300)
            self.debounce.timeout.connect(self.recalculate)

            root = QWidget()
            self.setCentralWidget(root)
            root_layout = QHBoxLayout(root)
            root_layout.setContentsMargins(10, 10, 10, 10)

            splitter = QSplitter(Qt.Horizontal)
            splitter.setChildrenCollapsible(False)
            root_layout.addWidget(splitter)

            left_scroll = QScrollArea()
            left_scroll.setWidgetResizable(True)
            left_scroll.setMinimumWidth(340)
            left_body = QWidget()
            left_scroll.setWidget(left_body)
            left_layout = QVBoxLayout(left_body)
            splitter.addWidget(left_scroll)

            self.panel_fields: dict[str, QLineEdit | QComboBox] = {}
            self.system_fields: dict[str, QLineEdit | QComboBox] = {}
            self._build_panel_form(left_layout)
            self._build_system_form(left_layout)
            self._build_layer_table(left_layout)
            actions = QHBoxLayout()
            calc_btn = QPushButton("Calculate")
            calc_btn.clicked.connect(self.recalculate)
            baseline_btn = QPushButton("Set baseline")
            baseline_btn.clicked.connect(self.set_baseline)
            actions.addWidget(calc_btn)
            actions.addWidget(baseline_btn)
            left_layout.addLayout(actions)
            left_layout.addStretch(1)

            right = QWidget()
            right_layout = QVBoxLayout(right)
            splitter.addWidget(right)
            splitter.setStretchFactor(0, 0)
            splitter.setStretchFactor(1, 1)
            splitter.setSizes([520, 1280])

            self.summary_frame = QFrame()
            self.summary_frame.setObjectName("summaryFrame")
            self.summary_frame.setStyleSheet(
                "#summaryFrame {"
                "background: #f6f8fa;"
                "border: 1px solid #d0d7de;"
                "border-radius: 6px;"
                "}"
                "QLabel { color: #1f2328; }"
            )
            summary_layout = QVBoxLayout(self.summary_frame)
            summary_layout.setContentsMargins(14, 10, 14, 10)
            self.top_label = QLabel("")
            self.top_label.setTextFormat(Qt.RichText)
            self.top_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            summary_layout.addWidget(self.top_label)
            self.delta_label = QLabel("")
            self.delta_label.setTextFormat(Qt.RichText)
            self.delta_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.delta_label.setStyleSheet("font-size: 12px; color: #57606a;")
            summary_layout.addWidget(self.delta_label)
            self.term_grid = QGridLayout()
            self.term_grid.setHorizontalSpacing(8)
            self.term_grid.setVerticalSpacing(4)
            summary_layout.addLayout(self.term_grid)
            self.term_labels: dict[str, QLabel] = {}
            for column, (key, title, _term_key) in enumerate(TERM_ORDER):
                title_label = QLabel(title)
                title_label.setStyleSheet("font-size: 11px; color: #57606a;")
                value_label = QLabel("")
                value_label.setTextFormat(Qt.RichText)
                value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
                value_label.setMinimumWidth(170)
                value_label.setStyleSheet(
                    "background: #ffffff;"
                    "border: 1px solid #d8dee4;"
                    "border-radius: 4px;"
                    "padding: 7px 9px;"
                    "font-size: 13px;"
                )
                self.term_grid.addWidget(title_label, 0, column)
                self.term_grid.addWidget(value_label, 1, column)
                self.term_labels[key] = value_label
            right_layout.addWidget(self.summary_frame)
            self.tabs = QTabWidget()
            right_layout.addWidget(self.tabs, 1)

            self.breakdown = QTableWidget()
            self.breakdown.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
            self.breakdown.horizontalHeader().setStretchLastSection(False)
            self.breakdown.horizontalHeader().setMinimumSectionSize(64)
            self.tabs.addTab(self.breakdown, "Breakdown")
            if QWebEngineView is not None:
                self.formula_view = QWebEngineView()
            else:
                self.formula_view = QTextBrowser()
            self.tabs.addTab(self.formula_view, "Formula")
            self._build_sweep_tab(FigureCanvas, Figure)

            self._populate_inputs()
            self.recalculate()

        def _build_panel_form(self, parent: QVBoxLayout) -> None:
            parent.addWidget(QLabel("<b>Panel</b>"))
            form = QFormLayout()
            parent.addLayout(form)
            for name in ["panel_w", "panel_h", "fps", "disp_bpp", "vbp", "vfp", "vsa", "dst_y"]:
                edit = QLineEdit()
                edit.textChanged.connect(self._schedule)
                self.panel_fields[name] = edit
                form.addRow(name, edit)
            dsc = QComboBox()
            dsc.addItems(list(DSC_LUT))
            dsc.currentTextChanged.connect(self._schedule)
            self.panel_fields["dsc_mode"] = dsc
            form.addRow("dsc_mode", dsc)

        def _build_system_form(self, parent: QVBoxLayout) -> None:
            parent.addWidget(QLabel("<b>System</b>"))
            form = QFormLayout()
            parent.addLayout(form)
            for name in [
                "PTW",
                "MO_derating",
                "MO_entries",
                "MO_entry_bytes",
                "OF_lines",
                "margin",
                "bus_width_B",
                "bus_util",
                "max_bus_port_BW_MBs",
                "ppc_comp",
                "ppc_scaler",
                "ppc_outfifo",
                "ppc_pbld",
                "ppc_ai_scaler",
                "ppc_wb",
                "vtap_scaler",
                "vtap_outfifo",
                "stream_clk_overhead",
                "dpuf_xres",
                "dpuf_yres",
                "dpu_xres",
                "dpu_yres",
                "wb_xres",
                "wb_yres",
            ]:
                edit = QLineEdit()
                if name in {"dpuf_xres", "dpuf_yres", "dpu_xres", "dpu_yres"}:
                    edit.setPlaceholderText("panel default")
                    edit.setToolTip("Blank uses panel_w/panel_h fallback")
                elif name in {"wb_xres", "wb_yres"}:
                    edit.setToolTip("0 disables writeback clock contribution unless wb_enabled is True")
                edit.textChanged.connect(self._schedule)
                self.system_fields[name] = edit
                form.addRow(name, edit)
            for name in ["ai_scaler_enabled", "wb_enabled"]:
                combo = QComboBox()
                combo.addItems(["False", "True"])
                combo.currentTextChanged.connect(self._schedule)
                self.system_fields[name] = combo
                form.addRow(name, combo)
            mode = QComboBox()
            mode.addItems(["dpuf0", "rotation_layers"])
            mode.currentTextChanged.connect(self._schedule)
            self.system_fields["ptw_group_mode"] = mode
            form.addRow("ptw_group_mode", mode)

        def _build_layer_table(self, parent: QVBoxLayout) -> None:
            header = QHBoxLayout()
            header.addWidget(QLabel("<b>Layers</b>"))
            add_btn = QPushButton("+ Add")
            add_btn.clicked.connect(self.add_layer)
            del_btn = QPushButton("- Del")
            del_btn.clicked.connect(self.delete_layer)
            header.addWidget(add_btn)
            header.addWidget(del_btn)
            parent.addLayout(header)
            self.layer_columns = [
                "name",
                "fmt",
                "src_w",
                "src_h",
                "dst_w",
                "dst_h",
                "compressed",
                "compression_mode",
                "hdr",
                "scaling",
                "scale_v",
                "rotation",
                "dpuf",
                "stream_coeff",
            ]
            self.layers_table = QTableWidget(0, len(self.layer_columns))
            self.layers_table.setHorizontalHeaderLabels(self.layer_columns)
            self.layers_table.setMinimumHeight(260)
            self.layers_table.setMinimumWidth(500)
            self.layers_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            self.layers_table.horizontalHeader().setStretchLastSection(True)
            self.layers_table.itemChanged.connect(self._schedule)
            parent.addWidget(self.layers_table)

        def _build_sweep_tab(self, figure_canvas: Any, figure_type: Any) -> None:
            self.sweep_tab = QWidget()
            layout = QVBoxLayout(self.sweep_tab)
            controls = QGridLayout()
            self.sweep_paths = [
                "panel.fps",
                "panel.panel_w",
                "panel.panel_h",
                "panel.disp_bpp",
                "system.PTW",
                "system.MO_derating",
                "layers[0].src_w",
                "layers[0].src_h",
                "layers[0].scale_v",
                "layers[0].stream_coeff",
            ]
            self.sweep_param = QComboBox()
            self.sweep_param.addItems(self.sweep_paths)
            self.sweep_start = QLineEdit("60")
            self.sweep_stop = QLineEdit("144")
            self.sweep_step = QLineEdit("12")
            self.sweep_2d = QCheckBox("2D")
            self.sweep_param2 = QComboBox()
            self.sweep_param2.addItems(self.sweep_paths)
            self.sweep_param2.setCurrentText("layers[0].src_h")
            self.sweep_start2 = QLineEdit("1080")
            self.sweep_stop2 = QLineEdit("2160")
            self.sweep_step2 = QLineEdit("1080")
            run_btn = QPushButton("Run sweep")
            run_btn.clicked.connect(self.run_sweep)
            export_btn = QPushButton("Export CSV")
            export_btn.clicked.connect(self.export_sweep_csv)
            controls.addWidget(QLabel("X param"), 0, 0)
            controls.addWidget(self.sweep_param, 0, 1)
            controls.addWidget(QLabel("start"), 0, 2)
            controls.addWidget(self.sweep_start, 0, 3)
            controls.addWidget(QLabel("stop"), 0, 4)
            controls.addWidget(self.sweep_stop, 0, 5)
            controls.addWidget(QLabel("step"), 0, 6)
            controls.addWidget(self.sweep_step, 0, 7)
            controls.addWidget(self.sweep_2d, 1, 0)
            controls.addWidget(self.sweep_param2, 1, 1)
            controls.addWidget(QLabel("start"), 1, 2)
            controls.addWidget(self.sweep_start2, 1, 3)
            controls.addWidget(QLabel("stop"), 1, 4)
            controls.addWidget(self.sweep_stop2, 1, 5)
            controls.addWidget(QLabel("step"), 1, 6)
            controls.addWidget(self.sweep_step2, 1, 7)
            controls.addWidget(run_btn, 0, 8)
            controls.addWidget(export_btn, 1, 8)
            layout.addLayout(controls)
            self.sweep_rows: list[dict[str, Any]] = []
            if figure_canvas is not None:
                self.figure = figure_type(figsize=(5, 3))
                self.canvas = figure_canvas(self.figure)
                layout.addWidget(self.canvas, 1)
            else:
                self.figure = None
                self.canvas = QTextBrowser()
                layout.addWidget(self.canvas, 1)
            self.tabs.addTab(self.sweep_tab, "Sweep")

        def _populate_inputs(self) -> None:
            panel = self.config.panel
            system = self.config.system
            for name, widget in self.panel_fields.items():
                value = getattr(panel, name)
                if isinstance(widget, QComboBox):
                    widget.setCurrentText(str(value))
                else:
                    widget.setText(str(value))
            for name, widget in self.system_fields.items():
                value = getattr(system, name)
                if name in {"dpuf_xres", "dpuf_yres", "dpu_xres", "dpu_yres"} and value is None:
                    text = "panel default"
                else:
                    text = "" if value is None else str(value)
                if isinstance(widget, QComboBox):
                    widget.setCurrentText(text)
                else:
                    widget.setText(text)
            self.layers_table.blockSignals(True)
            self.layers_table.setRowCount(len(self.config.layers))
            for row, layer in enumerate(self.config.layers):
                for col, name in enumerate(self.layer_columns):
                    value = getattr(layer, name)
                    text = "auto default" if name in {"dst_w", "dst_h"} and value is None else ("" if value is None else str(value))
                    item = QTableWidgetItem(text)
                    if name in {"dst_w", "dst_h"} and value is None:
                        item.setToolTip("Blank uses auto default from the layer source/rotation shape")
                    if isinstance(value, bool):
                        item.setCheckState(Qt.Checked if value else Qt.Unchecked)
                        item.setText("")
                    self.layers_table.setItem(row, col, item)
            self.layers_table.blockSignals(False)

        def _schedule(self) -> None:
            self.debounce.start()

        def add_layer(self) -> None:
            self.config.layers.append(
                LayerConfig(
                    name="Layer",
                    src_w=1080,
                    src_h=1080,
                    fmt="RGB_4B",
                    compressed=False,
                    compression_mode="sajc",
                    hdr=False,
                    scaling=False,
                    scale_v=1.0,
                    rotation=False,
                    dpuf=0,
                    stream_coeff=1.0,
                )
            )
            self._populate_inputs()
            self.recalculate()

        def delete_layer(self) -> None:
            row = self.layers_table.currentRow()
            if row >= 0 and row < len(self.config.layers):
                del self.config.layers[row]
                self._populate_inputs()
                self.recalculate()

        def read_config(self) -> SimulatorConfig:
            panel_values: dict[str, Any] = {}
            for name, widget in self.panel_fields.items():
                panel_values[name] = widget.currentText() if isinstance(widget, QComboBox) else widget.text()
            system_values: dict[str, Any] = {}
            for name, widget in self.system_fields.items():
                system_values[name] = widget.currentText() if isinstance(widget, QComboBox) else widget.text()
            panel = PanelConfig(
                panel_w=int(panel_values["panel_w"]),
                panel_h=int(panel_values["panel_h"]),
                fps=float(panel_values["fps"]),
                disp_bpp=float(panel_values["disp_bpp"]),
                vbp=int(panel_values["vbp"]),
                vfp=int(panel_values["vfp"]),
                vsa=int(panel_values["vsa"]),
                dsc_mode=str(panel_values["dsc_mode"]),
                dst_y=int(panel_values["dst_y"]),
            )
            system = SystemConfig(
                PTW=float(system_values["PTW"]),
                MO_derating=float(system_values["MO_derating"]),
                MO_entries=int(system_values["MO_entries"]),
                MO_entry_bytes=int(system_values["MO_entry_bytes"]),
                OF_lines=int(system_values["OF_lines"]),
                margin=float(system_values["margin"]),
                bus_width_B=float(system_values["bus_width_B"]),
                bus_util=float(system_values["bus_util"]),
                max_bus_port_BW_MBs=float(system_values["max_bus_port_BW_MBs"]),
                ppc_comp=float(system_values["ppc_comp"]),
                ppc_scaler=float(system_values["ppc_scaler"]),
                ppc_outfifo=float(system_values["ppc_outfifo"]),
                ppc_pbld=float(system_values["ppc_pbld"]),
                ppc_ai_scaler=float(system_values["ppc_ai_scaler"]),
                ppc_wb=float(system_values["ppc_wb"]),
                vtap_scaler=float(system_values["vtap_scaler"]),
                vtap_outfifo=float(system_values["vtap_outfifo"]),
                stream_clk_overhead=float(system_values["stream_clk_overhead"]),
                ptw_group_mode=str(system_values["ptw_group_mode"]),
                ai_scaler_enabled=str(system_values["ai_scaler_enabled"]) == "True",
                wb_enabled=str(system_values["wb_enabled"]) == "True",
                dpuf_xres=_optional_int(system_values["dpuf_xres"]),
                dpuf_yres=_optional_int(system_values["dpuf_yres"]),
                dpu_xres=_optional_int(system_values["dpu_xres"]),
                dpu_yres=_optional_int(system_values["dpu_yres"]),
                wb_xres=int(system_values["wb_xres"]),
                wb_yres=int(system_values["wb_yres"]),
                overlay_count_overrides=self.config.system.overlay_count_overrides,
            )
            layers = []
            for row in range(self.layers_table.rowCount()):
                values: dict[str, Any] = {}
                for col, name in enumerate(self.layer_columns):
                    item = self.layers_table.item(row, col)
                    if name in {"compressed", "hdr", "scaling", "rotation"}:
                        values[name] = item.checkState() == Qt.Checked if item else False
                    else:
                        values[name] = item.text() if item else ""
                layers.append(
                    LayerConfig(
                        name=values["name"],
                        fmt=values["fmt"],
                        src_w=int(values["src_w"]),
                        src_h=int(values["src_h"]),
                        dst_w=_optional_int(values["dst_w"]),
                        dst_h=_optional_int(values["dst_h"]),
                        compressed=values["compressed"],
                        compression_mode=values["compression_mode"] or "sajc",
                        hdr=values["hdr"],
                        scaling=values["scaling"],
                        scale_v=float(values["scale_v"]),
                        rotation=values["rotation"],
                        dpuf=int(values["dpuf"]),
                        stream_coeff=float(values["stream_coeff"]),
                    )
                )
            return SimulatorConfig(system=system, panel=panel, layers=layers)

        def recalculate(self) -> None:
            try:
                self.config = self.read_config()
                result = solve(self.config)
                self.last_result = result
                if self.baseline_result is None:
                    self.baseline_result = copy.deepcopy(result)
            except Exception as exc:
                self.top_label.setText(f"Input error: {exc}")
                return
            result = self.last_result
            self._refresh_summary(result, self.baseline_result)
            self._refresh_breakdown(result, self.baseline_result)
            html_doc = build_formula_html(result)
            if hasattr(self.formula_view, "setHtml"):
                self.formula_view.setHtml(html_doc)
            else:
                self.formula_view.setText(html_doc)

        def set_baseline(self) -> None:
            if self.last_result is None:
                self.recalculate()
            if self.last_result is not None:
                self.baseline_result = copy.deepcopy(self.last_result)
                self._refresh_summary(self.last_result, self.baseline_result)
                self._refresh_breakdown(self.last_result, self.baseline_result)

        def _refresh_summary(self, result: Result, baseline: Result | None = None) -> None:
            delta = result_delta_summary(result, baseline)
            self.top_label.setText(
                "<div>"
                f"<span style='font-size:22px; font-weight:700;'>DPU_IB = {_fmt(result.dpu_ib_MBps)} MB/s</span>"
                f"<span style='font-size:13px; color:#57606a;'>&nbsp;&nbsp;&nbsp;max term: "
                f"<b>{html.escape(str(result.binding_term))}</b></span>"
                f"<span style='font-size:13px; color:#57606a;'>&nbsp;&nbsp;&nbsp;DPU_ACLK: "
                f"<b>{_fmt(result.clock['DPU_ACLK_MHz'])} MHz</b> ({html.escape(str(result.clock['aclk_binding']))})</span>"
                "</div>"
            )
            self.delta_label.setText(html.escape(delta["headline"]))
            for term in term_summary_rows(result, baseline):
                value_text = (
                    f"<b>{html.escape(term['display'])}</b> {term['unit']}"
                    if term["value"] is not None
                    else "<b>N/A</b>"
                )
                if term["delta_display"]:
                    value_text += f" &nbsp; <span style='color:#57606a;'>Δ {html.escape(term['delta_display'])}</span>"
                if term["is_binding"]:
                    text = f"{value_text} &nbsp; <span style='color:#bf8700; font-weight:700;'>MAX</span>"
                    style = (
                        "background: #fff8c5;"
                        "border: 2px solid #bf8700;"
                        "border-radius: 4px;"
                        "padding: 7px 9px;"
                        "font-size: 14px;"
                    )
                else:
                    text = value_text
                    style = (
                        "background: #ffffff;"
                        "border: 1px solid #d8dee4;"
                        "border-radius: 4px;"
                        "padding: 7px 9px;"
                        "font-size: 13px;"
                    )
                self.term_labels[term["key"]].setText(text)
                self.term_labels[term["key"]].setStyleSheet(style)

        def _refresh_breakdown(self, result: Result, baseline: Result | None = None) -> None:
            rows = breakdown_rows(result, baseline)

            self.breakdown.setRowCount(len(rows))
            self.breakdown.setColumnCount(6)
            self.breakdown.setHorizontalHeaderLabels(["group", "name", "model", "delta", "measured", "note"])
            header = self.breakdown.horizontalHeader()
            header.setSectionResizeMode(QHeaderView.Interactive)
            header.setStretchLastSection(False)
            for row, row_data in enumerate(rows):
                background = QColor(row_data["background"])
                values = [
                    row_data["group"],
                    row_data["name"],
                    row_data["model"],
                    row_data["delta"],
                    row_data["measured"],
                    row_data["note"],
                ]
                for col, text in enumerate(values):
                    item = QTableWidgetItem(text)
                    item.setBackground(background)
                    item.setTextAlignment(Qt.AlignCenter)
                    if col == 3 and text not in {"", "0.00", "same"}:
                        delta_font = QFont(item.font())
                        delta_font.setBold(True)
                        item.setFont(delta_font)
                    if col != 4:
                        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    self.breakdown.setItem(row, col, item)
            if not getattr(self, "_breakdown_columns_initialized", False):
                for col, width in enumerate([90, 210, 135, 90, 110, 260]):
                    self.breakdown.setColumnWidth(col, width)
                self._breakdown_columns_initialized = True

        def run_sweep(self) -> None:
            try:
                cfg = self.read_config()
                param2 = self.sweep_param2.currentText()
                self.sweep_rows = sweep(
                    cfg,
                    self.sweep_param.currentText(),
                    float(self.sweep_start.text()),
                    float(self.sweep_stop.text()),
                    float(self.sweep_step.text()),
                    param2 if self.sweep_2d.isChecked() else None,
                    float(self.sweep_start2.text()) if self.sweep_2d.isChecked() else None,
                    float(self.sweep_stop2.text()) if self.sweep_2d.isChecked() else None,
                    float(self.sweep_step2.text()) if self.sweep_2d.isChecked() else None,
                )
            except Exception as exc:
                QMessageBox.warning(self, "Sweep error", str(exc))
                return
            if self.figure is not None:
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                param = self.sweep_param.currentText()
                if self.sweep_2d.isChecked():
                    param2 = self.sweep_param2.currentText()
                    x_values = sorted({row[param] for row in self.sweep_rows})
                    y_values = sorted({row[param2] for row in self.sweep_rows})
                    grid = [[math.nan for _ in x_values] for _ in y_values]
                    x_index = {value: index for index, value in enumerate(x_values)}
                    y_index = {value: index for index, value in enumerate(y_values)}
                    for row in self.sweep_rows:
                        grid[y_index[row[param2]]][x_index[row[param]]] = row["DPU_IB_MBps"]
                    image = ax.imshow(
                        grid,
                        aspect="auto",
                        origin="lower",
                        extent=[min(x_values), max(x_values), min(y_values), max(y_values)],
                    )
                    ax.set_xlabel(param)
                    ax.set_ylabel(param2)
                    ax.set_title("DPU_IB heatmap (MB/s)")
                    self.figure.colorbar(image, ax=ax, label="DPU_IB MB/s")
                else:
                    ax.plot([row[param] for row in self.sweep_rows], [row["DPU_IB_MBps"] for row in self.sweep_rows], label="DPU_IB")
                    ax.plot([row[param] for row in self.sweep_rows], [row["IB_outfifo_preload_MBps"] for row in self.sweep_rows], label="outfifo")
                    ax.plot([row[param] for row in self.sweep_rows], [row["IB_streaming_MBps"] for row in self.sweep_rows], label="streaming")
                    ax.set_xlabel(param)
                    ax.set_ylabel("MB/s")
                    ax.legend()
                self.canvas.draw()
            else:
                self.canvas.setText("\n".join(str(row) for row in self.sweep_rows))

        def export_sweep_csv(self) -> None:
            if not self.sweep_rows:
                self.run_sweep()
            if not self.sweep_rows:
                return
            path, _ = QFileDialog.getSaveFileName(self, "Export sweep CSV", "sweep.csv", "CSV (*.csv)")
            if path:
                write_sweep_csv(self.sweep_rows, path)

    app = QApplication.instance() or QApplication(sys.argv)
    window = SimulatorWindow(config)
    window.resize(1320, 820)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
