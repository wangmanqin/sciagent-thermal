"""
几何 / 体积 / 面积工具：散热器、通道、翅片阵列的几何参数计算。

这些计算单独看都不复杂，但在一个综合散热器设计题里要算十几处，
做成工具能减少 Agent 重复写样板代码。
"""

from __future__ import annotations
import math
import json


# ---------------------------------------------------------------------------
# 1) 截面几何
# ---------------------------------------------------------------------------

def rectangular_cross_section(width_m: float, height_m: float) -> dict:
    if min(width_m, height_m) <= 0:
        raise ValueError("width 和 height 必须为正")
    A = width_m * height_m
    P = 2 * (width_m + height_m)
    Dh = 4 * A / P
    return {
        "cross_section_area_m2": A,
        "wetted_perimeter_m": P,
        "hydraulic_diameter_m": Dh,
        "aspect_ratio": min(width_m, height_m) / max(width_m, height_m),
    }


def circular_cross_section(diameter_m: float) -> dict:
    if diameter_m <= 0:
        raise ValueError("diameter 必须为正")
    A = math.pi * diameter_m ** 2 / 4
    P = math.pi * diameter_m
    return {
        "cross_section_area_m2": A,
        "wetted_perimeter_m": P,
        "hydraulic_diameter_m": diameter_m,
    }


def triangular_cross_section_equilateral(side_m: float) -> dict:
    if side_m <= 0:
        raise ValueError("边长必须为正")
    A = math.sqrt(3) / 4 * side_m ** 2
    P = 3 * side_m
    return {
        "cross_section_area_m2": A,
        "wetted_perimeter_m": P,
        "hydraulic_diameter_m": 4 * A / P,
    }


def trapezoidal_cross_section(
    top_width_m: float, bottom_width_m: float, height_m: float,
) -> dict:
    if min(top_width_m, bottom_width_m, height_m) <= 0:
        raise ValueError("参数必须为正")
    A = 0.5 * (top_width_m + bottom_width_m) * height_m
    side = math.sqrt(height_m ** 2 + ((top_width_m - bottom_width_m) / 2) ** 2)
    P = top_width_m + bottom_width_m + 2 * side
    return {
        "cross_section_area_m2": A,
        "wetted_perimeter_m": P,
        "hydraulic_diameter_m": 4 * A / P,
        "side_length_m": side,
    }


# ---------------------------------------------------------------------------
# 2) 通道阵列（微通道散热器）
# ---------------------------------------------------------------------------

def channel_array(
    channel_width_m: float,
    channel_height_m: float,
    wall_thickness_m: float,
    sink_width_m: float,
    sink_length_m: float,
) -> dict:
    """
    计算矩形通道 + 固体壁 交替排列的散热器几何。

    返回：通道数、每通道流通面积、总流通面积、换热面积（通道四壁）
    """
    pitch = channel_width_m + wall_thickness_m
    n_channels = int(math.floor((sink_width_m + wall_thickness_m) / pitch))
    if n_channels <= 0:
        raise ValueError("通道数 <= 0，请检查几何参数")

    per_channel_area = channel_width_m * channel_height_m
    total_flow_area = per_channel_area * n_channels

    # 湿周（每根通道 4 壁）× 长度 × 通道数
    channel_perimeter = 2 * (channel_width_m + channel_height_m)
    heat_transfer_area = channel_perimeter * sink_length_m * n_channels

    return {
        "n_channels": n_channels,
        "channel_width_m": channel_width_m,
        "channel_height_m": channel_height_m,
        "wall_thickness_m": wall_thickness_m,
        "pitch_m": pitch,
        "per_channel_flow_area_m2": per_channel_area,
        "total_flow_area_m2": total_flow_area,
        "channel_perimeter_m": channel_perimeter,
        "heat_transfer_area_m2": heat_transfer_area,
        "hydraulic_diameter_m": 4 * per_channel_area / channel_perimeter,
    }


# ---------------------------------------------------------------------------
# 3) 翅片阵列
# ---------------------------------------------------------------------------

def fin_array(
    fin_thickness_m: float,
    fin_height_m: float,
    fin_spacing_m: float,
    base_width_m: float,
    base_length_m: float,
    fin_count: int = None,
) -> dict:
    pitch = fin_thickness_m + fin_spacing_m
    if fin_count is None:
        fin_count = int(math.floor((base_width_m + fin_spacing_m) / pitch))
    if fin_count <= 0:
        raise ValueError("翅片数 <= 0")

    # 单翅片传热面积（两面 + 顶面，底面与基板相连不计）
    single_fin_area = (
        2 * fin_height_m * base_length_m  # 两个侧面
        + fin_thickness_m * base_length_m  # 顶面
    )
    total_fin_area = single_fin_area * fin_count

    # 基板上未被翅片覆盖的部分
    fin_foot_area = fin_thickness_m * base_length_m * fin_count
    exposed_base_area = base_width_m * base_length_m - fin_foot_area

    return {
        "fin_count": fin_count,
        "fin_pitch_m": pitch,
        "single_fin_area_m2": single_fin_area,
        "total_fin_area_m2": total_fin_area,
        "exposed_base_area_m2": exposed_base_area,
        "total_surface_area_m2": total_fin_area + exposed_base_area,
    }


# ---------------------------------------------------------------------------
# 4) 辅助
# ---------------------------------------------------------------------------

def sphere_volume(diameter_m: float) -> dict:
    V = math.pi * diameter_m ** 3 / 6
    A = math.pi * diameter_m ** 2
    return {"volume_m3": V, "surface_area_m2": A,
            "diameter_m": diameter_m}


def cylinder_volume(
    diameter_m: float, length_m: float, ends: str = "both",
) -> dict:
    A_lat = math.pi * diameter_m * length_m
    A_end = math.pi * diameter_m ** 2 / 4
    if ends == "both":
        A_total = A_lat + 2 * A_end
    elif ends == "one":
        A_total = A_lat + A_end
    elif ends == "none":
        A_total = A_lat
    else:
        raise ValueError("ends 必须是 both / one / none")
    V = A_end * length_m
    return {"volume_m3": V,
            "lateral_area_m2": A_lat,
            "end_area_m2": A_end,
            "total_surface_area_m2": A_total}


# ---------------------------------------------------------------------------
# 工具注册
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "rectangular_cross_section",
        "description": "矩形截面的面积、湿周、水力直径、长宽比。",
        "input_schema": {
            "type": "object",
            "properties": {
                "width_m": {"type": "number"},
                "height_m": {"type": "number"},
            },
            "required": ["width_m", "height_m"],
        },
    },
    {
        "name": "circular_cross_section",
        "description": "圆形截面几何参数。",
        "input_schema": {
            "type": "object",
            "properties": {"diameter_m": {"type": "number"}},
            "required": ["diameter_m"],
        },
    },
    {
        "name": "triangular_cross_section_equilateral",
        "description": "等边三角形截面几何参数。",
        "input_schema": {
            "type": "object",
            "properties": {"side_m": {"type": "number"}},
            "required": ["side_m"],
        },
    },
    {
        "name": "trapezoidal_cross_section",
        "description": "梯形截面几何参数（上/下底 + 高）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "top_width_m": {"type": "number"},
                "bottom_width_m": {"type": "number"},
                "height_m": {"type": "number"},
            },
            "required": ["top_width_m", "bottom_width_m", "height_m"],
        },
    },
    {
        "name": "channel_array",
        "description": (
            "矩形微通道阵列散热器几何：通道数、单通道/总流通面积、"
            "换热面积、水力直径。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "channel_width_m": {"type": "number"},
                "channel_height_m": {"type": "number"},
                "wall_thickness_m": {"type": "number"},
                "sink_width_m": {"type": "number"},
                "sink_length_m": {"type": "number"},
            },
            "required": ["channel_width_m", "channel_height_m",
                         "wall_thickness_m", "sink_width_m", "sink_length_m"],
        },
    },
    {
        "name": "fin_array",
        "description": "直翅片阵列总换热面积、翅根面积等几何参数。",
        "input_schema": {
            "type": "object",
            "properties": {
                "fin_thickness_m": {"type": "number"},
                "fin_height_m": {"type": "number"},
                "fin_spacing_m": {"type": "number"},
                "base_width_m": {"type": "number"},
                "base_length_m": {"type": "number"},
                "fin_count": {"type": "integer"},
            },
            "required": ["fin_thickness_m", "fin_height_m", "fin_spacing_m",
                         "base_width_m", "base_length_m"],
        },
    },
    {
        "name": "sphere_volume",
        "description": "球体体积与表面积。",
        "input_schema": {
            "type": "object",
            "properties": {"diameter_m": {"type": "number"}},
            "required": ["diameter_m"],
        },
    },
    {
        "name": "cylinder_volume",
        "description": "圆柱体体积与表面积（ends=both/one/none 决定端面计入）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "diameter_m": {"type": "number"},
                "length_m": {"type": "number"},
                "ends": {"type": "string",
                         "enum": ["both", "one", "none"]},
            },
            "required": ["diameter_m", "length_m"],
        },
    },
]


def _wrap(fn):
    def _exec(args):
        return json.dumps(fn(**args), ensure_ascii=False, indent=2)
    return _exec


TOOL_EXECUTORS = {
    "rectangular_cross_section": _wrap(rectangular_cross_section),
    "circular_cross_section": _wrap(circular_cross_section),
    "triangular_cross_section_equilateral":
        _wrap(triangular_cross_section_equilateral),
    "trapezoidal_cross_section": _wrap(trapezoidal_cross_section),
    "channel_array": _wrap(channel_array),
    "fin_array": _wrap(fin_array),
    "sphere_volume": _wrap(sphere_volume),
    "cylinder_volume": _wrap(cylinder_volume),
}
