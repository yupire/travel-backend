"""景点室内外判断工具

根据景点的 typecode（高德 POI 分类）、类型关键词、或 mock 数据中的
`type` / `is_indoor` 字段，判断一个景点属于室内（indoor）还是室外（outdoor）。

判断优先级：
1. 显式 is_indoor 字段
2. 显式 type 字段（"indoor" / "outdoor"）
3. typecode 匹配（高德 POI 分类码）
4. 名称/标签关键词兜底
"""
from __future__ import annotations

import importlib.util
import logging
import os
from typing import Dict, List, Optional

# 从 spots / amap 模块导入（使用 try/except 避免触发 tools/__init__.py）
try:
    from tools.spots import _INDOOR_TYPECODES, get_top_spots
except ImportError:
    spec = importlib.util.spec_from_file_location(
        "spots",
        os.path.join(os.path.dirname(__file__), "spots.py"),
    )
    spots_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(spots_mod)
    _INDOOR_TYPECODES = spots_mod._INDOOR_TYPECODES
    get_top_spots = spots_mod.get_top_spots

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 名称 / 标签关键词兜底
# ---------------------------------------------------------------------------

# 强室内信号关键词
_INDOOR_KEYWORDS = {
    # 中文
    "博物馆", "纪念馆", "展览馆", "美术馆", "艺术馆", "科技馆",
    "水族馆", "图书馆", "文化馆", "文化中心", "剧院", "大剧院",
    "歌剧院", "音乐厅", "电影院", "影城", "商场", "购物中心",
    "商业街", "室内", "教堂", "寺庙", "宫", "殿", "塔", "陵", "寺",
    # 英文
    "museum", "gallery", "aquarium", "library", "theater", "theatre",
    "cinema", "mall", "shopping", "mall", "cathedral", "church",
    "temple", "palace", "hall", "indoor",
}

# 强室外信号关键词
_OUTDOOR_KEYWORDS = {
    # 中文
    "公园", "山", "海", "湖", "岛", "沙滩", "海滩", "花园", "广场",
    "古镇", "古街", "长城", "故宫", "天坛", "颐和园", "动物园",
    "植物园", "自然", "景区", "风景区", "度假", "滑雪", "温泉",
    # 英文
    "park", "mountain", "sea", "lake", "island", "beach", "garden",
    "square", "zoo", "safari", "outdoor", "resort", "ski",
}


# ---------------------------------------------------------------------------
# 核心判断函数
# ---------------------------------------------------------------------------

def classify_spot_indoor(
    *,
    is_indoor: Optional[bool] = None,
    spot_type: Optional[str] = None,
    typecode: Optional[str] = None,
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> bool:
    """根据一个景点的多个属性判断其是否为室内景点。

    判断优先级（先到先得）：
    1. 显式 is_indoor 字段
    2. spot_type 字段（"indoor" / "outdoor"）
    3. typecode 匹配高德 POI 室内分类
    4. 名称 / 标签关键词兜底（先匹配室内关键词，再匹配室外关键词）

    Args:
        is_indoor: 显式的室内标志
        spot_type: 显式类型字符串，取值为 "indoor" / "outdoor"
        typecode: 高德 POI 分类编码（如 "110103"）
        name: 景点名称（用于关键词兜底）
        tags: 景点标签列表（用于关键词兜底）

    Returns:
        bool: True 表示室内景点，False 表示室外景点
    """
    # 1. 显式 is_indoor
    if is_indoor is not None:
        return bool(is_indoor)

    # 2. 显式 type 字段
    if spot_type in ("indoor", "outdoor"):
        return spot_type == "indoor"

    # 3. typecode 匹配
    if typecode and typecode in _INDOOR_TYPECODES:
        return True
    if typecode:
        # 非室内 typecode 视为室外
        return False

    # 4. 关键词兜底
    haystack_parts = []
    if name:
        haystack_parts.append(name.lower())
    if tags:
        haystack_parts.extend(t.lower() for t in tags)
    if not haystack_parts:
        # 没有任何线索，默认室外（旅游景点大多是室外）
        return False
    haystack = " ".join(haystack_parts)

    # 室内关键词优先匹配
    for kw in _INDOOR_KEYWORDS:
        if kw.lower() in haystack:
            return True
    for kw in _OUTDOOR_KEYWORDS:
        if kw.lower() in haystack:
            return False

    # 兜底：默认室外
    return False


# ---------------------------------------------------------------------------
# 基于城市景点数据的批量判断
# ---------------------------------------------------------------------------

def classify_spots_by_city(city: str, spot_names: Optional[List[str]] = None) -> List[Dict]:
    """判断一个城市若干景点（按名称匹配）的室内外属性。

    数据来源：复用 get_top_spots() —— 中国城市走高德 API，其他城市走 mock。

    Args:
        city: 城市名称（中文或拼音）
        spot_names: 可选，景点名称过滤列表。None 表示返回该城市全部景点。

    Returns:
        List[Dict]: 每个元素为 {"id", "name", "is_indoor", "type"}
    """
    spots = get_top_spots(city)
    if spot_names:
        name_set = {n for n in spot_names}
        spots = [s for s in spots if s.get("name") in name_set or s.get("name_en") in name_set]

    results = []
    for s in spots:
        is_indoor = classify_spot_indoor(
            is_indoor=s.get("is_indoor"),
            spot_type=s.get("type"),
            name=s.get("name", ""),
            tags=s.get("tags", []),
        )
        results.append({
            "id": s.get("id", ""),
            "name": s.get("name", ""),
            "is_indoor": is_indoor,
            "type": "indoor" if is_indoor else "outdoor",
        })
    return results
