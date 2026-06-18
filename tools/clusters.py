"""景点地理聚类 — DBSCAN（按密度）+ GeoJSON 导出

依赖查询结果（spots 列表），输入元素至少需要 id/name/lat/lng，
可选项 is_indoor（或 type='indoor'/'outdoor'）。

支持：
- 优先使用 scikit-learn 的 DBSCAN（已安装时）
- 未安装时降级为纯 Python DBSCAN 实现
- 输出每个簇的中心点、覆盖半径、室内/外数量
- 可导出为 GeoJSON FeatureCollection，方便地图可视化
"""
from __future__ import annotations

import json
import logging
import math
import os
from typing import Dict, List, Optional

# 复用项目内已有 Haversine 实现
from tools.routing import haversine_km

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# sklearn 可选导入
# ---------------------------------------------------------------------------
try:
    import numpy as np  # noqa: F401  — sklearn 强依赖
    from sklearn.cluster import DBSCAN as _SkDBSCAN
    _HAS_SKLEARN = True
except Exception:  # ImportError 或 numpy 缺失等
    _SkDBSCAN = None
    _HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# DBSCAN 默认参数
# ---------------------------------------------------------------------------
DEFAULT_EPS_KM = 3.0
DEFAULT_MIN_SAMPLES = 2

# 室内/外判断字段（兼容 spots.py 返回的两种字段）
def _is_indoor_spot(spot: Dict) -> bool:
    if "is_indoor" in spot:
        return bool(spot["is_indoor"])
    return spot.get("type") == "indoor"


# ---------------------------------------------------------------------------
# 纯 Python DBSCAN 实现（sklearn 不可用时降级）
# ---------------------------------------------------------------------------
def _dbscan_python(
    coords: List[List[float]],
    eps_km: float,
    min_samples: int,
) -> List[int]:
    """纯 Python DBSCAN，输入坐标为 [[lat, lng], ...]，返回每个点的簇标签 (-1 = 噪声)"""
    n = len(coords)
    labels = [None] * n           # None 表示未访问
    cluster_id = 0

    def neighbors(i: int) -> List[int]:
        lat_i, lng_i = coords[i]
        result = []
        for j in range(n):
            if i == j:
                continue
            lat_j, lng_j = coords[j]
            if haversine_km(lat_i, lng_i, lat_j, lng_j) <= eps_km:
                result.append(j)
        return result

    for i in range(n):
        if labels[i] is not None:
            continue
        neigh = neighbors(i)
        if len(neigh) < min_samples:
            labels[i] = -1  # 噪声（暂定，后续可能被边界点 grab）
            continue
        # 创建新簇
        labels[i] = cluster_id
        seeds = list(neigh)
        seed_idx = 0
        while seed_idx < len(seeds):
            q = seeds[seed_idx]
            seed_idx += 1
            if labels[q] == -1:
                labels[q] = cluster_id  # 噪声升级为边界点
            elif labels[q] is None:
                labels[q] = cluster_id
                q_neigh = neighbors(q)
                if len(q_neigh) >= min_samples:
                    seeds.extend(q_neigh)
        cluster_id += 1

    return [lbl if lbl is not None else -1 for lbl in labels]


def _dbscan_sklearn(
    coords_rad: List[List[float]],
    eps_km: float,
    min_samples: int,
) -> List[int]:
    """sklearn DBSCAN；输入坐标已转为弧度 (lat, lng → radians)"""
    import numpy as np
    arr = np.asarray(coords_rad, dtype=float)
    # sklearn 的 haversine metric：把 km 转成弧度 (R=6371.0)
    eps_rad = eps_km / 6371.0
    model = _SkDBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine")
    return model.fit_predict(arr).tolist()


# ---------------------------------------------------------------------------
# 聚类主函数
# ---------------------------------------------------------------------------
def cluster_spots_by_geo(
    spots: List[Dict],
    eps_km: float = DEFAULT_EPS_KM,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> List[Dict]:
    """按地理坐标对景点做 DBSCAN 聚类。

    输入：
        spots: 景点列表（依赖 get_top_spots / geocode_spots 的结果）
        eps_km: 邻域半径（公里）
        min_samples: 形成簇所需的最少样本数

    输出：
        List[Dict]，每个元素代表一个簇，字段：
            cluster_id   int   簇编号（>= 0 为正常簇；< 0 为 DBSCAN
                                噪声/边界点按地理连通性再拆出的子簇，
                                cluster_id < 0 即 is_noise=True）
            size         int   簇内景点数（< 2 的孤点已过滤）
            center       {lat, lng}  簇质心
            spread_km    float 簇内最远点对距离（覆盖半径）
            indoor_count int
            outdoor_count int
            spot_ids     [str]
            spot_names   [str]
    """
    # 逻辑流程：
    #   1. 过滤掉缺失经纬度的无效景点；
    #   2. 调用 DBSCAN（优先 sklearn，失败/缺失则降级纯 Python）得到每点的簇标签；
    #   3. 把标签为 -1 的噪声点按 eps 邻域连通性再拆成地理相邻的子簇；
    #   4. 过滤孤点（size < 2），计算每簇质心 / 覆盖半径 / 室内外数量；
    #   5. 按 size 降序排序后返回（噪声簇置于最后）。
    log.info("开始地理聚类: 输入 %d 个景点, eps_km=%.2f, min_samples=%d",
             len(spots), eps_km, min_samples)

    if not spots:
        return []

    # 过滤无效坐标
    valid_spots = [s for s in spots if s.get("lat") is not None and s.get("lng") is not None]
    if not valid_spots:
        log.warning("无有效坐标景点，聚类结果为空")
        return []
    if len(valid_spots) < len(spots):
        log.debug("过滤掉 %d 个缺失坐标的景点，剩余 %d 个",
                  len(spots) - len(valid_spots), len(valid_spots))

    coords = [[s["lat"], s["lng"]] for s in valid_spots]

    if _HAS_SKLEARN:
        coords_rad = [[math.radians(c[0]), math.radians(c[1])] for c in coords]
        try:
            labels = _dbscan_sklearn(coords_rad, eps_km, min_samples)
            log.debug("使用 sklearn DBSCAN 完成聚类")
        except Exception as e:
            log.warning("sklearn DBSCAN 失败，降级到纯 Python 实现: %s", e)
            labels = _dbscan_python(coords, eps_km, min_samples)
    else:
        log.debug("sklearn 未安装，使用纯 Python DBSCAN")
        labels = _dbscan_python(coords, eps_km, min_samples)

    # 按 cluster_id 分组
    clusters: Dict[int, List[Dict]] = {}
    for spot, lbl in zip(valid_spots, labels):
        clusters.setdefault(int(lbl), []).append(spot)
    log.debug("DBSCAN 原始分组: %d 个簇, 噪声点 %d 个",
              len([k for k in clusters if k >= 0]), len(clusters.get(-1, [])))

    # DBSCAN 把"邻居不足"的点全标 -1，但它们可能分布在完全不同的城市。
    # 这里按 eps 邻域连通性把 -1 组再拆成子组，让下游拿到的是真正地理相邻的簇。
    if -1 in clusters and len(clusters[-1]) > 1:
        noise_spots = clusters.pop(-1)
        sub_id = -1
        visited = set()
        for i, s in enumerate(noise_spots):
            if i in visited:
                continue
            # BFS 找与 s 在 eps 内连通的未访问点
            component = [s]
            visited.add(i)
            queue = [i]
            while queue:
                k = queue.pop()
                for j, t in enumerate(noise_spots):
                    if j in visited:
                        continue
                    if haversine_km(
                        noise_spots[k]["lat"], noise_spots[k]["lng"],
                        t["lat"], t["lng"],
                    ) <= eps_km:
                        component.append(t)
                        visited.add(j)
                        queue.append(j)
            clusters[sub_id] = component
            sub_id -= 1
        log.debug("噪声点按连通性拆分为 %d 个子簇", -1 - sub_id)

    # 组装输出
    result: List[Dict] = []
    for lbl, members in clusters.items():
        # 过滤孤点：单点成簇对行程规划无意义
        if len(members) < 2:
            continue

        n = len(members)
        avg_lat = sum(m["lat"] for m in members) / n
        avg_lng = sum(m["lng"] for m in members) / n

        # 簇内最远点对距离
        max_dist = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                d = haversine_km(
                    members[i]["lat"], members[i]["lng"],
                    members[j]["lat"], members[j]["lng"],
                )
                if d > max_dist:
                    max_dist = d

        indoor = sum(1 for m in members if _is_indoor_spot(m))
        result.append({
            "cluster_id": lbl,
            "size": n,
            "center": {"lat": avg_lat, "lng": avg_lng},
            "spread_km": round(max_dist, 3),
            "indoor_count": indoor,
            "outdoor_count": n - indoor,
            "spot_ids": [m["id"] for m in members],
            "spot_names": [m["name"] for m in members],
        })

    # 簇按 size 降序、cluster_id 升序排（噪声簇放最后）
    result.sort(key=lambda c: (c["cluster_id"] == -1, -c["size"], c["cluster_id"]))
    log.info("聚类完成: 生成 %d 个有效簇（已过滤孤点）", len(result))
    return result


# ---------------------------------------------------------------------------
# GeoJSON 导出
# ---------------------------------------------------------------------------
def export_clusters_to_geojson(
    clusters: List[Dict],
    output_path: Optional[str] = None,
) -> Dict:
    """将聚类结果导出为 GeoJSON FeatureCollection。

    每个簇产出一个 Feature：
        - Point（簇中心）
        - properties: cluster_id, size, spread_km, indoor_count, outdoor_count, spot_names

    参数：
        clusters: cluster_spots_by_geo 的返回
        output_path: 落盘路径（None 时只返回 dict，不写文件）

    返回：
        GeoJSON FeatureCollection dict
    """
    # 逻辑流程：遍历每个簇 → 以簇中心生成一个 Point Feature（坐标按 GeoJSON
    # [lng, lat] 顺序）→ 汇总为 FeatureCollection → 可选落盘。
    log.info("导出聚类 GeoJSON: %d 个簇%s",
             len(clusters), f", 写入 {output_path}" if output_path else "（仅返回 dict）")

    features = []
    for c in clusters:
        center = c["center"]
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [center["lng"], center["lat"]],  # GeoJSON: [lng, lat]
            },
            "properties": {
                "cluster_id": c["cluster_id"],
                "size": c["size"],
                "spread_km": c["spread_km"],
                "indoor_count": c["indoor_count"],
                "outdoor_count": c["outdoor_count"],
                "spot_names": c["spot_names"],
                "is_noise": c["cluster_id"] < 0,
            },
        })

    fc = {
        "type": "FeatureCollection",
        "features": features,
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(fc, f, ensure_ascii=False, indent=2)
        log.info("聚类 GeoJSON 已导出: %s", output_path)

    return fc
