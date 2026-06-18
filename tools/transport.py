from typing import Dict


def get_transport(distance_km: float) -> Dict:
    """根据距离估算交通方式与信息（基于距离区间的启发式估算）。"""
    if distance_km < 0.8:
        return {
            "mode": "步行",
            "duration_min": max(5, int(distance_km * 15)),
            "cost": 0.0,
            "distance_km": round(distance_km, 2),
        }
    elif distance_km < 5.0:
        return {
            "mode": "地铁/公交",
            "duration_min": int(distance_km * 5 + 10),
            "cost": round(1.5 + distance_km * 0.3, 1),
            "distance_km": round(distance_km, 2),
        }
    elif distance_km < 20.0:
        return {
            "mode": "打车",
            "duration_min": int(distance_km * 3 + 5),
            "cost": round(distance_km * 2.0, 1),
            "distance_km": round(distance_km, 2),
        }
    else:
        return {
            "mode": "大巴/专车",
            "duration_min": int(distance_km * 2 + 20),
            "cost": round(distance_km * 1.5, 1),
            "distance_km": round(distance_km, 2),
        }
