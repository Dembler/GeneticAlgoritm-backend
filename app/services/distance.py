from __future__ import annotations

import math

from app.domain.models import Point


def haversine_km(a: Point, b: Point) -> float:
    radius_km = 6371.0088
    lat1 = math.radians(a.lat)
    lon1 = math.radians(a.lon)
    lat2 = math.radians(b.lat)
    lon2 = math.radians(b.lon)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    sin_dlat = math.sin(dlat / 2)
    sin_dlon = math.sin(dlon / 2)

    h = sin_dlat * sin_dlat + math.cos(lat1) * math.cos(lat2) * sin_dlon * sin_dlon
    return 2 * radius_km * math.asin(min(1, math.sqrt(h)))


def path_distance_km(points: list[Point]) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    for idx in range(len(points) - 1):
        total += haversine_km(points[idx], points[idx + 1])
    return total