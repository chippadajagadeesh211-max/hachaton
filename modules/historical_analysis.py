"""
SlopeGuard AI — Historical Landslide Inventory Analysis
Queries Geological Survey of India (GSI) inventory records near a target site.
"""

import numpy as np
import pandas as pd


def query_nearby_historical_landslides(
    target_lat: float,
    target_lon: float,
    inventory_df: pd.DataFrame,
    radius_km: float = 10.0,
    max_results: int = 50
) -> dict:
    """
    Finds historical landslide events within radius_km using spherical trigonometry.
    """
    if inventory_df.empty:
        return {
            'count': 0,
            'records': [],
            'message': 'No historical landslide record found in the available inventory.'
        }

    lats_rad = np.radians(inventory_df['latitude'].values)
    lons_rad = np.radians(inventory_df['longitude'].values)
    t_lat_rad = np.radians(target_lat)
    t_lon_rad = np.radians(target_lon)

    dlat = lats_rad - t_lat_rad
    dlon = lons_rad - t_lon_rad

    a = np.sin(dlat / 2.0)**2 + np.cos(t_lat_rad) * np.cos(lats_rad) * np.sin(dlon / 2.0)**2
    c = 2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))
    dist_km = 6371.0 * c

    mask = dist_km <= radius_km
    indices = np.where(mask)[0]

    if len(indices) == 0:
        nearest_idx = np.argmin(dist_km)
        nearest_dist = float(dist_km[nearest_idx])
        nearest_row = inventory_df.iloc[nearest_idx]
        return {
            'count': 0,
            'radius_km': radius_km,
            'records': [],
            'nearest_outside': {
                'district': str(nearest_row.get('district', 'Himachal Pradesh')),
                'material': str(nearest_row.get('material', 'Rock/Debris')),
                'movement': str(nearest_row.get('movement', 'Slide')),
                'year': str(int(nearest_row['year'])) if pd.notnull(nearest_row.get('year')) else 'Unrecorded',
                'distance_km': round(nearest_dist, 2)
            },
            'message': f"No historical landslide record found within {radius_km} km in the available inventory. Nearest cataloged event is {round(nearest_dist, 2)} km away in {nearest_row.get('district', 'the region')}."
        }

    sub_dist = dist_km[indices]
    sort_order = np.argsort(sub_dist)
    sorted_indices = indices[sort_order][:max_results]

    records = []
    for idx in sorted_indices:
        row = inventory_df.iloc[idx]
        d = float(dist_km[idx])
        records.append({
            'sno': int(row['sno']) if pd.notnull(row.get('sno')) else None,
            'slide_name': str(row['slide_name']) if pd.notnull(row.get('slide_name')) and row['slide_name'] != 'nan' else 'Unnamed Historical Slide',
            'district': str(row['district']) if pd.notnull(row.get('district')) else 'Unspecified',
            'material': str(row['material']) if pd.notnull(row.get('material')) else 'Debris/Rock',
            'movement': str(row['movement']) if pd.notnull(row.get('movement')) else 'Slide',
            'year': str(int(row['year'])) if pd.notnull(row.get('year')) else 'Unrecorded',
            'latitude': round(float(row['latitude']), 5),
            'longitude': round(float(row['longitude']), 5),
            'distance_km': round(d, 2),
            'distance_m': round(d * 1000, 0),
            'event_note': str(row['event_note']) if pd.notnull(row.get('event_note')) and row['event_note'] != 'nan' else None
        })

    return {
        'count': len(indices),
        'radius_km': radius_km,
        'records': records,
        'message': f"Found {len(indices)} historical landslide record(s) within {radius_km} km."
    }
