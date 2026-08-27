"""Plain-language definitions for terms used across the dashboard."""

README_INTRO = (
    "This dashboard tracks wildfire events over time. Each event moves through "
    "a lifecycle: **detected** (fire found) → **active** (still burning) → "
    "**ended** (no new detections) → **recovering** (vegetation regrowth) → "
    "**complete**. The satellite progression shows Sentinel-2 images before, "
    "during, and after the fire with vegetation/burn indices overlaid. Images "
    "are fetched on demand and cached locally."
)

TERM_GROUPS = {
    "Satellite & data sources": [
        (
            "Sentinel-2",
            "European satellite pair (S2A/S2B) with high-resolution optical imagery; the main image source.",
        ),
        (
            "L2A",
            "Sentinel-2 'surface reflectance' product with atmospheric correction.",
        ),
        ("VIIRS", "NASA/NOAA instrument detecting active fires at ~375 m resolution."),
        ("MODIS", "NASA instrument detecting active fires at ~1 km resolution."),
        ("NIR", "Near-infrared band; strongly reflected by healthy vegetation."),
        (
            "SWIR",
            "Shortwave-infrared band; sensitive to moisture and useful for detecting burns.",
        ),
        (
            "FIRMS",
            "NASA service delivering active-fire detections used to find and trigger fires.",
        ),
        ("NRT", "Near-Real-Time FIRMS data covering roughly the last 3 months."),
        ("SP", "Standard-Processing FIRMS archive reaching back years of history."),
        (
            "leastCC",
            "Mosaicking order that picks the least-cloudy Sentinel scene in a window.",
        ),
        (
            "WGS84",
            "Geographic coordinate system (latitude/longitude) used for all locations.",
        ),
    ],
    "Vegetation & burn indices": [
        (
            "NDVI",
            "Normalized Difference Vegetation Index: a -1..1 greenness measure; high values = dense healthy vegetation.",
        ),
        (
            "NDWI",
            "Normalized Difference Water Index: reflects vegetation/moisture content.",
        ),
        (
            "NBR",
            "Normalized Burn Ratio: combines NIR and SWIR to highlight burned areas.",
        ),
        (
            "dNBR",
            "Change in NBR (pre-fire minus post-fire); higher values = more severe burn.",
        ),
        (
            "ΔNDVI",
            "Per-pixel NDVI change relative to the pre-fire image; negative values mean vegetation loss.",
        ),
    ],
    "Fire metrics": [
        (
            "FRP",
            "Fire Radiative Power: heat output of a fire, reported in megawatts (MW).",
        ),
        ("MW", "Megawatt: the unit of FRP."),
        (
            "detection_count",
            "Number of satellite fire detections grouped into this event.",
        ),
        ("quiet_days", "Consecutive days with no new fire detection for this event."),
        (
            "weather_index",
            "0-100 fire-danger proxy derived from temperature, wind, precipitation, and soil moisture.",
        ),
        ("burned_area_ha", "Estimated burned area in hectares (1 ha = 0.01 km²)."),
        (
            "regrowth_ratio",
            "Recovery NDVI divided by pre-fire NDVI; a value over 1 means greener than before the fire.",
        ),
        (
            "bbox_growth_deg",
            "Daily spatial spread of detections (longitude + latitude span in degrees).",
        ),
    ],
    "Analysis concepts": [
        ("bbox", "Bounding box: the rectangle enclosing the fire's detections."),
        (
            "cluster",
            "A group of nearby fire detections treated as a single fire event.",
        ),
        ("cluster_id", "ID of the fire cluster produced by spatial grouping (DBSCAN)."),
        ("centroid", "Geographic center of the fire's detections."),
        (
            "event_id",
            "Unique identifier of the tracked event (first 8 characters shown).",
        ),
    ],
    "Lifecycle statuses": [
        ("detected", "Fires found via FIRMS; pre-fire analysis is queued."),
        (
            "active",
            "Still producing new detections; during-phase FRP is being collected.",
        ),
        ("ended", "No new detections; post-fire severity assessment runs next."),
        (
            "recovering",
            "Vegetation regrowth is sampled at 1, 3, 6, 9, and 12 months after the fire.",
        ),
        ("complete", "All recovery samples have been collected."),
    ],
    "Burn severity classes": [
        ("unburned", "dNBR below 0.10 — no burn signal."),
        ("low", "dNBR 0.10-0.27 — light scorch."),
        ("moderate", "dNBR 0.27-0.44 — canopy scorch / partial burn."),
        ("high", "dNBR 0.44-0.66 — heavy burn."),
        ("very_high", "dNBR at least 0.66 — severe burn."),
    ],
}

GLOSSARY = {
    term: definition for entries in TERM_GROUPS.values() for term, definition in entries
}
