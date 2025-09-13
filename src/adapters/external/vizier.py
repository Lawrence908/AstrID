"""
Thin Vizier wrappers for catalog queries and simple crossmatching utilities.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

try:
    from astroquery.vizier import Vizier
except Exception:  # pragma: no cover - optional dependency at runtime
    Vizier = None  # type: ignore[assignment]


@dataclass
class CatalogMatch:
    catalog: str
    ra_deg: float
    dec_deg: float
    separation_arcsec: float
    row: dict[str, Any]


def query_vizier_region(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    catalogs: Sequence[str] | None = None,
    row_limit: int = 200,
) -> list[Any]:
    """Query Vizier region; returns list of Astropy Tables (per catalog).

    Returns an empty list if Vizier is not installed or the query fails.
    """
    if Vizier is None:
        return []
    try:
        viz = Vizier(columns=["**"], row_limit=row_limit)
        result = viz.query_region(
            f"{ra_deg} {dec_deg}", radius=f"{radius_deg}d", catalog=list(catalogs or [])
        )
        return list(result or [])
    except Exception:
        return []


def crossmatch_nearest(
    tables: Sequence[Any],
    ra_deg: float,
    dec_deg: float,
    ra_col: str = "RA_ICRS",
    dec_col: str = "DE_ICRS",
    k: int = 5,
) -> list[CatalogMatch]:
    """Return up to k nearest catalog rows across all provided tables."""
    target = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    matches: list[CatalogMatch] = []
    for tbl in tables:
        try:
            src = SkyCoord(
                ra=tbl[ra_col].data * u.deg, dec=tbl[dec_col].data * u.deg, frame="icrs"
            )
            sep = src.separation(target).arcsec
            order = np.argsort(sep)
            take = order[: min(k, len(order))]
            for idx in take:
                row = tbl[int(idx)]
                matches.append(
                    CatalogMatch(
                        catalog=getattr(tbl, "_meta", {}).get("catalog", "unknown"),
                        ra_deg=float(row[ra_col]),
                        dec_deg=float(row[dec_col]),
                        separation_arcsec=float(sep[int(idx)]),
                        row={
                            k: row[k]
                            for k in tbl.colnames
                            if k in ("Source", "RA_ICRS", "DE_ICRS", "Gmag")
                        },
                    )
                )
        except Exception:
            continue
    matches.sort(key=lambda m: m.separation_arcsec)
    return matches[:k]
