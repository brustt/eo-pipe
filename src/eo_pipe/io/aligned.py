"""Naming and tag contract of Aligned rasters.

Writers call :func:`tags_for` and :func:`polarization_of`; readers call
:func:`identify`.  Both directions share the precedence rules below, so a raster
written by one step is always read back the same way by another.

An Aligned raster is identified by GeoTIFF tags (primary) and by parseable
tokens in its filename (fallback)::

    align_<TILE>_<datetime>_<stem>.tif
    align_31TCJ_20250314T051233_S1A_..._vv.tif

``<TILE>`` is a five-character MGRS tile name; ``<datetime>`` is 15 characters
(``YYYYMMDDTHHMMSS``) or 8 (``YYYYMMDD``) when the source carries no time.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

NAME_PREFIX = "align"

TAG_TILE = "TILE"
TAG_DATETIME = "DATETIME"
TAG_SOURCE = "SOURCE"
TAG_POLARIZATION = "POLARIZATION"
TAG_PARAMS = "ALIGN_PARAMS"

POLARIZATIONS = ("VV", "VH", "HH", "HV")

DEFAULT_DATE_PATTERN = r"(?<!\d)(\d{8}(?:T\d{6})?)(?!\d)"

_TOKEN_RE = re.compile(r"^\d{8}(?:T\d{6})?$")
_NAME_RE = re.compile(
    rf"^{NAME_PREFIX}_(?P<tile>\d{{2}}[C-HJ-NP-X][A-HJ-NP-Z]{{2}})"
    r"_(?P<dt>\d{8}(?:T\d{6})?)_(?P<stem>.+)\.tif$"
)
_POL_SUFFIX_RE = re.compile(r"(?i)(?:^|[_\-.])(vv|vh|hh|hv)(?=$|[_\-.])")


@dataclass(frozen=True)
class AlignedName:
    tile: str
    token: str
    stem: str


def datetime_token(stem: str, pattern: str = DEFAULT_DATE_PATTERN) -> str:
    """Extract the compact datetime token from *stem*.

    Groups of a custom *pattern* are concatenated, so ``(\\d{4})-(\\d{2})-(\\d{2})``
    yields ``YYYYMMDD``.

    Raises:
        ValueError: If no date is found or the result is not ``YYYYMMDD[THHMMSS]``.
    """
    match = re.search(pattern, stem)
    if match is None:
        raise ValueError(f"No acquisition date found in '{stem}' (pattern {pattern!r}).")
    token = "".join(g for g in match.groups() if g) if match.groups() else match.group(0)
    if not _TOKEN_RE.match(token):
        raise ValueError(
            f"Date {token!r} found in '{stem}' is not YYYYMMDD or YYYYMMDDTHHMMSS."
        )
    return token


def token_to_iso(token: str) -> str:
    """``20250314T051233`` -> ``2025-03-14T05:12:33``; ``20250314`` -> ``2025-03-14``."""
    date = f"{token[:4]}-{token[4:6]}-{token[6:8]}"
    return date if len(token) == 8 else f"{date}T{token[9:11]}:{token[11:13]}:{token[13:15]}"


def iso_to_datetime64(iso: str) -> np.datetime64:
    return np.datetime64(iso, "ns")


def aligned_name(tile: str, token: str, stem: str) -> str:
    return f"{NAME_PREFIX}_{tile}_{token}_{stem}.tif"


def parse_aligned_name(filename: str) -> Optional[AlignedName]:
    """Parse an Aligned raster filename; ``None`` when it does not follow the contract."""
    match = _NAME_RE.match(filename)
    if match is None:
        return None
    return AlignedName(match["tile"], match["dt"], match["stem"])


def polarization_from_stem(stem: str) -> Optional[str]:
    """Polarization from a ``_vv`` / ``_vh`` / ``_hh`` / ``_hv`` suffix token, else ``None``."""
    hits = _POL_SUFFIX_RE.findall(stem)
    return hits[-1].upper() if hits else None


def polarization_of(
    count: int,
    descriptions: Sequence[Optional[str]],
    stem: str,
    override: Optional[str] = None,
) -> Optional[str]:
    """Polarization to tag a freshly aligned raster with.

    Precedence: *override*, then (single-band only) the band description, then a
    ``_vv`` / ``_vh`` / ``_hh`` / ``_hv`` token of *stem*.  Multi-band rasters carry none.
    """
    if override:
        return override.upper()
    if count != 1:
        return None
    desc = ((descriptions[0] if descriptions else None) or "").upper()
    return desc if desc in POLARIZATIONS else polarization_from_stem(stem)


def tags_for(
    source_tags: Mapping[str, str],
    *,
    tile: str,
    token: str,
    source: str,
    params: Mapping[str, Any],
    polarization: Optional[str] = None,
) -> Dict[str, str]:
    """GeoTIFF tags of an Aligned raster: the source's own tags plus the identity tags."""
    tags = dict(source_tags)
    tags.update({
        TAG_TILE: tile,
        TAG_DATETIME: token_to_iso(token),
        TAG_SOURCE: source,
        TAG_PARAMS: json.dumps(dict(params)),
    })
    if polarization:
        tags[TAG_POLARIZATION] = polarization
    return tags


@dataclass(frozen=True)
class AlignedIdentity:
    """What an Aligned raster says about itself: Tile, acquisition time, data variables."""

    tile: str
    iso: str
    variables: List[str]
    tags: Dict[str, str]


def identify(
    filename: str,
    tags: Mapping[str, str],
    count: int,
    descriptions: Sequence[Optional[str]],
) -> AlignedIdentity:
    """Read the identity of an Aligned raster; tags win, the filename is the fallback.

    Variables are ``[POL]`` for a single polarized band, else the band descriptions
    (``band_N`` when a band has none).  Only an untagged single-band file falls back to
    the polarization in its filename.

    Raises:
        ValueError: If neither tags nor filename give the Tile and datetime.
    """
    parsed = parse_aligned_name(filename)
    tile = tags.get(TAG_TILE) or (parsed.tile if parsed else None)
    if tags.get(TAG_DATETIME):
        iso: Optional[str] = tags[TAG_DATETIME]
    else:
        iso = token_to_iso(parsed.token) if parsed else None
    if not tile or not iso:
        raise ValueError(
            f"{filename}: no TILE/DATETIME tags and the filename does not follow "
            f"'{NAME_PREFIX}_<TILE>_<datetime>_<stem>.tif'."
        )
    pol = tags.get(TAG_POLARIZATION)
    if pol is None and not tags.get(TAG_TILE) and count == 1 and parsed:
        pol = polarization_from_stem(parsed.stem)
    if count == 1 and pol:
        variables = [pol.upper()]
    else:
        variables = [d or f"band_{i}" for i, d in enumerate(descriptions, start=1)]
    return AlignedIdentity(tile=tile, iso=iso, variables=variables, tags=dict(tags))
