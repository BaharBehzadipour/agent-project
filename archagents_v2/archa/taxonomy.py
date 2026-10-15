"""Cleaning + normalisation of the ArcMeta metadata.

Everything here is pure python (no torch) so it can run anywhere.
The labelling LLM sometimes echoed the prompt template instead of a value
(e.g. style == "Architectural style"); `is_placeholder` catches those so they
never become training labels.
"""
import re
from typing import Optional

STYLE_CATEGORIES = [
    "Modern", "Gothic", "Baroque", "Renaissance", "Classical/Ancient", "Islamic",
    "Colonial", "Neoclassical", "Art Deco", "Art Nouveau", "Constructivist",
    "Other/Traditional", "Miscellaneous",
]
_STYLE_CAT_FIX = {
    "renaissance_art_periods": "Renaissance",
    "modern_contemporary_art_movements": "Modern",
}

CENTURY_BINS = ["ancient", "medieval", "c15-17", "c18", "c19", "c20", "c21"]

MATERIALS = ["glass", "brick", "wood", "stone", "concrete", "metal", "plaster",
             "marble", "tile", "earth", "thatch_bamboo", "gold"]
_MATERIAL_KEYWORDS = {
    "glass": ["glass"],
    "brick": ["brick"],
    "wood": ["wood", "timber", "half-timber", "log"],
    "stone": ["stone", "limestone", "sandstone", "granite", "travertine", "basalt", "slate"],
    "concrete": ["concrete", "cement"],
    "metal": ["steel", "metal", "iron", "aluminum", "aluminium", "titanium", "copper",
              "bronze", "lead", "zinc", "tin"],
    "plaster": ["plaster", "stucco", "render", "paint", "mortar", "whitewash"],
    "marble": ["marble"],
    "tile": ["tile", "shingle", "terracotta", "terra cotta", "ceramic"],
    "earth": ["adobe", "clay", "mud", "rammed earth", "earth", "laterite"],
    "thatch_bamboo": ["thatch", "straw", "bamboo", "reed", "palm"],
    "gold": ["gold"],
}

COLORS = ["white", "beige", "gray", "red", "brown", "yellow", "blue", "green",
          "pink", "black", "gold", "multicolor"]
_COLOR_SYN = {"silver": "gray", "grey": "gray", "cream": "beige", "tan": "beige",
              "orange": "red", "terracotta": "red", "multicolored": "multicolor",
              "earth tones": "brown", "ivory": "white"}

VIEWS = ["frontal", "side", "corner", "aerial", "low_angle", "closeup", "interior",
         "street", "other"]

_PLACEHOLDER_PATTERNS = [
    r"\(e\.g", r"(function|role|purpose|features|structure) of the structure|^dimensions of the structure", r"^architectural style$", r"^primary materials used",
    r"location \(country", r"\(country and province", r"title of the structure",
    r"^image view angle", r"^surrounding environment", r"^construction date",
    r"^main color$", r"commissioner or patron", r"architect or builder",
    r"purpose and original", r"conceptual identity", r"current role or function",
    r"visual/formal series", r"original title", r"is the architectural image",
    r"^unknown", r"^not specified", r"^n/?a$", r"^none$", r"^nan$", r"^null$",
    r"^true or false$",
]
_PH = re.compile("|".join(_PLACEHOLDER_PATTERNS), re.I)


def is_placeholder(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float):  # NaN
        return True
    s = str(v).strip()
    return s == "" or bool(_PH.search(s))


def clean_str(v) -> Optional[str]:
    return None if is_placeholder(v) else str(v).strip()


def norm_style_category(v) -> Optional[str]:
    s = clean_str(v)
    if s is None:
        return None
    s = _STYLE_CAT_FIX.get(s.lower(), s)
    return s if s in STYLE_CATEGORIES else None


def century_to_bin(c) -> Optional[str]:
    try:
        c = int(float(c))
    except (TypeError, ValueError):
        return None
    if c <= 5:   # BCE (negative) and early CE
        return "ancient"
    if c <= 14:
        return "medieval"
    if c <= 17:
        return "c15-17"
    return {18: "c18", 19: "c19", 20: "c20", 21: "c21"}.get(c)


def century_text(c) -> Optional[str]:
    try:
        c = int(float(c))
    except (TypeError, ValueError):
        return None
    if c <= 0:
        return "ancient"
    if 11 <= c % 100 <= 13:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(c % 10, "th")
    return f"{c}{suf} century"


def norm_materials(v) -> list:
    s = clean_str(v)
    if s is None:
        return []
    s = s.lower()
    out = []
    for canon, kws in _MATERIAL_KEYWORDS.items():
        if any(k in s for k in kws):
            out.append(canon)
    return out


def norm_color(v) -> Optional[str]:
    s = clean_str(v)
    if s is None:
        return None
    s = s.lower()
    if " and " in s and len(s.split(" and ")) >= 2 and "multicolor" not in s:
        s = s.split(" and ")[0]
    for syn, canon in _COLOR_SYN.items():
        if syn in s:
            return canon
    for c in COLORS:
        if c in s:
            return c
    return None


def norm_view(v) -> Optional[str]:
    s = clean_str(v)
    if s is None:
        return None
    s = s.lower()
    if "interior" in s:
        return "interior"
    if "aerial" in s or "bird" in s:
        return "aerial"
    if "close" in s or "detail" in s:
        return "closeup"
    if "low angle" in s or "from below" in s:
        return "low_angle"
    if "corner" in s:
        return "corner"
    if "side" in s:
        return "side"
    if "street" in s:
        return "street"
    if "front" in s:
        return "frontal"
    return "other"


def norm_continent(v) -> Optional[str]:
    s = clean_str(v)
    if s and "/" in s:      # "Asia/Europe" == "Europe/Asia"
        s = "/".join(sorted(p.strip() for p in s.split("/")))
    return s


def clean_record(cap: dict) -> dict:
    """Raw caption dict -> normalised field dict (None / [] when unusable)."""
    cent = cap.get("normalized_century")
    if cent in ("", None):
        cent = None
    style_raw = clean_str(cap.get("style"))
    ph_count = sum(
        is_placeholder(cap.get(k))
        for k in ("style", "primary_materials", "location", "view", "main_color")
    )
    return {
        "style_category": norm_style_category(cap.get("style_category")),
        "style": style_raw.lower() if style_raw else None,
        "century": None if cent is None else int(float(cent)),
        "century_bin": century_to_bin(cent),
        "continent": norm_continent(cap.get("continent")),
        "region": clean_str(cap.get("region")),
        "country": clean_str(cap.get("country_normalized")),
        "materials": norm_materials(cap.get("primary_materials")),
        "color": norm_color(cap.get("main_color")),
        "view": norm_view(cap.get("view")),
        "purpose": clean_str(cap.get("purpose")),
        "environment": (clean_str(cap.get("environment")) or "")[:80].lower() or None,
        "known": str(cap.get("known")).lower() == "true",
        "placeholder_row": ph_count >= 3,
    }


# ---------------------------------------------------------------------------
# captions
# ---------------------------------------------------------------------------
FIELD_ORDER = ["style", "century", "location", "materials", "color", "view"]


def caption_fields(rec: dict) -> dict:
    """Human-readable text per caption field ('' if missing)."""
    style = rec.get("style") or rec.get("style_category") or ""
    if style.lower() in ("miscellaneous", "other/traditional"):
        style = rec.get("style") or ""
    loc = rec.get("country") or rec.get("region") or rec.get("continent") or ""
    if loc.lower() in ("unknown", "none"):
        loc = ""
    mats = ", ".join(m.replace("_", "/") for m in rec.get("materials", [])[:3])
    view = (rec.get("view") or "").replace("_", " ")
    style = re.sub(r"\s*architecture$", "", style, flags=re.I).strip()
    return {
        "style": f"{style} architecture" if style else "",
        "century": century_text(rec.get("century")) or "",
        "location": f"in {loc.title()}" if loc else "",
        "materials": f"made of {mats}" if mats else "",
        "color": f"{rec['color']} tones" if rec.get("color") else "",
        "view": f"{view} view" if view and view != "other" else "",
        "environment": (rec.get("environment") or ""),
    }


def build_caption(rec: dict, drop_prob: float = 0.0, empty_prob: float = 0.0,
                  rng=None, only: Optional[list] = None) -> str:
    """Compose the training / inference prompt.

    drop_prob  : each field is independently dropped (attribute dropout, makes
                 every field controllable on its own).
    empty_prob : the whole caption is replaced with "" (classifier-free guidance).
    only       : keep only these fields (used for the per-task LoRAs).
    """
    import random
    rng = rng or random
    if empty_prob and rng.random() < empty_prob:
        return ""
    f = caption_fields(rec)
    parts = []
    order = FIELD_ORDER if only is None else [k for k in FIELD_ORDER + ["environment"] if k in only]
    for k in order:
        if not f[k]:
            continue
        if drop_prob and rng.random() < drop_prob:
            continue
        parts.append(f[k])
    if not parts:
        return "an architectural photograph"
    return "an architectural photograph, " + ", ".join(parts)
