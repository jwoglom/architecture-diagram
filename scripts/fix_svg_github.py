#!/usr/bin/env python3
"""Normalize Mermaid-exported SVGs for GitHub preview rendering.

GitHub's SVG preview can crop files whose root <svg> uses width/height="100%"
without a reliable intrinsic size. This script rewrites the root <svg> tag to
include explicit numeric width/height and a viewBox.

It supports two common Mermaid export styles:
1) Root <svg> already has a viewBox: reuse that width/height.
2) Root <svg> has no viewBox but wraps content in svg-pan-zoom viewport <g>:
   infer bounds from transformed geometry and synthesize a viewBox.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import xml.etree.ElementTree as ET
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SVG_TAG_RE = re.compile(r"<svg\b[^>]*>", re.IGNORECASE | re.DOTALL)
ATTR_RE = re.compile(r'([^\s=/>]+)\s*=\s*"([^"]*)"')
VIEWBOX_RE = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*$"
)
TRANSFORM_FUNC_RE = re.compile(r"([a-zA-Z]+)\s*\(([^)]*)\)")
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


Affine = tuple[float, float, float, float, float, float]  # a,b,c,d,e,f


@dataclass
class Bounds:
    min_x: float = math.inf
    min_y: float = math.inf
    max_x: float = -math.inf
    max_y: float = -math.inf

    def include_point(self, x: float, y: float) -> None:
        self.min_x = min(self.min_x, x)
        self.min_y = min(self.min_y, y)
        self.max_x = max(self.max_x, x)
        self.max_y = max(self.max_y, y)

    def include_points(self, points: Iterable[tuple[float, float]]) -> None:
        for x, y in points:
            self.include_point(x, y)

    @property
    def empty(self) -> bool:
        return (
            math.isinf(self.min_x)
            or math.isinf(self.min_y)
            or math.isinf(self.max_x)
            or math.isinf(self.max_y)
        )

    def padded(self, pad: float) -> "Bounds":
        if self.empty:
            return self
        return Bounds(
            self.min_x - pad,
            self.min_y - pad,
            self.max_x + pad,
            self.max_y + pad,
        )


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_float(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    value = value.strip()
    if not value:
        return default
    return float(value)


def fmt_num(value: float, *, force_int: bool = False) -> str:
    if force_int or abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    s = f"{value:.3f}".rstrip("0").rstrip(".")
    return s if s else "0"


def affine_identity() -> Affine:
    return (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def affine_compose(m: Affine, n: Affine) -> Affine:
    # Return m * n (apply n, then m).
    a, b, c, d, e, f = m
    A, B, C, D, E, F = n
    return (
        a * A + c * B,
        b * A + d * B,
        a * C + c * D,
        b * C + d * D,
        a * E + c * F + e,
        b * E + d * F + f,
    )


def affine_apply(m: Affine, x: float, y: float) -> tuple[float, float]:
    a, b, c, d, e, f = m
    return (a * x + c * y + e, b * x + d * y + f)


def parse_transform(transform: str | None) -> Affine:
    if not transform:
        return affine_identity()
    out = affine_identity()
    for name, raw_args in TRANSFORM_FUNC_RE.findall(transform):
        args = [float(x) for x in NUM_RE.findall(raw_args)]
        lname = name.lower()
        if lname == "matrix" and len(args) == 6:
            t = (args[0], args[1], args[2], args[3], args[4], args[5])
        elif lname == "translate" and len(args) >= 1:
            tx = args[0]
            ty = args[1] if len(args) >= 2 else 0.0
            t = (1.0, 0.0, 0.0, 1.0, tx, ty)
        elif lname == "scale" and len(args) >= 1:
            sx = args[0]
            sy = args[1] if len(args) >= 2 else sx
            t = (sx, 0.0, 0.0, sy, 0.0, 0.0)
        else:
            # Unsupported transforms (e.g. rotate/skew) are uncommon in Mermaid
            # exports. Ignore rather than fail hard.
            continue
        out = affine_compose(out, t)
    return out


def rect_points(x: float, y: float, w: float, h: float) -> list[tuple[float, float]]:
    return [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]


def parse_points_attr(points: str | None) -> list[tuple[float, float]]:
    if not points:
        return []
    nums = [float(x) for x in NUM_RE.findall(points)]
    if len(nums) < 2:
        return []
    if len(nums) % 2 == 1:
        nums = nums[:-1]
    return list(zip(nums[0::2], nums[1::2]))


def geometry_points(elem: ET.Element) -> list[tuple[float, float]]:
    tag = local_name(elem.tag)
    if tag in {"rect", "foreignObject", "image"}:
        x = parse_float(elem.get("x"), 0.0)
        y = parse_float(elem.get("y"), 0.0)
        w = parse_float(elem.get("width"), 0.0)
        h = parse_float(elem.get("height"), 0.0)
        if w < 0 or h < 0:
            return []
        return rect_points(x, y, w, h)
    if tag == "circle":
        cx = parse_float(elem.get("cx"), 0.0)
        cy = parse_float(elem.get("cy"), 0.0)
        r = parse_float(elem.get("r"), 0.0)
        return rect_points(cx - r, cy - r, 2 * r, 2 * r)
    if tag == "ellipse":
        cx = parse_float(elem.get("cx"), 0.0)
        cy = parse_float(elem.get("cy"), 0.0)
        rx = parse_float(elem.get("rx"), 0.0)
        ry = parse_float(elem.get("ry"), 0.0)
        return rect_points(cx - rx, cy - ry, 2 * rx, 2 * ry)
    if tag == "line":
        x1 = parse_float(elem.get("x1"), 0.0)
        y1 = parse_float(elem.get("y1"), 0.0)
        x2 = parse_float(elem.get("x2"), 0.0)
        y2 = parse_float(elem.get("y2"), 0.0)
        return [(x1, y1), (x2, y2)]
    if tag in {"polygon", "polyline"}:
        return parse_points_attr(elem.get("points"))
    return []


SKIP_SUBTREES = {
    "style",
    "script",
    "defs",
    "marker",
    "clipPath",
    "mask",
    "metadata",
    "title",
    "desc",
}


def compute_bounds_from_xml(text: str, padding: float) -> Bounds | None:
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        print(f"warning: XML parse failed ({exc}); cannot infer bounds", file=sys.stderr)
        return None

    bounds = Bounds()

    def walk(elem: ET.Element, ctm: Affine) -> None:
        tag = local_name(elem.tag)
        if tag in SKIP_SUBTREES:
            return

        local_ctm = affine_compose(ctm, parse_transform(elem.get("transform")))

        pts = geometry_points(elem)
        if pts:
            bounds.include_points(affine_apply(local_ctm, x, y) for x, y in pts)

        for child in list(elem):
            walk(child, local_ctm)

    walk(root, affine_identity())

    if bounds.empty:
        return None

    padded = bounds.padded(padding)
    # Clamp extremely small negatives caused by floating point noise.
    if abs(padded.min_x) < 1e-6:
        padded.min_x = 0.0
    if abs(padded.min_y) < 1e-6:
        padded.min_y = 0.0
    return padded


def parse_root_tag_attrs(svg_tag: str) -> OrderedDict[str, str]:
    attrs: OrderedDict[str, str] = OrderedDict()
    for name, value in ATTR_RE.findall(svg_tag):
        attrs[name] = value
    return attrs


def build_root_tag(attrs: OrderedDict[str, str]) -> str:
    return "<svg " + " ".join(f'{k}="{v}"' for k, v in attrs.items()) + ">"


def normalize_style(style: str | None) -> str | None:
    if style is None:
        return None
    entries: list[tuple[str, str]] = []
    for chunk in style.split(";"):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        key, value = chunk.split(":", 1)
        entries.append((key.strip().lower(), value.strip()))

    style_map: OrderedDict[str, str] = OrderedDict(entries)
    # Keep responsive width but avoid deleting any Mermaid-specific flags.
    if "max-width" not in style_map:
        style_map["max-width"] = "100%"
    return "; ".join(f"{k}: {v}" for k, v in style_map.items()) + ";"


def parse_viewbox(viewbox: str | None) -> tuple[float, float, float, float] | None:
    if not viewbox:
        return None
    m = VIEWBOX_RE.match(viewbox)
    if not m:
        return None
    return tuple(float(x) for x in m.groups())  # type: ignore[return-value]


def is_percent_dimension(value: str | None) -> bool:
    return bool(value and value.strip().endswith("%"))


def patch_svg_file(path: Path, *, padding: float, dry_run: bool) -> bool:
    text = path.read_text(encoding="utf-8")
    m = SVG_TAG_RE.search(text)
    if not m:
        print(f"{path}: no root <svg> tag found", file=sys.stderr)
        return False

    old_tag = m.group(0)
    attrs = parse_root_tag_attrs(old_tag)

    viewbox = parse_viewbox(attrs.get("viewBox"))
    inferred = None
    if viewbox is None:
        inferred = compute_bounds_from_xml(text, padding)
        if inferred is None:
            print(f"{path}: unable to infer bounds (leaving unchanged)", file=sys.stderr)
            return False
        min_x, min_y, max_x, max_y = inferred.min_x, inferred.min_y, inferred.max_x, inferred.max_y
        width = max_x - min_x
        height = max_y - min_y
        attrs["viewBox"] = " ".join(
            (fmt_num(min_x), fmt_num(min_y), fmt_num(width), fmt_num(height))
        )
    else:
        min_x, min_y, width, height = viewbox

    # Always write explicit intrinsic dimensions for GitHub preview.
    attrs["width"] = fmt_num(width, force_int=True)
    attrs["height"] = fmt_num(height, force_int=True)

    if "style" in attrs:
        normalized = normalize_style(attrs["style"])
        if normalized is not None:
            attrs["style"] = normalized

    new_tag = build_root_tag(attrs)
    if new_tag == old_tag:
        print(f"{path}: no changes needed")
        return True

    new_text = text[: m.start()] + new_tag + text[m.end() :]
    if not dry_run:
        path.write_text(new_text, encoding="utf-8")

    source = "existing viewBox" if viewbox is not None else "inferred bounds"
    print(
        f"{path}: patched root svg ({source}) -> width={attrs['width']} height={attrs['height']} viewBox={attrs.get('viewBox')}"
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="SVG files to patch in place")
    parser.add_argument(
        "--padding",
        type=float,
        default=16.0,
        help="Padding (px) added around inferred bounds when synthesizing a viewBox (default: 16)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze and print changes without writing files",
    )
    args = parser.parse_args()

    ok = True
    for file_arg in args.files:
        path = Path(file_arg)
        if not path.exists():
            print(f"{path}: file not found", file=sys.stderr)
            ok = False
            continue
        if path.suffix.lower() != ".svg":
            print(f"{path}: not an .svg file", file=sys.stderr)
            ok = False
            continue
        ok = patch_svg_file(path, padding=args.padding, dry_run=args.dry_run) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
