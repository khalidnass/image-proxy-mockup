"""A tiny pure-Python PNG writer: no image library needed."""
from __future__ import annotations

import hashlib
import struct
import zlib


def _chunk(kind: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + kind + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))


def write_png_rows(width: int, height: int, row) -> bytes:
    """row(y) -> the bytes of one RGB row (3 * width). Returns a valid 8-bit RGB PNG.

    Whole rows, not pixels: a 1024x1024 picture is a thousand slices, not a
    million calls, so the mock stays cheap at the largest size it serves."""
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter type: none
        raw += row(y)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", header)
            + _chunk(b"IDAT", zlib.compress(bytes(raw), 6)) + _chunk(b"IEND", b""))


def write_png(width: int, height: int, pixel) -> bytes:
    """pixel(x, y) -> (r, g, b). Slow per pixel; for small pictures."""
    return write_png_rows(width, height,
                          lambda y: bytes(c for x in range(width) for c in pixel(x, y)))


def _colours(seed: bytes) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    h = hashlib.sha256(seed).digest()
    return (h[0], h[1], h[2]), (h[3], h[4], h[5])


def gradient(seed: bytes, width: int = 256, height: int | None = None) -> bytes:
    """A diagonal gradient between two colours taken from the seed's hash."""
    height = width if height is None else height
    a, b = _colours(seed)
    span = max(width + height - 2, 1)
    # one colour per diagonal (x + y); each row is a slice of this line
    line = bytes(int(a[i] + (b[i] - a[i]) * (d / span))
                 for d in range(width + height - 1) for i in range(3))
    return write_png_rows(width, height, lambda y: line[3 * y: 3 * (y + width)])


def stripes(seed: bytes, width: int = 256, height: int | None = None) -> bytes:
    """Stripes in two colours from the seed's hash: the 'edited' look."""
    height = width if height is None else height
    a, b = _colours(seed)
    band = 8 + seed[0] % 24 if seed else 16
    line = b"".join(bytes(a if (d // band) % 2 == 0 else b) for d in range(width + height - 1))
    return write_png_rows(width, height, lambda y: line[3 * y: 3 * (y + width)])
