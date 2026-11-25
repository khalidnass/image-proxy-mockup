"""A tiny pure-Python PNG writer: no image library needed."""
from __future__ import annotations

import hashlib
import struct
import zlib


def _chunk(kind: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + kind + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))


def write_png(width: int, height: int, pixel) -> bytes:
    """pixel(x, y) -> (r, g, b). Returns the bytes of a valid 8-bit RGB PNG."""
    rows = bytearray()
    for y in range(height):
        rows.append(0)  # filter type: none
        for x in range(width):
            rows.extend(pixel(x, y))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", header)
            + _chunk(b"IDAT", zlib.compress(bytes(rows), 6)) + _chunk(b"IEND", b""))


def _colours(seed: bytes) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    h = hashlib.sha256(seed).digest()
    return (h[0], h[1], h[2]), (h[3], h[4], h[5])


def gradient(seed: bytes, size: int = 256) -> bytes:
    """A diagonal gradient between two colours taken from the seed's hash."""
    a, b = _colours(seed)

    def px(x, y):
        t = (x + y) / (2 * (size - 1))
        return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))

    return write_png(size, size, px)


def stripes(seed: bytes, size: int = 256) -> bytes:
    """Stripes in two colours from the seed's hash: the 'edited' look."""
    a, b = _colours(seed)
    width = 8 + seed[0] % 24 if seed else 16

    def px(x, y):
        return a if ((x + y) // width) % 2 == 0 else b

    return write_png(size, size, px)
