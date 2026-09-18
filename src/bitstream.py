"""Version 1 single-image Omni-RAD container (network byte order).

Header: magic/version, original H/W, hyperlatent H/W, float16 lambda;
then two length-prefixed entropy streams (y, z). T is recomputed on decode.
"""
import math
import struct

_HEADER = struct.Struct('>4s4Ie')
_LENGTH = struct.Struct('>I')
_MAGIC = b'ORD1'


def _read_exact(stream, size):
    data = stream.read(size)
    if len(data) != size:
        raise ValueError('Truncated Omni-RAD bitstream')
    return data


def write_bitstream(path, output, height, width):
    shape = output['shape']
    lmbda = float(output['lmbda_val'][0])
    if min(height, width, *shape) <= 0 or not math.isfinite(lmbda) or lmbda <= 0:
        raise ValueError('Invalid bitstream dimensions or lambda')
    if len(output['strings']) != 2 or any(len(s) != 1 for s in output['strings']):
        raise ValueError('Expected one image and two entropy streams')
    with open(path, 'wb') as stream:
        stream.write(_HEADER.pack(_MAGIC, height, width, *shape, lmbda))
        for group in output['strings']:
            stream.write(_LENGTH.pack(len(group[0])))
            stream.write(group[0])


def read_bitstream(path):
    with open(path, 'rb') as stream:
        magic, height, width, sh, sw, lmbda = _HEADER.unpack(_read_exact(stream, _HEADER.size))
        if magic != _MAGIC:
            raise ValueError('Unsupported Omni-RAD bitstream version')
        if min(height, width, sh, sw) <= 0 or not math.isfinite(lmbda) or lmbda <= 0:
            raise ValueError('Invalid bitstream dimensions or lambda')
        strings = []
        for _ in range(2):
            size, = _LENGTH.unpack(_read_exact(stream, _LENGTH.size))
            strings.append([_read_exact(stream, size)])
        if stream.read(1):
            raise ValueError('Unexpected trailing bitstream data')
    return {'strings': strings, 'lmbda_val': [lmbda]}, (sh, sw), (height, width)
