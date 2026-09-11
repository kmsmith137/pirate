"""Randomized unit tests for pirate_frb.chimefrb.AssembledChunk.

Dispatched from ``python -m pirate_frb test --cfrb``.

This file also holds a pure-python writer and reader for the msgpack format. The writer
exists only to generate test inputs -- nothing outside this file should write these files
-- and the reader is the oracle the C++ parser and decode kernels are checked against.

WHAT THIS FILE CANNOT ESTABLISH: the python reader here is a second implementation written
from the same reading of ch_frb_io as the C++ one, so a shared misreading of the format
would pass every test below. That is what the spot checks in ``misc/chimefrb/`` are for --
they link the real ch_frb_io reader, writer and decode. See notes/chimefrb.md.
"""

import os
import struct

import numpy as np

from . import AssembledChunk
from ..utils import atomic_print
from .testutils import default_rng as _default_rng


# The format's fixed header string, and the only version we generate or accept.
FORMAT_MAGIC = b'assembled_chunk in msgpack format'
FORMAT_VERSION = 2

# Where test files go. tmpfs, so a full-size chunk costs a few ms; falls back to $TMPDIR.
_SCRATCH_DIR = '/dev/shm' if os.path.isdir('/dev/shm') else None


####################################################################################################
#
# Pure-python msgpack writer.
#
# msgpack scalars are BIG-endian; the 'bin' bodies are raw little-endian x86 memory. The
# width of an integer or bin header is a free choice for the writer, so pack_uint() and
# pack_bin() take a 'width' argument: randomizing it is how we check that the C++ reader
# handles every legal encoding rather than just the one ch_frb_io happens to emit.


def pack_uint(value, width=None):
    """Encode a nonnegative int. 'width' in {1,2,4,8} forces an encoding; None picks the smallest."""

    assert value >= 0
    if width is None:
        width = 1 if value < 0x80 else (1 if value <= 0xff else (2 if value <= 0xffff else (4 if value <= 0xffffffff else 8)))
        if value < 0x80:
            return struct.pack('>B', value)          # positive fixint
    if width == 1:
        assert value <= 0xff
        return b'\xcc' + struct.pack('>B', value)
    if width == 2:
        assert value <= 0xffff
        return b'\xcd' + struct.pack('>H', value)
    if width == 4:
        assert value <= 0xffffffff
        return b'\xce' + struct.pack('>I', value)
    assert width == 8
    return b'\xcf' + struct.pack('>Q', value)


def pack_bin(payload, width=None):
    """Encode a bytes payload as a msgpack 'bin'. 'width' in {1,2,4} forces the header width."""

    n = len(payload)
    if width is None:
        width = 1 if n <= 0xff else (2 if n <= 0xffff else 4)
    if width == 1:
        assert n <= 0xff
        return b'\xc4' + struct.pack('>B', n) + payload
    if width == 2:
        assert n <= 0xffff
        return b'\xc5' + struct.pack('>H', n) + payload
    assert width == 4
    return b'\xc6' + struct.pack('>I', n) + payload


def pack_str(s):
    assert len(s) <= 0xff
    return b'\xd9' + struct.pack('>B', len(s)) + s


def pack_array_header(nitems):
    return b'\xdc' + struct.pack('>H', nitems)


class Chunk:
    """One chunk's contents, in the form the writer and reader both use.

    Attribute names match ``AssembledChunk``, so a test can compare the two field by field.
    """

    def __init__(self, beam_id, nupfreq, nt_per_packet, fpga_counts_per_sample, binning,
                 nfreq_coarse, nt_coarse, fpga_begin, frame0_nano, nrfifreq, has_rfi_mask,
                 scales, offsets, data, rfi_mask):
        self.version = FORMAT_VERSION
        self.compression = 0
        self.beam_id = beam_id
        self.nupfreq = nupfreq
        self.nt_per_packet = nt_per_packet
        self.fpga_counts_per_sample = fpga_counts_per_sample
        self.binning = binning
        self.nfreq_coarse = nfreq_coarse
        self.nt_coarse = nt_coarse
        self.nt_per_chunk = nt_coarse * nt_per_packet
        self.nscales = nfreq_coarse * nt_coarse
        self.ndata = nfreq_coarse * nupfreq * self.nt_per_chunk
        self.fpga_begin = fpga_begin
        self.fpga_end = fpga_begin + self.nt_per_chunk * fpga_counts_per_sample * binning
        self.frame0_nano = frame0_nano
        self.nrfifreq = nrfifreq
        self.has_rfi_mask = has_rfi_mask
        self.scales = scales        # float32 (nfreq_coarse, nt_coarse)
        self.offsets = offsets      # float32 (nfreq_coarse, nt_coarse)
        self.data = data            # uint8   (nfreq, nt)
        self.rfi_mask = rfi_mask    # uint8   (nrfifreq, nt/8), or None

    @property
    def nfreq(self):
        return self.nfreq_coarse * self.nupfreq

    @property
    def nt(self):
        return self.nt_per_chunk

    def to_msgpack(self, rng=None, version=FORMAT_VERSION, compression=0):
        """Serialize. 'version' and 'compression' are overridable so tests can build files
        the reader is supposed to REJECT."""

        # Randomize the encoding widths where the writer has a free choice, so that the
        # reader is exercised on more than the one encoding ch_frb_io emits.
        def w_int(v, lo=1):
            if rng is None:
                return None
            choices = [x for x in (1, 2, 4, 8) if x >= lo and v <= (1 << (8*x)) - 1]
            return choices[rng.integers(len(choices))] if choices else None

        def w_bin(payload):
            if rng is None:
                return None
            n = len(payload)
            choices = [x for x in (1, 2, 4) if n <= (1 << (8*x)) - 1]
            return choices[rng.integers(len(choices))] if choices else None

        sc = np.ascontiguousarray(self.scales, dtype=np.float32).tobytes()
        of = np.ascontiguousarray(self.offsets, dtype=np.float32).tobytes()
        da = np.ascontiguousarray(self.data, dtype=np.uint8).tobytes()
        rm = (np.ascontiguousarray(self.rfi_mask, dtype=np.uint8).tobytes()
              if (self.has_rfi_mask and self.rfi_mask is not None) else b'')

        nitems = 21 if version == 2 else 17
        out = [pack_array_header(nitems),
               pack_str(FORMAT_MAGIC),
               pack_uint(version, 1),
               pack_uint(compression, 1),
               pack_uint(len(da), w_int(len(da))),
               pack_uint(self.beam_id, w_int(self.beam_id)),
               pack_uint(self.nupfreq, w_int(self.nupfreq)),
               pack_uint(self.nt_per_packet, w_int(self.nt_per_packet)),
               pack_uint(self.fpga_counts_per_sample, w_int(self.fpga_counts_per_sample)),
               pack_uint(self.nt_coarse, w_int(self.nt_coarse)),
               pack_uint(self.nscales, w_int(self.nscales)),
               pack_uint(self.ndata, w_int(self.ndata)),
               pack_uint(self.fpga_begin, w_int(self.fpga_begin)),
               pack_uint(self.fpga_end - self.fpga_begin, w_int(self.fpga_end - self.fpga_begin)),
               pack_uint(self.binning, w_int(self.binning)),
               pack_bin(sc, w_bin(sc)),
               pack_bin(of, w_bin(of)),
               pack_bin(da, w_bin(da))]

        if version == 2:
            out += [pack_uint(self.frame0_nano, w_int(self.frame0_nano)),
                    pack_uint(self.nrfifreq, w_int(self.nrfifreq)),
                    b'\xc3' if self.has_rfi_mask else b'\xc2',
                    pack_bin(rm, w_bin(rm))]

        return b''.join(out)


####################################################################################################
#
# Pure-python msgpack reader (the oracle), and the reference decode.


class _Cursor:
    def __init__(self, buf):
        self.buf = buf
        self.p = 0

    def u8(self):
        v = self.buf[self.p]
        self.p += 1
        return v

    def get_uint(self):
        tag = self.u8()
        if tag < 0x80:
            return tag
        if tag == 0xcc:
            v = self.buf[self.p]; self.p += 1; return v
        if tag == 0xcd:
            v = struct.unpack_from('>H', self.buf, self.p)[0]; self.p += 2; return v
        if tag == 0xce:
            v = struct.unpack_from('>I', self.buf, self.p)[0]; self.p += 4; return v
        if tag == 0xcf:
            v = struct.unpack_from('>Q', self.buf, self.p)[0]; self.p += 8; return v
        raise RuntimeError(f'unexpected msgpack integer tag 0x{tag:02x}')

    def get_bool(self):
        tag = self.u8()
        if tag == 0xc2:
            return False
        if tag == 0xc3:
            return True
        raise RuntimeError(f'unexpected msgpack bool tag 0x{tag:02x}')

    def get_str(self):
        tag = self.u8()
        n = (tag & 0x1f) if (tag & 0xe0) == 0xa0 else self.u8()
        s = bytes(self.buf[self.p : self.p+n]); self.p += n
        return s

    def get_bin(self):
        tag = self.u8()
        if tag == 0xc4:
            n = self.buf[self.p]; self.p += 1
        elif tag == 0xc5:
            n = struct.unpack_from('>H', self.buf, self.p)[0]; self.p += 2
        elif tag == 0xc6:
            n = struct.unpack_from('>I', self.buf, self.p)[0]; self.p += 4
        else:
            raise RuntimeError(f'unexpected msgpack bin tag 0x{tag:02x}')
        b = bytes(self.buf[self.p : self.p+n]); self.p += n
        return b

    def get_array_header(self):
        tag = self.u8()
        if (tag & 0xf0) == 0x90:
            return tag & 0x0f
        if tag == 0xdc:
            n = struct.unpack_from('>H', self.buf, self.p)[0]; self.p += 2; return n
        if tag == 0xdd:
            n = struct.unpack_from('>I', self.buf, self.p)[0]; self.p += 4; return n
        raise RuntimeError(f'unexpected msgpack array tag 0x{tag:02x}')


def read_msgpack(filename):
    """Pure-python reader. Returns a Chunk. This is the unit tests' oracle -- it is
    deliberately a straight-line transcription of the format, with no attempt to be fast."""

    with open(filename, 'rb') as f:
        buf = f.read()

    cur = _Cursor(buf)
    nitems = cur.get_array_header()
    magic = cur.get_str()
    assert magic == FORMAT_MAGIC, f'bad header string {magic!r}'
    version = cur.get_uint()
    assert nitems == (21 if version == 2 else 17), f'{nitems} items for version {version}'

    compression = cur.get_uint()
    data_size = cur.get_uint()
    beam_id = cur.get_uint()
    nupfreq = cur.get_uint()
    nt_per_packet = cur.get_uint()
    fpga_counts_per_sample = cur.get_uint()
    nt_coarse = cur.get_uint()
    nscales = cur.get_uint()
    ndata = cur.get_uint()
    fpga_begin = cur.get_uint()
    cur.get_uint()   # fpga_end - fpga_begin; recomputed by the Chunk constructor
    binning = cur.get_uint()

    nfreq_coarse = nscales // nt_coarse
    nt_per_chunk = nt_coarse * nt_per_packet

    scales = np.frombuffer(cur.get_bin(), dtype=np.float32).reshape(nfreq_coarse, nt_coarse)
    offsets = np.frombuffer(cur.get_bin(), dtype=np.float32).reshape(nfreq_coarse, nt_coarse)
    data = np.frombuffer(cur.get_bin(), dtype=np.uint8).reshape(nfreq_coarse*nupfreq, nt_per_chunk)

    frame0_nano, nrfifreq, has_rfi_mask, rfi_mask = 0, 0, False, None
    if version == 2:
        frame0_nano = cur.get_uint()
        nrfifreq = cur.get_uint()
        has_rfi_mask = cur.get_bool()
        mask_bytes = cur.get_bin()
        if has_rfi_mask:
            rfi_mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(nrfifreq, nt_per_chunk // 8)
        else:
            nrfifreq = 0

    assert cur.p == len(buf), f'{len(buf) - cur.p} trailing bytes'
    assert compression == 0, f'compression={compression}'
    assert data_size == ndata

    return Chunk(beam_id, nupfreq, nt_per_packet, fpga_counts_per_sample, binning,
                 nfreq_coarse, nt_coarse, fpga_begin, frame0_nano, nrfifreq, has_rfi_mask,
                 scales, offsets, data, rfi_mask)


def _broadcast_rfimask(chunk):
    """Unpack the bit-packed mask to a (nfreq, nt) float32 array of 0.0/1.0.

    LSB-first within each byte, and the frequency axis is nrfifreq (coarser than nfreq), so
    each mask row covers nfreq/nrfifreq fine channels.
    """

    bits = np.unpackbits(chunk.rfi_mask, axis=1, bitorder='little').astype(np.float32)
    fdiv = chunk.nfreq // chunk.nrfifreq
    return np.repeat(bits, fdiv, axis=0)


def decode_intensity_reference(chunk, apply_rfimask):
    """Reference decode, in plain float32 (see the tolerance note in test_decode())."""

    d = chunk.data.reshape(chunk.nfreq_coarse, chunk.nupfreq, chunk.nt_coarse, chunk.nt_per_packet)
    sc = chunk.scales[:, None, :, None]
    of = chunk.offsets[:, None, :, None]
    out = (sc.astype(np.float32) * d.astype(np.float32) + of.astype(np.float32))
    out = out.reshape(chunk.nfreq, chunk.nt)
    if apply_rfimask:
        out = out * _broadcast_rfimask(chunk)
    return np.ascontiguousarray(out, dtype=np.float32)


def decode_weights_reference(chunk, apply_rfimask):
    w = ((chunk.data != 0) & (chunk.data != 255)).astype(np.float32)
    if apply_rfimask:
        w = w * _broadcast_rfimask(chunk)
    return np.ascontiguousarray(w, dtype=np.float32)


def intensity_tolerance(chunk):
    """Elementwise-derived, then maximized: the largest disagreement possible between our
    kernel's single fused multiply-add and the reference's separate multiply and add.

    With u = eps_f32/2 the unit roundoff,
        reference = (s*x*(1+d1) + o)*(1+d2),  kernel = (s*x + o)*(1+d3),  |di| <= u
        |difference| <= u*|s*x| + 2*u*|s*x + o| <= eps_f32 * (2*|s*x| + |o|)

    This is a bound, not a fitted number: it must be an ABSOLUTE tolerance, because
    offset = -128*scale + mean makes the intensity scale*(x-128) + mean, so the two terms
    nearly cancel for x near 128 and a relative tolerance on the result is meaningless.
    """

    d = chunk.data.reshape(chunk.nfreq_coarse, chunk.nupfreq, chunk.nt_coarse, chunk.nt_per_packet)
    sx = np.abs(chunk.scales[:, None, :, None].astype(np.float64) * d)
    return float(np.finfo(np.float32).eps * np.max(2.0*sx + np.abs(chunk.offsets[:, None, :, None])))


####################################################################################################
#
# Randomization.


def make_random_chunk(full_size=False, force_ntpp16=False, has_rfi_mask=None):
    """Draw a random Chunk.

    Chunks are small by default (a few KB): the C++ parser derives nfreq_coarse and
    nt_per_chunk from the file rather than hardwiring ch_frb_io's 1024s, so a test does not
    need a 17 MB file to exercise it. 'full_size=True' asks for the real CHIME geometry,
    which is worth sampling occasionally but costs ~50x more per iteration.

    'force_ntpp16' restricts nt_per_packet to 16, which is what the decode kernels support.
    """

    if full_size:
        nfreq_coarse, nupfreq, nt_per_packet, nt_coarse = 1024, 16, 16, 64
    else:
        nfreq_coarse = int(np.random.randint(1, 9))
        nupfreq = int(np.random.randint(1, 17))            # odd and even both matter
        nt_per_packet = 16 if force_ntpp16 else int(2 ** np.random.randint(2, 6))   # 4,8,16,32
        nt_coarse = int(np.random.randint(1, 9))

    nt_per_chunk = nt_coarse * nt_per_packet
    nfreq = nfreq_coarse * nupfreq

    # An RFI mask needs nt_per_chunk divisible by 8, and its frequency axis must divide
    # nfreq (the mask is coarser than the data by an integer factor).
    fdiv_choices = [f for f in range(1, nupfreq+1) if (nupfreq % f) == 0]
    mask_ok = (nt_per_chunk % 8) == 0
    if has_rfi_mask is None:
        has_rfi_mask = mask_ok and (np.random.uniform() < 0.75)
    has_rfi_mask = bool(has_rfi_mask and mask_ok)

    nrfifreq = 0
    rfi_mask = None
    if has_rfi_mask:
        nrfifreq = nfreq // int(fdiv_choices[np.random.randint(len(fdiv_choices))])
        # Runs of set/clear bits, not uniform noise: an off-by-one in the byte or bit index
        # is invisible against noise but obvious against runs.
        nmb = nt_per_chunk // 8
        bits = np.zeros((nrfifreq, nt_per_chunk), dtype=np.uint8)
        for i in range(nrfifreq):
            t = 0
            val = int(np.random.randint(2))
            while t < nt_per_chunk:
                runlen = int(np.random.randint(1, 3*nt_per_packet + 2))
                bits[i, t:t+runlen] = val
                t += runlen
                val ^= 1
        rfi_mask = np.packbits(bits, axis=1, bitorder='little').reshape(nrfifreq, nmb)

    # Scales: zero an order-one fraction of blocks, since real files have ~38% zero (those
    # are blocks where no packet arrived). Offsets take both signs.
    scales = np.random.uniform(0.1, 900.0, size=(nfreq_coarse, nt_coarse)).astype(np.float32)
    p_zero = np.clip(np.random.uniform(-0.1, 0.8), 0.0, 1.0)
    zero_mask = np.random.uniform(size=scales.shape) < p_zero
    scales[zero_mask] = 0.0
    offsets = np.random.uniform(-7300.0, 7300.0, size=(nfreq_coarse, nt_coarse)).astype(np.float32)
    offsets[zero_mask] = 0.0    # matches the encoder: a never-written block is all zeros

    data = np.random.randint(0, 256, size=(nfreq, nt_per_chunk), dtype=np.uint8)
    data = data.reshape(nfreq_coarse, nupfreq, nt_coarse, nt_per_packet)
    data[zero_mask[:, None, :, None].repeat(nupfreq, 1).repeat(nt_per_packet, 3)] = 0
    data = data.reshape(nfreq, nt_per_chunk)

    # Sentinels (0 and 255) drive the weights, so sample them deliberately: 'p' is
    # sometimes 0 and sometimes 1, giving all-masked and no-masked chunks a few percent of
    # the time each rather than never.
    p = np.clip(np.random.uniform(-0.1, 1.1), 0.0, 1.0)
    hit = np.random.uniform(size=data.shape) < (0.2 * p)
    data[hit] = np.where(np.random.uniform(size=int(hit.sum())) < 0.5, 0, 255).astype(np.uint8)

    return Chunk(beam_id=int(np.random.randint(0, 65536)),
                 nupfreq=nupfreq, nt_per_packet=nt_per_packet,
                 fpga_counts_per_sample=int(np.random.randint(1, 3201)),
                 binning=int(2 ** np.random.randint(0, 4)),
                 nfreq_coarse=nfreq_coarse, nt_coarse=nt_coarse,
                 fpga_begin=int(np.random.randint(0, 1 << 40)),
                 frame0_nano=int(np.random.randint(0, 1 << 50)),
                 nrfifreq=nrfifreq, has_rfi_mask=has_rfi_mask,
                 scales=scales, offsets=offsets, data=data, rfi_mask=rfi_mask)


####################################################################################################
#
# Test files. These go in /dev/shm (tmpfs), with a pid-tagged name: other agents run
# 'pirate_frb test' on the same machine, and a fixed name in a shared directory would make
# two runs corrupt each other.


_file_counter = 0


class TempChunkFile:
    """Context manager: writes 'payload' to a uniquely-named scratch file, removes it on exit
    (including on an exception, so a failing test does not leak RAM-backed files)."""

    def __init__(self, payload):
        global _file_counter
        _file_counter += 1
        d = _SCRATCH_DIR
        if d is None:
            import tempfile
            d = tempfile.gettempdir()
        self.filename = os.path.join(d, f'pirate_cfrb_{os.getpid()}_{_file_counter}.msg')
        self.payload = payload

    def __enter__(self):
        with open(self.filename, 'wb') as f:
            f.write(self.payload)
        return self.filename

    def __exit__(self, *exc):
        try:
            os.unlink(self.filename)
        except FileNotFoundError:
            pass
        return False


####################################################################################################
#
# The tests.


def _assert_metadata_equal(got, want, tag):
    for field in ('version', 'compression', 'beam_id', 'binning', 'nupfreq', 'nt_per_packet',
                  'fpga_counts_per_sample', 'nt_coarse', 'nscales', 'ndata', 'nrfifreq',
                  'fpga_begin', 'fpga_end', 'frame0_nano', 'has_rfi_mask',
                  'nfreq_coarse', 'nt_per_chunk', 'nfreq', 'nt'):
        g, w = getattr(got, field), getattr(want, field)
        assert g == w, f'{tag}: {field}: C++ gave {g}, python reference gave {w}'


def test_parse(chunk=None, rng=None):
    """Round-trip: write a chunk, read it with both readers, compare.

    The four raw arrays are compared BIT-EXACT -- they are memcpy paths, so any tolerance
    would be hiding a bug.
    """

    if chunk is None:
        chunk = make_random_chunk()
    if rng is None:
        rng = _default_rng()

    with TempChunkFile(chunk.to_msgpack(rng=rng)) as fn:
        c = AssembledChunk.from_msgpack(fn)
        ref = read_msgpack(fn)

    tag = (f'nfreq_coarse={chunk.nfreq_coarse} nupfreq={chunk.nupfreq} '
           f'nt_per_packet={chunk.nt_per_packet} nt_coarse={chunk.nt_coarse} '
           f'has_rfi_mask={chunk.has_rfi_mask}')

    _assert_metadata_equal(c, ref, tag)
    _assert_metadata_equal(c, chunk, tag)

    assert np.array_equal(np.asarray(c.data), ref.data), f'{tag}: data mismatch'
    assert np.array_equal(np.asarray(c.scales), ref.scales), f'{tag}: scales mismatch'
    assert np.array_equal(np.asarray(c.offsets), ref.offsets), f'{tag}: offsets mismatch'
    if chunk.has_rfi_mask:
        assert np.array_equal(np.asarray(c.rfi_mask), ref.rfi_mask), f'{tag}: rfi_mask mismatch'
    else:
        assert c.rfi_mask is None, f'{tag}: expected rfi_mask to be None'

    # fraction_missing() counts zero-scale blocks, i.e. blocks where no packet arrived.
    want = float(np.mean(chunk.scales == 0.0))
    assert abs(c.fraction_missing() - want) < 1.0e-6, \
        f'{tag}: fraction_missing {c.fraction_missing()} != {want}'

    return c, ref, tag


def test_decode(chunk=None):
    """Decode with both kernels, for both values of apply_rfimask, against the numpy reference."""

    if chunk is None:
        chunk = make_random_chunk(force_ntpp16=True)
    assert chunk.nt_per_packet == 16

    c, ref, tag = test_parse(chunk)

    mask_choices = [False, True] if chunk.has_rfi_mask else [False]
    for apply_rfimask in mask_choices:
        # Weights are a comparison against 0 and 255 -- no arithmetic, so bit-exact.
        w = np.asarray(c.decode_weights(apply_rfimask=apply_rfimask))
        w_ref = decode_weights_reference(ref, apply_rfimask)
        assert np.array_equal(w, w_ref), \
            f'{tag}: weights mismatch (apply_rfimask={apply_rfimask}), ' \
            f'{int(np.sum(w != w_ref))} of {w.size} samples differ'

        # Intensity: our kernel is one FMA, the reference rounds twice. See
        # intensity_tolerance() for the derivation -- the bound is not fitted.
        i = np.asarray(c.decode_intensity(apply_rfimask=apply_rfimask))
        i_ref = decode_intensity_reference(ref, apply_rfimask)
        atol = intensity_tolerance(ref)
        err = float(np.max(np.abs(i - i_ref))) if i.size else 0.0
        assert err <= atol, \
            f'{tag}: intensity max error {err:.6g} exceeds bound {atol:.6g} ' \
            f'(apply_rfimask={apply_rfimask})'

    # Applying the mask must be exactly a multiply by 0.0 or 1.0, so this is bit-exact.
    if chunk.has_rfi_mask:
        i0 = np.asarray(c.decode_intensity(apply_rfimask=False))
        i1 = np.asarray(c.decode_intensity(apply_rfimask=True))
        assert np.array_equal(i1, i0 * _broadcast_rfimask(ref)), \
            f'{tag}: apply_rfimask is not a clean 0/1 multiply'

    # 'out=' must write in place and return the same buffer.
    out = np.zeros((c.nfreq, c.nt), dtype=np.float32)
    got = c.decode_intensity(apply_rfimask=False, out=out)
    assert got is out or np.shares_memory(got, out), f'{tag}: out= was not written in place'
    assert np.array_equal(out, np.asarray(c.decode_intensity(apply_rfimask=False))), \
        f'{tag}: out= gave a different answer'


def test_rejections():
    """Every malformed input the reader is supposed to refuse. A silent misparse here would
    be worse than a crash, so each case asserts that an exception is raised."""

    chunk = make_random_chunk()
    rng = _default_rng()

    def expect_raise(payload, what):
        with TempChunkFile(payload) as fn:
            try:
                AssembledChunk.from_msgpack(fn)
            except RuntimeError:
                return
            raise AssertionError(f'expected from_msgpack() to reject {what}')

    # Version 1 (a 17-item array), and a bogus version number.
    expect_raise(chunk.to_msgpack(rng=rng, version=1), 'a version-1 file')
    expect_raise(chunk.to_msgpack(rng=rng, version=3), 'a version-3 file')

    # Compressed.
    expect_raise(chunk.to_msgpack(rng=rng, compression=1), 'a bitshuffle-compressed file')

    # Truncation, at a random point. Cutting anywhere must fail, never misparse: the
    # interesting cases are inside the header, inside a body, and inside the tail.
    good = chunk.to_msgpack(rng=rng)
    for _ in range(4):
        n = int(np.random.randint(0, len(good)))
        expect_raise(good[:n], f'a file truncated to {n} of {len(good)} bytes')

    # Trailing garbage: the computed layout would no longer account for the whole file.
    expect_raise(good + b'\x00' * 8, 'a file with trailing bytes')

    # A corrupt header string.
    expect_raise(b'\xdc\x00\x15' + pack_str(b'not an assembled_chunk') + good[38:],
                 'a file with the wrong header string')

    # nt_per_packet != 16 parses fine but must not decode.
    c8 = make_random_chunk(force_ntpp16=False)
    while c8.nt_per_packet == 16:
        c8 = make_random_chunk(force_ntpp16=False)
    with TempChunkFile(c8.to_msgpack(rng=rng)) as fn:
        c = AssembledChunk.from_msgpack(fn)     # parses
        try:
            c.decode_intensity(apply_rfimask=False)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f'expected decode to reject nt_per_packet={c8.nt_per_packet}')

    # apply_rfimask=True on a file with no mask must raise, not silently do nothing.
    cm = make_random_chunk(force_ntpp16=True, has_rfi_mask=False)
    with TempChunkFile(cm.to_msgpack(rng=rng)) as fn:
        c = AssembledChunk.from_msgpack(fn)
        for fn_name in ('decode_intensity', 'decode_weights'):
            try:
                getattr(c, fn_name)(apply_rfimask=True)
            except RuntimeError:
                continue
            raise AssertionError(f'expected {fn_name}(apply_rfimask=True) to reject a maskless file')


def test_assembled_chunk(i):
    """One iteration, dispatched from 'python -m pirate_frb test --cfrb'.

    'i' is the iteration index: the expensive full-size draw (a 17 MB file, ~50x the cost of
    a small one) runs only on the first iteration, since its purpose is to cover the real
    CHIME geometry rather than to be sampled repeatedly.
    """

    if i == 0:
        test_decode(make_random_chunk(full_size=True))
        atomic_print('    test_assembled_chunk: full-size (16384 x 1024) draw passed')

    test_parse()                       # any nt_per_packet, parse only
    test_decode()                      # nt_per_packet == 16, parse + decode
    test_rejections()
