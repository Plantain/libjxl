"""ctypes bindings for the pinned libjxl hybrid tile API (no numpy dependency).

Set JXL_HYBRID_LIBRARY or pass library= to Encoder. Inputs are contiguous
512x512x3 uint8 buffers (bytes, memoryview, or numpy arrays). StageDump owns its
storage; .fields exposes the public C views for GPU-stage comparisons, and
.overrides([1, 3]) borrows selected stage outputs. Use one Encoder context for
repeated calls. encode_batch uses the C library's persistent tile pool.
"""
from __future__ import annotations

import ctypes as C
import os
from pathlib import Path
import threading

U8 = C.POINTER(C.c_uint8)
I8 = C.POINTER(C.c_int8)
I32 = C.POINTER(C.c_int32)
F32 = C.POINTER(C.c_float)


class TileDecisions(C.Structure):
    _fields_ = [("xsize_blocks", C.c_uint32), ("ysize_blocks", C.c_uint32),
                ("global_scale", C.c_int32), ("quant_dc", C.c_int32),
                ("raw_quant_field", I32), ("ac_strategy", U8),
                ("ytox_map", I8), ("ytob_map", I8),
                ("ytox_dc", C.c_int32), ("ytob_dc", C.c_int32),
                ("color_factor", C.c_uint32), ("base_correlation_x", C.c_float),
                ("base_correlation_b", C.c_float), ("epf_sharpness", U8),
                ("x_qm_scale", C.c_uint32), ("b_qm_scale", C.c_uint32)]


class ColorMap(C.Structure):
    _fields_ = [("ytox_map", I8), ("ytob_map", I8),
                ("ytox_dc", C.c_int32), ("ytob_dc", C.c_int32),
                ("color_factor", C.c_uint32), ("base_correlation_x", C.c_float),
                ("base_correlation_b", C.c_float)]


class Stage1(C.Structure):
    _fields_ = [("quant_field", F32), ("masking", F32), ("masking1x1", F32)]


class Stage4(C.Structure):
    _fields_ = [("quant_field", F32), ("raw_quant_field", I32),
                ("global_scale", C.c_int32), ("quant_dc", C.c_int32),
                ("cmap", ColorMap)]


class Stage5(C.Structure):
    _fields_ = [("raw_quant_field", I32), ("global_scale", C.c_int32),
                ("quant_dc", C.c_int32)]


class _Dump(C.Structure):
    _fields_ = [("decisions", TileDecisions), ("s1", Stage1), ("s2", ColorMap),
                ("s3", U8), ("s4", Stage4), ("s5", Stage5), ("s6", U8),
                ("opsin", F32 * 3), ("codestream", U8),
                ("codestream_size", C.c_size_t), ("opaque", C.c_void_p)]


class StageOverrides(C.Structure):
    _fields_ = [("stages", C.c_uint32), ("s1", C.POINTER(Stage1)),
                ("s2", C.POINTER(ColorMap)), ("s3", U8),
                ("s4", C.POINTER(Stage4)), ("s5", C.POINTER(Stage5)), ("s6", U8)]


class Options(C.Structure):
    _fields_ = [("distance", C.c_float), ("effort", C.c_uint32), ("threads", C.c_uint32)]


class _Input(C.Structure):
    _fields_ = [("rgb", U8), ("stride", C.c_size_t),
                ("decisions", C.POINTER(TileDecisions)), ("opsin", F32 * 3),
                ("overrides", C.POINTER(StageOverrides))]


class _Output(C.Structure):
    _fields_ = [("data", U8), ("size", C.c_size_t), ("status", C.c_int)]


def _library(path=None):
    if path is None:
        path = os.environ.get("JXL_HYBRID_LIBRARY")
    if path is None:
        root = Path(__file__).resolve().parents[2]
        for name in ("libjxl_hybrid.so", "libjxl_hybrid.dylib", "jxl_hybrid.dll"):
            for build in ("build-hybrid", "build"):
                candidate = root / build / "lib" / name
                if candidate.exists():
                    path = candidate
                    break
            if path is not None:
                break
    if path is None:
        raise FileNotFoundError("Set JXL_HYBRID_LIBRARY to libjxl_hybrid.so")
    lib = C.CDLL(str(path))
    signatures = {
        "JxlHybridEncoderCreate": ([C.POINTER(Options)], C.c_void_p),
        "JxlHybridEncoderDestroy": ([C.c_void_p], None),
        "JxlHybridStageDumpCreate": ([], C.POINTER(_Dump)),
        "JxlHybridStageDumpDestroy": ([C.POINTER(_Dump)], None),
        "JxlHybridFree": ([C.c_void_p], None),
        "JxlHybridRecord": ([C.c_void_p, U8, C.c_size_t, C.POINTER(_Dump)], C.c_int),
        "JxlHybridEncodeTile": ([C.c_void_p, U8, C.c_size_t, C.POINTER(TileDecisions),
                                  C.POINTER(F32), C.POINTER(StageOverrides),
                                  C.POINTER(U8), C.POINTER(C.c_size_t)], C.c_int),
        "JxlHybridEncodeBatch": ([C.c_void_p, C.POINTER(_Input), C.c_size_t,
                                   C.POINTER(_Output)], C.c_int),
    }
    for name, (args, result) in signatures.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, result
    return lib


def _check(status):
    if status:
        raise ValueError(f"libjxl_hybrid failed (status {status}: "
                         f"{'invalid input' if status == 1 else 'encoder error'})")


def _rgb_buffer(rgb):
    view = memoryview(rgb)
    if view.itemsize != 1 or view.format not in ("B", "b", "c"):
        raise ValueError("RGB must be uint8 bytes")
    if not view.c_contiguous or view.nbytes != 512 * 512 * 3:
        raise ValueError("RGB must be a contiguous 512x512x3 uint8 tile")
    if view.ndim != 1 and view.shape != (512, 512, 3):
        raise ValueError("RGB array shape must be (512, 512, 3)")
    raw = view.cast("B")
    kind = C.c_uint8 * raw.nbytes
    return kind.from_buffer_copy(raw) if raw.readonly else kind.from_buffer(raw)


def _borrow(value):
    if value is None:
        return None
    owner = getattr(value, "_owner", None)
    if owner is not None:
        owner._check_open()
    return C.pointer(value)


class StageDump:
    def __init__(self, lib, distance):
        self._lib, self.distance = lib, distance
        self._ptr = lib.JxlHybridStageDumpCreate()
        if not self._ptr:
            raise MemoryError("Could not allocate stage dump")

    def _check_open(self):
        if not self._ptr:
            raise ValueError("StageDump is closed")

    @property
    def fields(self):
        self._check_open()
        return self._ptr.contents

    @property
    def decisions(self):
        result = self.fields.decisions
        result._owner = self
        return result

    @property
    def opsin(self):
        result = self.fields.opsin
        result._owner = self
        return result

    @property
    def codestream(self):
        d = self.fields
        return C.string_at(d.codestream, d.codestream_size)

    def overrides(self, stages=range(1, 7)):
        stages = tuple(stages)
        if any(s not in range(1, 7) for s in stages):
            raise ValueError("Stages are numbered 1..6")
        d = self.fields
        o = StageOverrides(sum(1 << (s - 1) for s in set(stages)),
                           C.pointer(d.s1), C.pointer(d.s2), d.s3,
                           C.pointer(d.s4), C.pointer(d.s5), d.s6)
        o._owner = self
        return o

    def close(self):
        if self._ptr:
            self._lib.JxlHybridStageDumpDestroy(self._ptr)
            self._ptr = None

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()


class Encoder:
    def __init__(self, distance=1.2, effort=7, threads=16, library=None):
        self._ptr = None
        self._lock = threading.RLock()
        self._lib = _library(library)
        self.distance = distance
        self._ptr = self._lib.JxlHybridEncoderCreate(C.byref(Options(distance, effort, threads)))
        if not self._ptr:
            raise ValueError("Invalid options or encoder allocation failed")

    def _check_open(self):
        if not self._ptr:
            raise ValueError("Encoder is closed")

    def record(self, rgb):
        buf = _rgb_buffer(rgb)
        with self._lock:
            self._check_open()
            dump = StageDump(self._lib, self.distance)
            _check(self._lib.JxlHybridRecord(self._ptr, buf, 1536, dump._ptr))
            return dump

    def encode(self, rgb, decisions=None, overrides=None, opsin=None):
        buf = _rgb_buffer(rgb)
        if isinstance(decisions, StageDump):
            decisions = decisions.decisions
        dp, op = _borrow(decisions), _borrow(overrides)
        if opsin is not None and getattr(opsin, "_owner", None):
            opsin._owner._check_open()
        data, size = U8(), C.c_size_t()
        with self._lock:
            self._check_open()
            _check(self._lib.JxlHybridEncodeTile(self._ptr, buf, 1536, dp,
                                               opsin, op, C.byref(data), C.byref(size)))
            try:
                return C.string_at(data, size.value)
            finally:
                self._lib.JxlHybridFree(data)

    def encode_batch(self, rgbs, decisions=None, overrides=None, opsin=None):
        rgbs = list(rgbs)
        count = len(rgbs)
        columns = []
        for seq in (decisions, overrides, opsin):
            seq = [None] * count if seq is None else list(seq)
            if len(seq) != count:
                raise ValueError("Batch input lengths differ")
            columns.append(seq)
        decisions, overrides, opsin = columns
        buffers = [_rgb_buffer(rgb) for rgb in rgbs]
        inputs, outputs = (_Input * count)(), (_Output * count)()
        keepalive = []
        for i, buf in enumerate(buffers):
            decision = decisions[i]
            if isinstance(decision, StageDump):
                decision = decision.decisions
            dp, op = _borrow(decision), _borrow(overrides[i])
            keepalive.extend((decision, dp, op))
            inputs[i].rgb, inputs[i].stride = buf, 1536
            inputs[i].decisions, inputs[i].overrides = dp, op
            if opsin[i] is not None:
                if getattr(opsin[i], "_owner", None):
                    opsin[i]._owner._check_open()
                inputs[i].opsin = opsin[i]
        with self._lock:
            self._check_open()
            status = self._lib.JxlHybridEncodeBatch(self._ptr, inputs, count, outputs)
            try:
                _check(status)
                return [C.string_at(o.data, o.size) for o in outputs]
            finally:
                for out in outputs:
                    self._lib.JxlHybridFree(out.data)

    def close(self):
        with self._lock:
            if self._ptr:
                self._lib.JxlHybridEncoderDestroy(self._ptr)
                self._ptr = None

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()


_local = threading.local()


def _default(distance, effort, threads=1):
    key = (distance, effort, threads)
    if getattr(_local, "key", None) != key:
        old = getattr(_local, "encoder", None)
        if old is not None:
            old.close()
        _local.encoder = Encoder(distance, effort, threads)
        _local.key = key
    return _local.encoder


def record(rgb, *, distance=1.2, effort=7):
    return _default(distance, effort).record(rgb)


def encode(rgb, decisions=None, overrides=None, *, distance=1.2, effort=7, opsin=None):
    return _default(distance, effort).encode(rgb, decisions, overrides, opsin)


def encode_batch(rgbs, decisions=None, overrides=None, *, distance=1.2, effort=7, opsin=None, threads=16):
    return _default(distance, effort, threads).encode_batch(rgbs, decisions, overrides, opsin)


def scoreboard_encode(rgb, distance):
    """WP0 spec: py:jxl_hybrid:scoreboard_encode, CPU e8 heuristics/e7 backend.

    This is the CPU oracle for scoreboard comparisons. Full-decision backend
    timing is measured separately by hybrid_gate with decisions precomputed.
    """
    return encode(rgb, distance=distance)
