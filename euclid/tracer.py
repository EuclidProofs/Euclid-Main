"""Time-provenance tracing.

`TimeTagged` behaves like a NumPy array, but every element carries a tag: the
latest input row (date index) that the value was computed from. A tag of -1
means the value depends on no input data (a constant).

Tags flow through arithmetic, reductions, windows, indexing and selection, so
after one run of a strategy each output value knows how far forward in time its
inputs reach. Anything the tracer cannot follow raises `Untraceable`; it never
guesses. Callers must then fall back to another method.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from numpy.lib.stride_tricks import sliding_window_view

NO_DATA = -1


class Untraceable(Exception):
    """Raised when an operation's time provenance cannot be tracked."""


_HANDLERS: dict[Callable, Callable] = {}


def _implements(*funcs: Callable) -> Callable:
    def register(handler: Callable) -> Callable:
        for f in funcs:
            _HANDLERS[f] = handler
        return handler
    return register


def _split(x: Any) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(x, TimeTagged):
        return x.values, x.tags
    a = np.asarray(x)
    return a, np.full(a.shape, NO_DATA, dtype=np.int64)


def _plain(x: Any) -> Any:
    if isinstance(x, TimeTagged):
        raise Untraceable("a traced value was used as a parameter of an operation")
    return x


def _max_tags(shape: tuple, *tags: np.ndarray) -> np.ndarray:
    out = np.full(shape, NO_DATA, dtype=np.int64)
    for t in tags:
        out = np.maximum(out, np.broadcast_to(t, shape))
    return out


def _top(tags: np.ndarray) -> int:
    return int(tags.max()) if tags.size else NO_DATA


class TimeTagged(NDArrayOperatorsMixin):
    __array_priority__ = 1000

    def __init__(self, values: Any, tags: Any) -> None:
        self.values = np.asarray(values)
        tags = np.asarray(tags, dtype=np.int64)
        if tags.shape != self.values.shape:
            tags = np.broadcast_to(tags, self.values.shape).copy()
        elif not tags.flags.writeable:
            tags = tags.copy()
        self.tags = tags

    # ---- array-like surface -------------------------------------------------
    shape = property(lambda self: self.values.shape)
    ndim = property(lambda self: self.values.ndim)
    dtype = property(lambda self: self.values.dtype)
    size = property(lambda self: self.values.size)
    T = property(lambda self: TimeTagged(self.values.T, self.tags.T))

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        return f"TimeTagged(shape={self.shape}, latest_tag={_top(self.tags)})"

    def _escape(self, what: str):
        raise Untraceable(what)

    def __array__(self, *args, **kwargs):
        self._escape("the input was converted to a plain array (for example by pandas or np.asarray)")

    def __bool__(self):
        self._escape("a data value was used in Python control flow (if / while)")

    def __float__(self):
        self._escape("a data value was converted to a Python float")

    def __int__(self):
        self._escape("a data value was converted to a Python int")

    __index__ = __int__

    # ---- indexing -----------------------------------------------------------
    @staticmethod
    def _key(key: Any) -> tuple[Any, int]:
        parts = key if isinstance(key, tuple) else (key,)
        idx_tag, plain = NO_DATA, []
        for k in parts:
            if isinstance(k, TimeTagged):
                idx_tag = max(idx_tag, _top(k.tags))
                plain.append(k.values)
            else:
                plain.append(k)
        return (tuple(plain) if isinstance(key, tuple) else plain[0]), idx_tag

    def __getitem__(self, key: Any) -> "TimeTagged":
        plain, idx_tag = self._key(key)
        return TimeTagged(self.values[plain], np.maximum(self.tags[plain], idx_tag))

    def __setitem__(self, key: Any, value: Any) -> None:
        plain, idx_tag = self._key(key)
        v, t = _split(value)
        self.values[plain] = v
        self.tags[plain] = np.maximum(t, idx_tag)
        if idx_tag > NO_DATA:
            # Which elements were written depended on data, so the untouched
            # elements depend on that data too.
            same_shape_mask = (isinstance(key, TimeTagged) and key.dtype == bool
                               and key.shape == self.shape)
            self.tags[...] = np.maximum(self.tags, key.tags if same_shape_mask else idx_tag)

    # ---- NumPy protocols ----------------------------------------------------
    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        if "where" in kwargs:
            raise Untraceable(f"{ufunc.__name__} with a where= argument")
        if ufunc is np.matmul and method == "__call__":
            result = _matmul(*inputs)
        elif method == "__call__":
            if ufunc.signature is not None:
                raise Untraceable(f"generalised ufunc {ufunc.__name__}")
            parts = [_split(x) for x in inputs]
            res = ufunc(*[p[0] for p in parts], **kwargs)
            if isinstance(res, tuple):
                return tuple(TimeTagged(r, _max_tags(np.shape(r), *[p[1] for p in parts]))
                             for r in res)
            result = TimeTagged(res, _max_tags(np.shape(res), *[p[1] for p in parts]))
        elif method == "reduce":
            v, t = _split(inputs[0])
            res = ufunc.reduce(v, **kwargs)
            tag = np.maximum.reduce(t, axis=kwargs.get("axis", 0),
                                    keepdims=kwargs.get("keepdims", False))
            result = TimeTagged(res, np.broadcast_to(tag, np.shape(res)))
        elif method == "accumulate":
            v, t = _split(inputs[0])
            result = TimeTagged(ufunc.accumulate(v, **kwargs),
                                np.maximum.accumulate(t, axis=kwargs.get("axis", 0)))
        else:
            raise Untraceable(f"{ufunc.__name__}.{method}")

        if out is not None:
            target = out[0] if isinstance(out, tuple) else out
            if not isinstance(target, TimeTagged) or (isinstance(out, tuple) and len(out) != 1):
                raise Untraceable(f"{ufunc.__name__} writing into a plain output array")
            target.values[...] = result.values
            target.tags[...] = result.tags
            return target
        return result

    def __array_function__(self, func, types, args, kwargs):
        handler = _HANDLERS.get(func)
        if handler is None:
            raise Untraceable(f"numpy.{getattr(func, '__name__', func)} is not supported by the tracer")
        if "out" in kwargs and kwargs["out"] is not None:
            raise Untraceable(f"numpy.{func.__name__} with an out= argument")
        return handler(func, *args, **kwargs)

    # ---- common methods, routed through the handlers ------------------------
    def astype(self, dtype, **kwargs):
        return TimeTagged(self.values.astype(dtype, **kwargs), self.tags)

    def copy(self):
        return TimeTagged(self.values.copy(), self.tags.copy())

    def reshape(self, *shape):
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
        return np.reshape(self, shape)

    def ravel(self):
        return np.ravel(self)

    flatten = ravel


def _method(func: Callable) -> Callable:
    return lambda self, *a, **k: func(self, *a, **k)


for _name in ("mean", "sum", "std", "var", "min", "max", "prod", "any", "all", "argmax",
              "argmin", "cumsum", "cumprod", "clip", "round", "transpose", "squeeze", "repeat"):
    setattr(TimeTagged, _name, _method(getattr(np, _name)))


# ---- handlers for NumPy functions -------------------------------------------

def _reduction(axis_pos: int) -> Callable:
    def handler(func, a, *args, **kwargs):
        v, t = _split(a)
        args = tuple(_plain(x) for x in args)
        axis = kwargs.get("axis", args[axis_pos - 1] if len(args) >= axis_pos else None)
        res = func(v, *args, **kwargs)
        tag = t.max(axis=axis, keepdims=kwargs.get("keepdims", False)) if t.size else NO_DATA
        return TimeTagged(res, np.broadcast_to(tag, np.shape(res)))
    return handler


_implements(np.sum, np.mean, np.std, np.var, np.min, np.max, np.amin, np.amax, np.prod,
            np.median, np.nansum, np.nanmean, np.nanstd, np.nanvar, np.nanmin, np.nanmax,
            np.nanmedian, np.nanprod, np.any, np.all, np.argmax, np.argmin, np.nanargmax,
            np.nanargmin, np.count_nonzero)(_reduction(1))
_implements(np.percentile, np.quantile, np.nanpercentile, np.nanquantile)(_reduction(2))


@_implements(np.cumsum, np.cumprod, np.nancumsum, np.nancumprod)
def _accumulate(func, a, axis=None, **kwargs):
    v, t = _split(a)
    res = func(v, axis=axis, **kwargs)
    tags = np.maximum.accumulate(t.ravel() if axis is None else t, axis=0 if axis is None else axis)
    return TimeTagged(res, tags)


@_implements(np.diff)
def _diff(func, a, n=1, axis=-1, **kwargs):
    if kwargs:
        raise Untraceable("numpy.diff with prepend/append")
    v, t = _split(a)
    t = np.moveaxis(t, axis, -1)
    for _ in range(n):
        t = np.maximum(t[..., 1:], t[..., :-1])
    return TimeTagged(np.diff(v, n=n, axis=axis), np.moveaxis(t, -1, axis))


@_implements(np.roll, np.flip, np.flipud, np.fliplr, np.transpose, np.reshape, np.ravel,
             np.squeeze, np.expand_dims, np.swapaxes, np.moveaxis, np.broadcast_to, np.repeat,
             np.tile, np.take, np.copy, np.atleast_1d, np.atleast_2d, sliding_window_view)
def _positional(func, a, *args, **kwargs):
    """Operations that only move elements: apply the same move to the tags."""
    v, t = _split(a)
    args = tuple(_plain(x) for x in args)
    return TimeTagged(func(v, *args, **kwargs), func(t, *args, **kwargs))


@_implements(np.concatenate, np.stack, np.vstack, np.hstack, np.column_stack)
def _join(func, arrays, *args, **kwargs):
    parts = [_split(x) for x in arrays]
    kwargs.pop("dtype", None)
    return TimeTagged(func([p[0] for p in parts], *args, **kwargs),
                      func([p[1] for p in parts], *args, **kwargs))


@_implements(np.pad)
def _pad(func, a, pad_width, mode="constant", **kwargs):
    v, t = _split(a)
    if mode == "constant":
        tags = np.pad(t, pad_width, mode="constant", constant_values=NO_DATA)
    elif mode in ("edge", "reflect", "symmetric", "wrap"):
        tags = np.pad(t, pad_width, mode=mode)
    else:
        raise Untraceable(f"numpy.pad with mode={mode!r}")
    return TimeTagged(np.pad(v, pad_width, mode=mode, **kwargs), tags)


@_implements(np.where)
def _where(func, condition, x=None, y=None):
    if x is None or y is None:
        raise Untraceable("numpy.where with a single argument")
    parts = [_split(p) for p in (condition, x, y)]
    res = np.where(*[p[0] for p in parts])
    return TimeTagged(res, _max_tags(res.shape, *[p[1] for p in parts]))


@_implements(np.clip)
def _clip(func, a, a_min=None, a_max=None, **kwargs):
    parts = [_split(p) for p in (a, a_min, a_max) if p is not None]
    res = np.clip(_split(a)[0], None if a_min is None else _split(a_min)[0],
                  None if a_max is None else _split(a_max)[0], **kwargs)
    return TimeTagged(res, _max_tags(res.shape, *[p[1] for p in parts]))


@_implements(np.round, np.around, np.nan_to_num, np.real, np.sort_complex)
def _elementwise(func, a, *args, **kwargs):
    v, t = _split(a)
    return TimeTagged(func(v, *args, **kwargs), t)


@_implements(np.zeros_like, np.ones_like, np.empty_like, np.full_like)
def _like(func, a, *args, **kwargs):
    v, _ = _split(a)
    args = tuple(_plain(x) for x in args)
    res = func(v, *args, **kwargs)
    return TimeTagged(res, np.full(res.shape, NO_DATA, dtype=np.int64))


@_implements(np.sort, np.argsort)
def _sort(func, a, axis=-1, **kwargs):
    v, t = _split(a)
    res = func(v, axis=axis, **kwargs)
    tag = t.max() if axis is None else t.max(axis=axis, keepdims=True)
    return TimeTagged(res, np.broadcast_to(tag, res.shape))


@_implements(np.shape)
def _shape(func, a):
    return a.shape


@_implements(np.ndim)
def _ndim(func, a):
    return a.ndim


@_implements(np.size)
def _size(func, a, axis=None):
    return a.size if axis is None else a.shape[axis]


def _matmul(a: Any, b: Any) -> TimeTagged:
    av, at = _split(a)
    bv, bt = _split(b)
    if not (1 <= av.ndim <= 2 and 1 <= bv.ndim <= 2):
        raise Untraceable("matrix product of arrays with more than 2 dimensions")
    res = np.matmul(av, bv)
    a_rows = at.max(axis=-1) if at.size else NO_DATA  # each output row uses a whole row of a
    b_cols = bt.max(axis=0) if bt.size else NO_DATA  # and a whole column of b
    if av.ndim == 2 and bv.ndim == 2:
        tags = np.maximum(a_rows[:, None], b_cols[None, :])
    else:
        tags = np.maximum(a_rows, b_cols)
    return TimeTagged(res, np.broadcast_to(tags, np.shape(res)))


@_implements(np.dot, np.matmul)
def _dot(func, a, b):
    return _matmul(a, b)


def _convolve_tags(a_tags: np.ndarray, m: int, kernel_tag: int, mode: str) -> np.ndarray:
    n = len(a_tags)
    padded = np.pad(a_tags, m - 1, mode="constant", constant_values=NO_DATA)
    full = np.maximum(sliding_window_view(padded, m).max(axis=-1), kernel_tag)
    if mode == "full":
        return full
    if mode == "valid":
        return full[m - 1:n]
    if mode == "same":
        start = (m - 1) // 2
        return full[start:start + n]
    raise Untraceable(f"convolution mode {mode!r}")


@_implements(np.convolve)
def _convolve(func, a, v, mode="full"):
    av, at = _split(a)
    kv, kt = _split(v)
    if av.ndim != 1 or kv.ndim != 1 or len(av) < len(kv):
        raise Untraceable("numpy.convolve on these shapes")
    return TimeTagged(np.convolve(av, kv, mode=mode), _convolve_tags(at, len(kv), _top(kt), mode))


@_implements(np.correlate)
def _correlate(func, a, v, mode="valid"):
    av, at = _split(a)
    kv, kt = _split(v)
    if av.ndim != 1 or kv.ndim != 1 or len(av) < len(kv):
        raise Untraceable("numpy.correlate on these shapes")
    return TimeTagged(np.correlate(av, kv, mode=mode), _convolve_tags(at, len(kv), _top(kt), mode))


def tag_by_row(values: np.ndarray) -> TimeTagged:
    """Wrap input data so that every element in row t carries the tag t."""
    values = np.array(values, dtype=float)
    rows = np.arange(values.shape[0], dtype=np.int64).reshape((-1,) + (1,) * (values.ndim - 1))
    return TimeTagged(values, np.broadcast_to(rows, values.shape).copy())
