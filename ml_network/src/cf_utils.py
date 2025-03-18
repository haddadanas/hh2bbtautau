# Important utils from columnflow in reduced form to avoid importing the whole package
from __future__ import annotations

__all__ = [
    "UNSET", "EMPTY_INT", "EMPTY_FLOAT", "DotDict", "ItemEval", "Route", "eval_item",
    "get_ak_routes", "remove_ak_column", "flat_np_view", "has_ak_column", "brace_expand"
]

import collections
import itertools
import re
from typing import Sequence, Any


import awkward as ak
import numpy as np
import order as od


#: Placeholder for an unset value.
UNSET = object()

#: Empty-value definition in places where an integer number is expected but not present.
EMPTY_INT = -99999

#: Empty-value definition in places where a float number is expected but not present.
EMPTY_FLOAT = -99999.0


class DotDict(collections.OrderedDict):

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            raise AttributeError("'{}' object has no attribute '{}'".format(
                self.__class__.__name__, attr))

    def __setattr__(self, attr, value):
        self[attr] = value

    def copy(self):
        return self.__class__(self)

    @classmethod
    def wrap(cls, *args, **kwargs):
        wrap = lambda d: cls((k, wrap(v)) for k, v in d.items()) if isinstance(d, dict) else d
        return wrap(collections.OrderedDict(*args, **kwargs))


class ItemEval(object):
    """
    Simple item evaluation helper, similar to NumPy's ``s_``. Example:

    .. code-block:: python

        ItemEval()[0:2, ...]
        # -> (slice(0, 2), Ellipsis)

        ItemEval()("[0:2, ...]")
        # -> (slice(0, 2), Ellipsis)
    """

    def __getitem__(self, item: Any) -> Any:
        return item

    def __call__(self, s: str) -> Any:
        return eval(f"self{s}")


#: ItemEval singleton mimicking a function.
eval_item = ItemEval()


class Route(od.TagMixin):
    """
    Route objects describe the location of columns in nested arrays and are basically wrappers
    around a sequence of nested fields. Additionally, they provide convenience methods for
    conversion into column names, either in dot or nano-style underscore format.

    The constructor takes another *route* instance, a sequence of strings, or a string in dot format
    to initialize the fields. Most operators are overwritten to work with routes in a tuple-like
    fashion. Examples:

    .. code-block:: python

        route = Route("Jet.pt")
        # same as Route(("Jet", "pt"))

        len(route)
        # -> 2

        route.fields
        # -> ("Jet", "pt")

        route.column
        # -> "Jet.pt"

        route.nano_column
        # -> "Jet_pt"

        route[-1]
        # -> "pt"

        route += "jec_up"
        route.fields
        # -> ("Jet", "pt", "jec_up")

        route[1:]
        # -> "pt.jec_up"

    .. py:attribute:: fields

        type: tuple
        read-only

        The fields of this route.

    .. py:attribute:: column

        type: string
        read-only

        The name of the corresponding column in dot format.

    .. py:attribute:: nano_column

        type: string
        read-only

        The name of the corresponding column in nano-style underscore format.

    .. py:attribute:: string_column

        type: string
        read-only

        The name of the corresponding column in dot format, but only consisting of string fields,
        i.e., without slicing or indexing fields.

    .. py:attribute:: string_nano_column

        type: string
        read-only

        The name of the corresponding column in nano-style underscore format, but only consisting of
        string fields, i.e., without slicing or indexing fields.
    """

    DOT_SEP = "."
    NANO_SEP = "_"

    @classmethod
    def slice_to_str(cls, s: slice) -> str:
        s_str = ("" if s.start is None else str(s.start)) + ":"
        s_str += "" if s.stop is None else str(s.stop)
        if s.step is not None:
            s_str += f":{s.step}"
        return s_str

    @classmethod
    def _join(
        cls,
        sep: str,
        fields: Sequence[str | int | slice | type(Ellipsis) | list | tuple],
        _outer: bool = True,
    ) -> str:
        """
        Joins a sequence of *fields* into a string with a certain separator *sep* and returns it.
        """
        s = ""
        for field in fields:
            if isinstance(field, str):
                s += (sep if s else "") + (field if _outer else f"'{field}'")
            elif isinstance(field, int):
                s += f"[{field}]" if _outer else str(field)
            elif isinstance(field, slice):
                field_str = cls.slice_to_str(field)
                s += f"[{field_str}]" if _outer else field_str
            elif isinstance(field, type(Ellipsis)):
                s += "[...]" if _outer else "..."
            elif isinstance(field, tuple):
                field_str = ",".join(cls._join(sep, [f], _outer=False) for f in field)
                s += f"[{field_str}]" if _outer else field_str
            elif isinstance(field, list):
                field_str = ",".join(cls._join(sep, [f], _outer=False) for f in field)
                field_str = f"[{field_str}]"
                s += f"[{field_str}]" if _outer else field_str
            else:
                raise TypeError(f"cannot interpret field '{field}' for joining")
        return s

    @classmethod
    def join(
        cls,
        fields: Sequence[str | int | slice | type(Ellipsis) | list | tuple],
    ) -> str:
        """
        Joins a sequence of strings into a string in dot format and returns it.
        """
        return cls._join(cls.DOT_SEP, fields)

    @classmethod
    def join_nano(
        cls,
        fields: Sequence[str | int | slice | type(Ellipsis) | list | tuple],
    ) -> str:
        """
        Joins a sequence of strings into a string in nano-style underscore format and returns it.
        """
        return cls._join(cls.NANO_SEP, fields)

    @classmethod
    def _split(
        cls,
        sep: str,
        column: str,
    ) -> tuple[str | int | slice | type(Ellipsis) | list | tuple]:
        """
        Splits a string at a separator *sep* and returns the fragments, potentially with selection,
        slice and advanced indexing expressions.

        :param sep: Separator to be used to split *column* into subcomponents
        :param column: Name of the column to be split
        :raises ValueError: If *column* is malformed, specifically if brackets are not encountered
            in pairs (i.e. opening backet w/o closing and vice versa).
        :return: tuple of subcomponents extracted from *column*
        """
        # first extract and replace possibly nested slices
        # note: a regexp would be far cleaner, but there are edge cases which possibly require
        #       sequential regexp evaluations which might be less maintainable
        slices = []
        repl = lambda i: f"__slice_{i}__"
        repl_cre = re.compile(r"^__slice_(\d+)__$")
        while True:
            depth = 0
            slice_start = -1
            for i, s in enumerate(column):
                if s == "[":
                    depth += 1
                    # remember the starting point when the slice started
                    if depth == 1:
                        slice_start = i
                elif s == "]":
                    if depth <= 0:
                        raise ValueError(f"invalid column format '{column}'")
                    depth -= 1
                    # when we are back at depth 0, the slice ended
                    if depth == 0:
                        # save the slice
                        slices.append(column[slice_start:i + 1])
                        # insert a temporary replacement
                        start = column[:slice_start]
                        tmp = repl(len(slices) - 1)
                        rest = column[i + 1:]
                        if rest and not rest.startswith((sep, "[")):
                            raise ValueError(f"invalid column format '{column}'")
                        column = start + (sep if start else "") + tmp + rest
                        # start over
                        break
            else:
                # when this point is reached all slices have been replaced
                break

        # evaluate all slices
        slices = [eval_item(s) for s in slices]

        # split parts and fill back evaluated slices
        parts = []
        for part in column.split(sep):
            m = repl_cre.match(part)
            parts.append(slices[int(m.group(1))] if m else part)

        return tuple(parts)

    @classmethod
    def split(cls, column: str) -> tuple[str | int | slice | type(Ellipsis) | list | tuple]:
        """
        Splits a string assumed to be in dot format and returns the fragments, potentially with
        selection, slice and advanced indexing expressions.

        :param column: Name of the column to be split
        :raises ValueError: If *column* is malformed, specifically if brackets
            are not encountered in pairs (i.e. opening backet w/o closing and vice versa).
        :return: tuple of subcomponents extracted from *column*
        """
        return cls._split(cls.DOT_SEP, column)

    @classmethod
    def split_nano(cls, column: str) -> tuple[str | int | slice | type(Ellipsis) | list | tuple]:
        """
        Splits a string assumed to be in nano-style underscore format and returns the fragments,
        potentially with selection, slice and advanced indexing expressions.

        :param column: Name of the column to be split
        :raises ValueError: If *column* is malformed, specifically if brackets are not encountered
            in pairs (i.e. opening backet w/o closing and vice versa).
        :return: tuple of subcomponents extracted from *column*
        """
        return cls._split(cls.NANO_SEP, column)

    def __init__(self, route: Any | None = None, tags: dict | None = None):
        super().__init__(tags=tags)

        # initial storage of fields
        self._fields = []

        # use the add method to set the initial value
        if route:
            self.add(route)

            # when route was a Route instance itself, add its tags
            if isinstance(route, Route):
                self.add_tag(route.tags)

    @property
    def fields(self) -> tuple:
        return tuple(self._fields)

    @property
    def column(self) -> str:
        return self.join(self._fields)

    @property
    def nano_column(self) -> str:
        return self.join_nano(self._fields)

    @property
    def string_column(self) -> str:
        return self.join(f for f in self._fields if isinstance(f, str))

    @property
    def string_nano_column(self) -> str:
        return self.join_nano(f for f in self._fields if isinstance(f, str))

    def __str__(self) -> str:
        return self.join(self._fields)

    def __repr__(self) -> str:
        tags = ""
        if self.tags:
            tags = f" (tags={','.join(sorted(self.tags))})"
        return f"<{self.__class__.__name__} '{self}'{tags} at {hex(id(self))}>"

    def __hash__(self) -> int:
        return hash(str(self.fields))

    def __len__(self) -> int:
        return len(self._fields)

    def __eq__(self, other: Route | str | Sequence[str | int | slice | type(Ellipsis) | list]) -> bool:
        if isinstance(other, Route):
            return self.fields == other.fields
        if isinstance(other, (list, tuple)):
            return self.fields == tuple(other)
        if isinstance(other, str):
            return self.column == other
        return False

    def __lt__(self, other: Route | str | Sequence[str | int | slice | type(Ellipsis) | list]) -> bool:
        if isinstance(other, Route):
            return self.fields < other.fields
        if isinstance(other, (list, tuple)):
            return self.fields < tuple(other)
        if isinstance(other, str):
            return self.column < other
        return False

    def __bool__(self) -> bool:
        return len(self._fields) > 0

    def __nonzero__(self) -> bool:
        return self.__bool__()

    def __add__(
        self,
        other: Route | str | Sequence[str | int | slice | type(Ellipsis) | list | tuple],
    ) -> Route:
        route = self.copy()
        route.add(other)
        return route

    def __radd__(
        self,
        other: Route | str | Sequence[str | int | slice | type(Ellipsis) | list | tuple],
    ) -> Route:
        other = Route(other)
        other.add(self)
        return other

    def __iadd__(
        self,
        other: Route | str | Sequence[str | int | slice | type(Ellipsis) | list | tuple],
    ) -> Route:
        self.add(other)
        return self

    def __getitem__(
        self,
        index: Any,
    ) -> Route | str | int | slice | type(Ellipsis) | list | tuple:
        # detect slicing and return a new instance with the selected fields
        field = self._fields.__getitem__(index)
        return field if isinstance(index, int) else self.__class__(field)

    def __setitem__(
        self,
        index: Any,
        value: str | int | slice | type(Ellipsis) | list | tuple,
    ) -> None:
        self._fields.__setitem__(index, value)

    def add(
        self,
        other: Route | str | Sequence[str | int | slice | type(Ellipsis) | list | tuple],
    ) -> None:
        """
        Adds an *other* route instance, or the fields extracted from either a sequence of strings or
        a string in dot format to the fields if *this* instance. A *ValueError* is raised when
        *other* could not be interpreted.
        """
        if isinstance(other, Route):
            self._fields.extend(other._fields)
        elif isinstance(other, (list, tuple)):
            self._fields.extend(list(other))
        elif isinstance(other, str):
            self._fields.extend(self.split(other))
        else:
            raise ValueError(f"cannot add '{other}' to route '{self}'")

    def pop(self, index: int = -1) -> str:
        """
        Removes a field at *index* and returns it.
        """
        return self._fields.pop(index)

    def reverse(self) -> None:
        """
        Reverses the fields of this route in-place.
        """
        self._fields[:] = self._fields[::-1]

    def copy(self) -> Route:
        """
        Returns a copy if this instance.
        """
        return self.__class__(self._fields)

    def apply(
        self,
        ak_array: ak.Array,
        null_value: Any = UNSET,
    ) -> ak.Array:
        """
        Returns a selection of *ak_array* using the fields in this route. When the route is empty,
        *ak_array* is returned unchanged. When *null_value* is set, it is used to fill up missing
        elements in the selection corresponding to this route. Example:

        .. code-block:: python

            # select the 6th jet in each event
            Route("Jet.pt[:, 5]").apply(events)
            # -> might lead to "index out of range" errors for events with fewer jets

            Route("Jet.pt[:, 5]").apply(events, -999)
            # -> [
            #     34.15625,
            #     17.265625,
            #     -999.0,  # 6th jet was missing here
            #     19.40625,
            #     ...
            # ]
        """
        if not self:
            return ak_array

        pad = null_value is not UNSET

        # traverse fields and perform the lookup iteratively
        res = ak_array
        for i, f in enumerate(self._fields):
            # in most scenarios we can just look for the field except when
            # - padding is enabled, and
            # - f is the last field, and
            # - f is an integer (indexing), list (advanced indexing) or tuple (slicing)
            if not pad or not isinstance(f, (list, tuple, int)) or i < len(self) - 1:
                res = res[f]

            else:
                # at this point f is either an integer, a list or a tuple and padding is enabled,
                # so determine the pad size depending on f
                max_idx = -1
                pad_axis = 0
                if isinstance(f, int):
                    max_idx = f
                elif isinstance(f, list):
                    if all(isinstance(i, int) for i in f):
                        max_idx = max(f)
                else:  # tuple
                    last = f[-1]
                    if isinstance(last, int):
                        max_idx = last
                        pad_axis = len(f) - 1
                    elif isinstance(last, list) and all(isinstance(i, int) for i in last):
                        max_idx = max(last)
                        pad_axis = len(f) - 1

                # do the padding on the last axis
                if max_idx >= 0:
                    res = ak.pad_none(res, max_idx + 1, axis=pad_axis)

                # lookup the field
                res = res[f]

                # fill nones
                if max_idx >= 0 and null_value is not None:
                    # res can be an array or a value itself
                    # TODO: is there a better check than testing for the type attribute?
                    if getattr(res, "type", None) is None:
                        if res is None:
                            res = null_value
                    else:
                        res = ak.fill_none(res, null_value)

        return res


def get_ak_routes(
    ak_array: ak.Array,
    max_depth: int = 0,
) -> list[Route]:
    """
    Extracts all routes pointing to columns of a potentially deeply nested awkward array *ak_array*
    and returns them in a list of :py:class:`Route` instances. Example:

    .. code-block:: python

        # let arr be a nested array (e.g. from opening nano files via coffea)
        # (note that route objects serialize to strings using dot format)

        print(get_ak_routes(arr))
        # [
        #    "event",
        #    "luminosityBlock",
        #    "run",
        #    "Jet.pt",
        #    "Jet.mass",
        #    ...
        # ]

    When positive, *max_depth* controls the maximum size of returned route tuples. When negative,
    routes are shortened by the passed amount of elements. In both cases, only unique routes are
    returned.
    """
    routes = []

    # use recursive lookup pattern over (container, current route) pairs
    lookup = [(ak_array, ())]
    while lookup:
        arr, fields = lookup.pop(0)
        if getattr(arr, "fields", None) and (max_depth <= 0 or len(fields) < max_depth):
            # extend the lookup with nested fields
            lookup.extend([
                (arr[field], fields + (field,))
                for field in arr.fields
            ])
        else:
            # no sub fields found or positive max_depth reached, store the route
            # but check negative max_depth first
            if max_depth < 0:
                fields = fields[:max_depth]
            # create the route
            route = Route(fields)
            # add when not empty and unique
            if route and route not in routes:
                routes.append(route)

    return routes


def remove_ak_column(
    ak_array: ak.Array,
    route: Route | Sequence[str] | str,
    remove_empty: bool = True,
    silent: bool = False,
) -> ak.Array:
    """
    Removes a *route* from an awkward array *ak_array* and returns a new view with the corresponding
    column removed. When *route* points to a nested field that would be empty after removal, the
    parent field is removed completely unless *remove_empty* is *False*.

    Note that *route* can be a :py:class:`Route` instance, a sequence of strings where each string
    refers to a subfield, e.g. ``("Jet", "pt")``, or a string with dot format (e.g. ``"Jet.pt"``).
    Unless *silent* is *True*, a *ValueError* is raised when the route does not exist.
    """
    # force creating a view for consistent behavior
    ak_array = ak.Array(ak_array)

    # verify that the route is not empty
    route = Route(route)
    if not route:
        if silent:
            return ak_array
        raise ValueError("route must not be empty")

    # verify that the route exists
    if not has_ak_column(ak_array, route):
        if silent:
            return ak_array
        raise ValueError(f"no column found in array for route '{route}'")

    # remove it
    ak_array = ak.without_field(ak_array, route.fields)

    # remove empty parent fields
    if remove_empty and len(route) > 1:
        for i in range(len(route) - 1):
            parent_route = route[:-(i + 1)]
            if not parent_route.apply(ak_array).fields:
                ak_array = ak.without_field(ak_array, parent_route.fields)

    return ak_array


def flat_np_view(ak_array: ak.Array, axis: int | None = None) -> np.array:
    """
    Takes an *ak_array* and returns a fully flattened numpy view. The flattening is applied along
    *axis*. See *ak.flatten* for more info.
    """
    return np.asarray(ak.flatten(ak_array, axis=axis))


def has_ak_column(
    ak_array: ak.Array,
    route: Route | Sequence[str] | str,
) -> bool:
    """
    Returns whether an awkward array *ak_array* contains a nested field identified by a *route*. A
    route can be a :py:class:`Route` instance, a tuple of strings where each string refers to a
    subfield, e.g. ``("Jet", "pt")``, or a string with dot format (e.g. ``"Jet.pt"``).
    """
    route = Route(route)

    # handle empty route
    if not route:
        return False

    try:
        route.apply(ak_array)
    except (ValueError, IndexError):
        return False

    return True


def brace_expand(s, split_csv=False, escape_csv_sep=True):
    """
    Expands brace statements in a string *s* and returns a list containing all possible string
    combinations. When *split_csv* is *True*, the input string is split by all comma characters
    located outside braces, except for escaped ones when *escape_csv_sep* is *True*, and the
    expansion is performed sequentially on all elements. Example:

    .. code-block:: python

        brace_expand("A{1,2}B")
        # -> ["A1B", "A2B"]

        brace_expand("A{1,2}B{3,4}C")
        # -> ["A1B3C", "A1B4C", "A2B3C", "A2B4C"]

        brace_expand("A{1,2}B,C{3,4}D")
        # note the full 2x2 expansion
        # -> ["A1B,C3D", "A1B,C4D", "A2B,C3D", "A2B,C4D"]

        brace_expand("A{1,2}B,C{3,4}D", split_csv=True)
        # note the 2+2 sequential expansion
        # -> ["A1B", "A2B", "C3D", "C4D"]

        brace_expand("A{1,2}B,C{3}D", split_csv=True)
        # note the 2+1 sequential expansion
        # -> ["A1B", "A2B", "C3D"]
    """
    # first, replace escaped braces
    br_open = "__law_brace_open__"
    br_close = "__law_brace_close__"
    s = s.replace(r"\{", br_open).replace(r"\}", br_close)

    # compile the expression that finds brace statements
    cre = re.compile(r"\{[^\{]*\}")

    # take into account csv splitting
    if split_csv:
        # replace csv separators in brace statements to avoid splitting
        br_sep = "__law_brace_csv_sep__"
        _s = cre.sub(lambda m: m.group(0).replace(",", br_sep), s)
        # replace escaped commas
        if escape_csv_sep:
            escaped_sep = "__law_escaped_csv_sep__"
            _s = _s.replace(r"\,", escaped_sep)
        # split by real csv separators except escaped ones when requested
        parts = _s.split(",")
        # add back normal commas
        if escape_csv_sep:
            parts = [part.replace(escaped_sep, ",") for part in parts]
        # start recursion when a comma was found, otherwise continue
        if len(parts) > 1:
            # replace csv separators in braces again and recurse
            parts = [part.replace(br_sep, ",") for part in parts]
            return sum((brace_expand(part, split_csv=False) for part in parts), [])

    # split the string into n sequences with values to expand and n+1 fixed entities
    sequences = cre.findall(s)
    entities = cre.split(s)
    if len(sequences) + 1 != len(entities):
        raise ValueError("the number of sequences ({}) and the number of fixed entities ({}) are "
            "not compatible".format(",".join(sequences), ",".join(entities)))

    # split each sequence by comma
    sequences = [seq[1:-1].split(",") for seq in sequences]

    # create a template using the fixed entities used for formatting
    tmpl = "{}".join(entities)

    # build all combinations
    res = []
    for values in itertools.product(*sequences):
        _s = tmpl.format(*values)

        # insert escaped braces again
        _s = _s.replace(br_open, r"\{").replace(br_close, r"\}")

        res.append(_s)

    return res
