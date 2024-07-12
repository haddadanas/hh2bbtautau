# coding: utf-8

"""
Process ID producer relevant for the stitching of the DY samples.
"""

import functools

import law

from columnflow.production import Producer, producer
from columnflow.util import maybe_import, InsertableDict
from columnflow.columnar_util import set_ak_column

from hbt.util import IF_DATASET_IS_DY

np = maybe_import("numpy")
ak = maybe_import("awkward")
sp = maybe_import("scipy")
maybe_import("scipy.sparse")


logger = law.logger.get_logger(__name__)

# helper
set_ak_column_i64 = functools.partial(set_ak_column, value_type=np.int64)


@producer(
    uses={IF_DATASET_IS_DY("LHE.NpNLO", "LHE.Vpt")},
    produces={IF_DATASET_IS_DY("process_id")},
)
def process_ids_dy(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Assigns each dy event a single process id, based on the number of jets and the di-lepton pt of
    the LHE record. This is used for the stitching of the DY samples.
    """
    # as always, we assume that each dataset has exactly one process associated to it
    if len(self.dataset_inst.processes) != 1:
        raise NotImplementedError(
            f"dataset {self.dataset_inst.name} has {len(self.dataset_inst.processes)} processes "
            "assigned, which is not yet implemented",
        )
    process_inst = self.dataset_inst.processes.get_first()

    # get the number of nlo jets and the di-lepton pt
    njets = events.LHE.NpNLO
    pt = events.LHE.Vpt

    # raise a warning if a datasets was already created for a specific "bin" (leaf process),
    # but actually does not fit
    njets_range = process_inst.x("njets", None)
    if njets_range is not None:
        if isinstance(njets_range, int):
            njets_range = (njets_range, njets_range + 1)
        outliers = (njets < njets_range[0]) | (njets >= njets_range[1])
        if ak.any(outliers):
            logger.warning(
                f"dataset {self.dataset_inst.name} is meant to contain njet values in the range "
                f"[{njets_range[0]}, {njets_range[0]}), but found {ak.sum(outliers)} events "
                "outside this range",
            )
    pt_range = process_inst.x("ptll", None)
    if pt_range is not None:
        outliers = (pt < pt_range[0]) | (pt >= pt_range[1])
        if ak.any(outliers):
            logger.warning(
                f"dataset {self.dataset_inst.name} is meant to contain ptll values in the range "
                f"[{pt_range[0]}, {pt_range[1]}), but found {ak.sum(outliers)} events outside this "
                "range",
            )

    # lookup the id and check for invalid values
    process_ids = np.array(self.id_table[0, self.key_func(njets, pt)].todense())[0]
    invalid_mask = process_ids == 0
    if ak.any(invalid_mask):
        raise ValueError(
            f"found {sum(invalid_mask)} dy events that could not be assigned to a process",
        )

    # store them
    events = set_ak_column_i64(events, "process_id", process_ids)

    return events


@process_ids_dy.setup
def process_ids_dy_setup(
    self: Producer,
    reqs: dict,
    inputs: dict,
    reader_targets: InsertableDict,
) -> None:
    # helper to get the njets value from a process in a 2-tuple
    def get_njets(proc):
        njets = proc.x.njets
        if isinstance(njets, list):
            njets = tuple(njets)
        if not isinstance(njets, tuple):
            njets = (njets, njets + 1)
        return njets

    # define stitching ranges for the DY datasets covered by this producer's dy_inclusive_dataset
    stitching_ranges = {}
    for proc in self.dy_leaf_processes:
        njets = get_njets(proc)
        stitching_ranges.setdefault(njets, [])
        if proc.has_aux("ptll"):
            stitching_ranges[njets].append(tuple(proc.x.ptll))

    # sort by the first element of the ptll range
    stitching_ranges = [
        (nj_range, sorted(stitching_ranges[nj_range], key=lambda ptll_range: ptll_range[0]))
        for nj_range in sorted(stitching_ranges.keys(), key=lambda nj_range: nj_range[0])
    ]

    # define a key function that maps njets and pt to a unique key for use in a lookup table
    def key_func(njets, pt):
        # potentially convert single values into arrays
        single = False
        if isinstance(njets, int):
            assert isinstance(pt, (int, float))
            njets = np.array([njets], dtype=np.int32)
            pt = np.array([pt], dtype=np.float32)
            single = True

        # map into bins (index 0 means no binning)
        nj_bins = np.zeros(len(njets), dtype=np.int32)
        pt_bins = np.zeros(len(pt), dtype=np.int32)
        for nj_bin, (nj_range, pt_ranges) in enumerate(stitching_ranges, 1):
            # nj_bin
            nj_mask = (nj_range[0] <= njets) & (njets < nj_range[1])
            nj_bins[nj_mask] = nj_bin
            # pt_bin
            for pt_bin, (pt_min, pt_max) in enumerate(pt_ranges, 1):
                pt_mask = (pt_min <= pt) & (pt < pt_max)
                pt_bins[nj_mask & pt_mask] = pt_bin

        # compute the key
        key = nj_bins * 100 + pt_bins

        return key[0] if single else key

    # save it
    self.key_func = key_func

    # define the lookup table and fill it
    max_key = key_func(99999, 99999)
    self.id_table = sp.sparse.lil_matrix((1, max_key + 1), dtype=np.int64)
    for proc in self.dy_leaf_processes:
        njets = get_njets(proc)[0]
        pt = proc.x("ptll", [-1])[0]
        self.id_table[0, key_func(njets, pt)] = proc.id
