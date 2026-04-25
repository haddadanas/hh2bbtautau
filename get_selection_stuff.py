import awkward as ak

from glob import glob
import re
import order as od

from hbt.ml.ml_metrices import SelectionMetrix

# path_pattern = re.compile(r"/data/dust/user/haddadan/hbt_cache/hbt_store/analysis_hbt/cf\.SelectEvents/run3_2022_preEE_limited/(.*)/nominal/calib__default/sel__default/sel0/results_\d\.parquet")
path_pattern = re.compile(r"/data/dust/user/haddadan/hbt_cache/hbt_store/analysis_hbt/cf\.SelectEvents/run3_2022_preEE_limited/(.*)/nominal/calib__default/sel__loose/sel1/results_\d\.parquet")

# pathes = glob("/data/dust/user/haddadan/hbt_cache/hbt_store/analysis_hbt/cf.SelectEvents/run3_2022_preEE_limited/*/nominal/calib__default/sel__default/sel0/results_*.parquet")
pathes = glob("/data/dust/user/haddadan/hbt_cache/hbt_store/analysis_hbt/cf.SelectEvents/run3_2022_preEE_limited/*/nominal/calib__default/sel__loose/sel1/results_*.parquet")


# dict with keys = match group, value path
d = {path_pattern.match(path).group(1): path for path in pathes}

# get the arrays
arrays = {key: ak.from_parquet(val) for key, val in d.items()}

# base mask
def get_base_mask(arr):
    mask = ak.ones_like(arr.lepton) == 1
    for key in arr.fields:
        if key in ["lepton", "all_but_bjet"]:
            continue
        mask = mask & (arr[key])
    return mask

base_masks = {key: get_base_mask(arr.steps) for key, arr in arrays.items()}
lepton_masks = {key: arr.steps.lepton for key, arr in arrays.items()}

sel_lepton_masks = {key: lepton_masks[key][base_masks[key]] for key in lepton_masks}
trues = {key: ak.ones_like(arr) for key, arr in sel_lepton_masks.items()}
trues = {key: arr == 1 if key.startswith("hh") else arr == 0 for key, arr in trues.items()}

sel_lepton_masks = ak.concatenate(list(sel_lepton_masks.values()), axis=0).to_numpy() 
trues = ak.concatenate(list(trues.values()), axis=0).to_numpy()

metric = SelectionMetrix(od.Config(name="", id=0))

eff = metric.selection_efficiency(selection_mask=sel_lepton_masks)
acc = metric.signal_acceptance(trues, selection_mask=sel_lepton_masks)
pur = metric.signal_purity(trues, selection_mask=sel_lepton_masks)


print(f"eff: {eff}, acc: {acc}, pur: {pur}")