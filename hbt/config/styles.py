# coding: utf-8

"""
Style definitions.
"""

import order as od

from columnflow.util import DotDict


def stylize_processes(config: od.Config) -> None:
    """
    Adds process colors and adjust labels.
    """
    cfg = config

    # recommended cms colors
    cfg.x.colors = DotDict(
        bright_blue="#3f90da",
        dark_blue="#011c87",
        purple="#832db6",
        aubergine="#964a8b",
        yellow="#f7c331",
        bright_orange="#ffa90e",
        dark_orange="#e76300",
        red="#bd1f01",
        teal="#92dadd",
        grey="#94a4a2",
        brown="#a96b59",
        color1="#ff0000",
        color2="#00ff00",
        color3="#0000ff",
        color4="#4d6b43",
    )

    dy_label = {
        "dy_m4to10": r"DY $m_{4-10}$",
        "dy_m10to50": r"DY $m_{10-50}$",
        "dy_m50toinf": r"DY $m_{50-\infty}$",
        "dy_m50toinf_0j": r"DY $m_{50-\infty}$ 0j",
        "dy_m50toinf_1j_pt0to40": r"DY $m_{50-\infty}$ 1j $p_{T}^{0-40}$",
        "dy_m50toinf_1j_pt40to100": r"DY $m_{50-\infty}$ 1j $p_{T}^{40-100}$",
        "dy_m50toinf_1j_pt100to200": r"DY $m_{50-\infty}$ 1j $p_{T}^{100-200}$",
        "dy_m50toinf_1j_pt200to400": r"DY $m_{50-\infty}$ 1j $p_{T}^{200-400}$",
        "dy_m50toinf_1j_pt400to600": r"DY $m_{50-\infty}$ 1j $p_{T}^{400-600}$",
        "dy_m50toinf_1j_pt600toinf": r"DY $m_{50-\infty}$ 1j $p_{T}^{600-\infty}$",
        "dy_m50toinf_2j_pt0to40": r"DY $m_{50-\infty}$ 2j $p_{T}^{0-40}$",
        "dy_m50toinf_2j_pt40to100": r"DY $m_{50-\infty}$ 2j $p_{T}^{40-100}$",
        "dy_m50toinf_2j_pt100to200": r"DY $m_{50-\infty}$ 2j $p_{T}^{100-200}$",
        "dy_m50toinf_2j_pt200to400": r"DY $m_{50-\infty}$ 2j $p_{T}^{200-400}$",
        "dy_m50toinf_2j_pt400to600": r"DY $m_{50-\infty}$ 2j $p_{T}^{400-600}$",
        "dy_m50toinf_2j_pt600toinf": r"DY $m_{50-\infty}$ 2j $p_{T}^{600-\infty}$",
    }

    for kl in ["0", "1", "2p45", "5"]:
        if (p := config.get_process(f"hh_ggf_hbb_htt_kl{kl}_kt1", default=None)):
            p.color1 = cfg.x.colors.bright_blue
            p.label = (
                r"$HH_{ggf} \rightarrow bb\tau\tau$ __SCALE__"
                "\n"
                rf"($\kappa_{{\lambda}}$={kl.replace('p', '.')},$\kappa_{{t}}$=1)"
            )

    if (p := config.get_process("hh_vbf_hbb_htt_kv1_k2v1_kl1", default=None)):
        p.color1 = cfg.x.colors.dark_blue
        p.label = (
            r"$HH_{vbf} \rightarrow bb\tau\tau$ __SCALE__"
            "\n"
            r"($\kappa_{\lambda}$=1,$\kappa_{V}$=1,$\kappa_{2V}$=1)"
        )

    if (p := config.get_process("h", default=None)):
        p.color1 = cfg.x.colors.purple

    if (p := config.get_process("tt", default=None)):
        p.color1 = cfg.x.colors.bright_orange
        p.label = r"$t\bar{t}$"

    if (p := config.get_process("st", default=None)):
        p.color1 = cfg.x.colors.aubergine

    if (p := config.get_process("dy", default=None)):
        p.color1 = cfg.x.colors.dark_orange

    if (p := config.get_process("vv", default=None)):
        p.color1 = cfg.x.colors.yellow

    if (p := config.get_process("vvv", default=None)):
        p.color1 = cfg.x.colors.yellow

    if (p := config.get_process("multiboson", default=None)):
        p.color1 = cfg.x.colors.yellow

    if (p := config.get_process("w", default=None)):
        p.color1 = cfg.x.colors.teal
        p.label = "W"

    if (p := config.get_process("z", default=None)):
        p.color1 = cfg.x.colors.brown
        p.label = "Z"

    if (p := config.get_process("v", default=None)):
        p.color1 = cfg.x.colors.teal

    if (p := config.get_process("ewk", default=None)):
        p.color1 = cfg.x.colors.brown

    if (p := config.get_process("ttv", default=None)):
        p.color1 = cfg.x.colors.grey
        p.label = r"$t\bar{t} + V$"

    if (p := config.get_process("ttvv", default=None)):
        p.color1 = cfg.x.colors.grey
        p.label = r"$t\bar{t} + VV$"

    if (p := config.get_process("tt_multiboson", default=None)):
        p.color1 = cfg.x.colors.grey

    if (p := config.get_process("qcd", default=None)):
        p.color1 = cfg.x.colors.red

    for dy, col in zip(cfg.x.dy_group, cfg.x.colors.values()):
        if (p := config.get_process(dy, default=None)):
            p.color1 = col
            p.label = dy_label[dy]
