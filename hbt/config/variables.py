# coding: utf-8

"""
Definition of variables.
"""

import order as od

from columnflow.columnar_util import EMPTY_FLOAT


def add_variables(config: od.Config) -> None:
    """
    Adds all variables to a *config*.
    """
    config.add_variable(
        name="event",
        expression="event",
        binning=(1, 0.0, 1.0e9),
        x_title="Event number",
        discrete_x=True,
    )
    config.add_variable(
        name="run",
        expression="run",
        binning=(1, 100000.0, 500000.0),
        x_title="Run number",
        discrete_x=True,
    )
    config.add_variable(
        name="met_covx",
        expression="MET.covXX",
        binning=(50, 0.0, 200.0),
        x_title=r"$\sigma_x$",
    )
    config.add_variable(
        name="met_covy",
        expression="MET.covYY",
        binning=(50, 0.0, 200.0),
        x_title=r"$\sigma_y$",
    )
    config.add_variable(
        name="lumi",
        expression="luminosityBlock",
        binning=(1, 0.0, 5000.0),
        x_title="Luminosity block",
        discrete_x=True,
    )
    config.add_variable(
        name="n_jet",
        expression="n_jet",
        binning=(11, -0.5, 10.5),
        x_title="Number of jets",
        discrete_x=True,
    )
    config.add_variable(
        name="n_hhbtag",
        expression="n_hhbtag",
        binning=(4, -0.5, 3.5),
        x_title="Number of HH b-tags",
        discrete_x=True,
    )
    config.add_variable(
        name="ht",
        binning=[0, 80, 120, 160, 200, 240, 280, 320, 400, 500, 600, 800],
        unit="GeV",
        x_title="HT",
    )
    config.add_variable(
        name="jet_pt",
        expression="Jet.pt",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"all Jet $p_{T}$",
    )
    config.add_variable(
        name="jet1_pt",
        expression="Jet.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 1 $p_{T}$",
    )
    config.add_variable(
        name="muon1_pt",
        expression="Muon.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 120.0),
        unit="GeV",
        x_title=r"Muon 1 $p_{T}$",
    )
    config.add_variable(
        name="met_pz",
        expression="MET.pz",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 120.0),
        unit="GeV",
        x_title=r"MET $p_{z}$",
    )
    config.add_variable(
        name="top_mass",
        expression="Top.mass",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$m_t$",
    )
    config.add_variable(
        name="w_mass",
        expression="mW",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 200.0),
        unit="GeV",
        x_title=r"$m_W$",
    )
    config.add_variable(
        name="w_mass_ana",
        expression="mW_ana",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 200.0),
        unit="GeV",
        x_title=r"$m_W$",
    )
    config.add_variable(
        name="w_mass_kin",
        expression="mW_kin",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 200.0),
        unit="GeV",
        x_title=r"$m_W$",
    )
    config.add_variable(
        name="w_mass2",
        expression="mW_corr",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 200.0),
        unit="GeV",
        x_title=r"$m_W$",
    )
    config.add_variable(
        name="w_mass_ana2",
        expression="mW_corr_ana",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 200.0),
        unit="GeV",
        x_title=r"$m_W$",
    )
    config.add_variable(
        name="w_mass_kin2",
        expression="mW_corr_kin",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 200.0),
        unit="GeV",
        x_title=r"$m_W$",
    )
    config.add_variable(
        name="muon_phi",
        expression="Muon.phi",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.15, 3.15),
        x_title=r"Muon $\phi$",
    )
    config.add_variable(
        name="muon2_pt",
        expression="Muon.pt[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 120.0),
        unit="GeV",
        x_title=r"Muon 2 $p_{T}$",
    )
    config.add_variable(
        name="m2mu",
        expression="m2mu",
        null_value=EMPTY_FLOAT,
        binning=(60, 60, 160),
        x_title=r"invariant mass $m_{\mu\mu}$",
    )
    config.add_variable(
        name="m1mu",
        expression="Muon.mass[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(60, 60, 120),
        x_title=r"invariant mass $m_{\mu}$",
    )
    config.add_variable(
        name="m2tau",
        expression="m2tau",
        null_value=EMPTY_FLOAT,
        binning=(30, 60, 300),
        x_title=r"invariant mass $m_{\tau\tau}$",
    )
    config.add_variable(
        name="mtW",
        expression="mT_W",
        null_value=EMPTY_FLOAT,
        binning=(100, 0.0, 150.0),
        unit="GeV",
        x_title=r"Transverse mass $m_{T}$",
    )
    config.add_variable(
        name="mtZ",
        expression="mT_Z",
        null_value=EMPTY_FLOAT,
        binning=(100, 40.0, 120.0),
        unit="GeV",
        x_title=r"Transverse mass $m_{T}$",
    )
    config.add_variable(
        name="m4mu",
        expression="m4mu",
        null_value=EMPTY_FLOAT,
        binning=(100, 0.0, 200.0),
        unit="GeV",
        x_title=r"$m_{4\mu}$",
    )
    config.add_variable(
        name="jet1_eta",
        expression="Jet.eta[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Jet 1 $\eta$",
    )
    config.add_variable(
        name="jet2_pt",
        expression="Jet.pt[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 2 $p_{T}$",
    )
    config.add_variable(
        name="met_phi",
        expression="MET.phi",
        null_value=EMPTY_FLOAT,
        binning=(33, -3.3, 3.3),
        x_title=r"MET $\phi$",
    )

    config.add_variable(
        name="e_pt",
        expression="Electron.pt",
        null_value=EMPTY_FLOAT,
        binning=(400, 0, 400),
        x_title=r"Electron p$_{T}$",
    )

    # weights
    config.add_variable(
        name="mc_weight",
        expression="mc_weight",
        binning=(200, -10, 10),
        x_title="MC weight",
    )
    config.add_variable(
        name="pu_weight",
        expression="pu_weight",
        binning=(40, 0, 2),
        x_title="Pileup weight",
    )
    config.add_variable(
        name="normalized_pu_weight",
        expression="normalized_pu_weight",
        binning=(40, 0, 2),
        x_title="Normalized pileup weight",
    )
    config.add_variable(
        name="btag_weight",
        expression="btag_weight",
        binning=(60, 0, 3),
        x_title="b-tag weight",
    )
    config.add_variable(
        name="normalized_btag_weight",
        expression="normalized_btag_weight",
        binning=(60, 0, 3),
        x_title="Normalized b-tag weight",
    )
    config.add_variable(
        name="normalized_njet_btag_weight",
        expression="normalized_njet_btag_weight",
        binning=(60, 0, 3),
        x_title="$N_{jet}$ normalized b-tag weight",
    )

    # cutflow variables
    config.add_variable(
        name="cf_njet",
        expression="cutflow.n_jet",
        binning=(17, -0.5, 16.5),
        x_title="Jet multiplicity",
        discrete_x=True,
    )
    config.add_variable(
        name="cf_ht",
        expression="cutflow.ht",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$H_{T}$",
    )
    config.add_variable(
        name="cf_jet1_pt",
        expression="cutflow.jet1_pt",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 1 $p_{T}$",
    )
    config.add_variable(
        name="cf_jet1_eta",
        expression="cutflow.jet1_eta",
        binning=(40, -5.0, 5.0),
        x_title=r"Jet 1 $\eta$",
    )
    config.add_variable(
        name="cf_jet1_phi",
        expression="cutflow.jet1_phi",
        binning=(32, -3.2, 3.2),
        x_title=r"Jet 1 $\phi$",
    )
    config.add_variable(
        name="cf_jet2_pt",
        expression="cutflow.jet2_pt",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 2 $p_{T}$",
    )

    # variables of interest
    config.add_variable(
        name="hh_mass",
        expression="hh.mass",
        binning=(20, 250, 750.0),
        unit="GeV",
        x_title=r"$m_{hh}$",
    )
    config.add_variable(
        name="hh_pt",
        expression="hh.pt",
        binning=(100, 0, 500.0),
        unit="GeV",
        x_title=r"$p_T$",
    )
    config.add_variable(
        name="hh_eta",
        expression="hh.eta",
        binning=(100, -3.0, 3.0),
        unit="GeV",
        x_title=r"$\eta$",
    )

    config.add_variable(
        name="ditau_mass",
        expression="diTau.mass",
        binning=(20, 50, 200.0),
        unit="GeV",
        x_title=r"$m_{\tau\tau}$",
    )
    config.add_variable(
        name="ditau_pt",
        expression="diTau.pt",
        binning=(100, 0, 500.0),
        unit="GeV",
        x_title=r"$p_T$",
    )
    config.add_variable(
        name="ditau_eta",
        expression="diTau.eta",
        binning=(100, -3.0, 3.0),
        unit="GeV",
        x_title=r"$\eta$",
    )

    config.add_variable(
        name="dibjet_mass",
        expression="diBJet.mass",
        binning=(20, 0, 500.0),
        unit="GeV",
        x_title=r"$m_{bb}$",
    )
    config.add_variable(
        name="dibjet_pt",
        expression="diBJet.pt",
        binning=(100, 0, 500.0),
        unit="GeV",
        x_title=r"$p_T$",
    )
    config.add_variable(
        name="dibjet_eta",
        expression="diBJet.eta",
        binning=(100, -3.0, 3.0),
        unit="GeV",
        x_title=r"$\eta$",
    )

    # outputs of the resonant pDNN at SM-like mass and spin values
    for proc in ["hh", "tt", "dy"]:
        config.add_variable(
            name=f"res_pdnn_{proc}",
            expression=f"res_pdnn_s0_m500_{proc}",
            binning=(25, 0.0, 1.0),
            x_title=rf"{proc.upper()} output node, res. pDNN$_{{m_{{HH}}=500\,GeV,s=0}}$",
        )
