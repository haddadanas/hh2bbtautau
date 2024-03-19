# coding: utf-8

"""
Definition of variables.
"""

from os import name
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
        binning=(100, 0.0, 150.0),
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
