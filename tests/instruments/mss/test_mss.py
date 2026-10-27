import numpy as np
import matplotlib.pyplot as plt
import json
import pytest

from turban.instruments.mss.api import mrd_to_level1
from turban.instruments.mss.config import MssDeviceConfig
from turban.instruments.mss.mss_mrd import read_mrd, raw_to_level0, level0_to_level1
from turban.process.shear.api import ShearProcessing, ShearLevel1, ShearConfig
from turban.utils.filepaths import atomix_benchmark_baltic_mrd_fpath

SHEAR_SENSITIVITIES = {"SHE1": 3.90e-4, "SHE2": 4.05e-4}
SENSORNAMES_CTD = {"cond": "COND", "temp": "TEMP", "press": "PRESS"}


def test_from_mrd_guessing_sensornames_ctd_deprecated(caplog):
    with caplog.at_level("WARNING"):
        mss_conf = MssDeviceConfig.from_mrd(
            filename=atomix_benchmark_baltic_mrd_fpath,
            shear_sensitivities=SHEAR_SENSITIVITIES,
        )
    assert "Deprecated" in caplog.text
    assert mss_conf.sensornames_ctd == SENSORNAMES_CTD


def test_from_mrd_unknown_sensornames_ctd():
    with pytest.raises(ValueError, match="P9999"):
        MssDeviceConfig.from_mrd(
            filename=atomix_benchmark_baltic_mrd_fpath,
            shear_sensitivities=SHEAR_SENSITIVITIES,
            sensornames_ctd={"cond": "COND", "temp": "TEMP", "press": "P9999"},
        )


def test_mrd_to_level1_without_mss_cfg():
    cfg = ShearConfig(
        sampfreq=1024.0,
        segment_length=2048,
        segment_overlap=1024,
        chunk_length=4 * 2048,
        chunk_overlap=1024,
    )
    level1 = mrd_to_level1(
        atomix_benchmark_baltic_mrd_fpath,
        "shear",
        cfg,
        shear_sensitivities=SHEAR_SENSITIVITIES,
        sensornames_ctd=SENSORNAMES_CTD,
    )
    assert isinstance(level1, ShearLevel1)


def test_mss():
    """Only checks loading/saving of config, metadata and data"""
    mss_conf = MssDeviceConfig.from_mrd(
        filename=atomix_benchmark_baltic_mrd_fpath,
        shear_sensitivities={
            "SHE1": 3.90e-4,
            "SHE2": 4.05e-4,
        },  # sensors 32 and 33 (MSS038)
        sensornames_ctd=SENSORNAMES_CTD,
        offset=0,
    )
    assert mss_conf.sensornames_ctd == SENSORNAMES_CTD

    with open(atomix_benchmark_baltic_mrd_fpath, "rb") as f:
        data_raw = read_mrd(f)
    data_level0 = raw_to_level0(mss_conf, data_raw)

    data_level1 = level0_to_level1(mss_conf, data_level0)

    # save an existing configuration
    with open(f"out/tests/instruments/mss/SH2_0330_config.json", "w") as f:
        f.write(mss_conf.model_dump_json(indent=4))

    # load back in again
    with open(f"out/tests/instruments/mss/SH2_0330_config.json", "r") as f:
        mss_conf2 = MssDeviceConfig.model_validate(json.load(f))

    assert mss_conf == mss_conf2

    # full MRD -> ShearLevel1 path via the api, including derived aux data
    cfg = ShearConfig(
        sampfreq=1024.0,
        segment_length=2048,
        segment_overlap=1024,
        chunk_length=4 * 2048,
        chunk_overlap=1024,
    )
    level1 = mrd_to_level1(atomix_benchmark_baltic_mrd_fpath, "shear", cfg, mss_conf)
    assert isinstance(level1, ShearLevel1)
