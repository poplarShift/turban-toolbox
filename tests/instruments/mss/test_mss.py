import numpy as np
import matplotlib.pyplot as plt
import json
import pytest
from beartype.roar import BeartypeCallHintParamViolation
from pydantic import ValidationError

from turban.instruments.mss.api import mrd_to_level1
from turban.instruments.mss.config import MssDeviceConfig, MssShearSensor
from turban.instruments.mss.mss_mrd import (
    read_mrd,
    mrd_calibrate,
    count_to_time,
)
from turban.process.shear.api import ShearProcessing, ShearLevel1, ShearConfig
from turban.utils.filepaths import atomix_benchmark_baltic_mrd_fpath

SHEAR_SENSITIVITIES = {"SHE1": 3.90e-4, "SHE2": 4.05e-4}
SENSORNAMES_CTD = {"cond": "COND", "temp": "TEMP", "press": "PRESS"}


@pytest.fixture(scope="module")
def mss_conf_and_raw():
    mss_conf = MssDeviceConfig.from_mrd(
        filename=atomix_benchmark_baltic_mrd_fpath,
        shear_sensitivities=SHEAR_SENSITIVITIES,
        sensornames_ctd=SENSORNAMES_CTD,
    )
    with open(atomix_benchmark_baltic_mrd_fpath, "rb") as f:
        data_raw = read_mrd(f)
    return mss_conf, data_raw


def test_count_to_time():
    # regular counter, offset as from calibration with coefficients [32768, 1]
    count = np.arange(5) + 32768.0
    np.testing.assert_allclose(count_to_time(count, 2.0), np.arange(5) / 2.0)


def test_count_to_time_wraparound_and_gap(caplog):
    # wraps around at 65536 and two records (2, 3) are missing
    count = np.array([65534, 65535, 0, 1, 4, 5])
    with caplog.at_level("WARNING"):
        time = count_to_time(count, 1.0)
    np.testing.assert_allclose(time, [0, 1, 2, 3, 6, 7])
    assert "1 gaps, 2 data records are missing" in caplog.text


def test_count_to_time_duplicate(caplog):
    count = np.array([10, 11, 11, 12])
    with caplog.at_level("WARNING"):
        time = count_to_time(count, 1.0)
    np.testing.assert_allclose(time, [0, 1, 1, 2])
    assert "1 repeated values" in caplog.text


@pytest.mark.parametrize("missing", [None, "nan", "interpolate"])
def test_mrd_calibrate_missing(mss_conf_and_raw, missing):
    # SH2_0330.MRD has no missing records, so simulate a gap of two records
    mss_conf, data_raw = mss_conf_and_raw
    nrecords = data_raw["numsamples"]
    data_raw_gap = dict(
        data_raw,
        channels=np.delete(data_raw["channels"], [1000, 1001], axis=0),
        numsamples=nrecords - 2,
    )
    data = mrd_calibrate(mss_conf, data_raw_gap, missing=missing)

    assert np.issubdtype(data["time"].dtype, np.datetime64)
    dt = np.diff(data["time_count"].values) * mss_conf.sampling_freq
    press = data["PRESS"].values
    if missing is None:
        assert data.sizes["time"] == nrecords - 2
        np.testing.assert_allclose(np.unique(dt), [1, 3])
        assert not np.isnan(press).any()
    else:
        assert data.sizes["time"] == nrecords
        np.testing.assert_allclose(dt, 1)
        np.testing.assert_array_equal(np.flatnonzero(data["inserted"]), [1000, 1001])
        if missing == "nan":
            assert np.isnan(press).sum() == 2
            assert np.isnan(press[1000:1002]).all()
            # NaN must not spread through the filters of the derived quantities
            for v in ["PSPD_REL", "SHEAR", "utemp"]:
                isnan = np.isnan(data[v].values)
                assert isnan[..., 1000:1002].all()
                assert not isnan[..., :1000].any() and not isnan[..., 1002:].any()
        else:
            assert not np.isnan(press).any()
            np.testing.assert_allclose(
                press[1000:1002],
                np.interp([1000, 1001], [999, 1002], press[[999, 1002]]),
            )


def test_mrd_calibrate_missing_invalid(mss_conf_and_raw):
    mss_conf, data_raw = mss_conf_and_raw
    with pytest.raises(BeartypeCallHintParamViolation):
        mrd_calibrate(mss_conf, data_raw, missing="zero")


def test_shear_sensor_name():
    MssShearSensor(name="SHE3", coefficients=[0, 1], channel=1, sensitivity=4e-4)
    with pytest.raises(ValidationError):
        MssShearSensor(name="She3", coefficients=[0, 1], channel=1, sensitivity=4e-4)


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


def test_mrd_to_level1_without_mss_cfg(caplog):
    cfg = ShearConfig(
        sampfreq=1024.0,
        segment_length=2048,
        segment_overlap=1024,
        chunk_length=4 * 2048,
        chunk_overlap=1024,
    )
    with caplog.at_level("WARNING"):
        level1 = mrd_to_level1(
            atomix_benchmark_baltic_mrd_fpath,
            "shear",
            cfg,
            shear_sensitivities=SHEAR_SENSITIVITIES,
            sensornames_ctd=SENSORNAMES_CTD,
        )
    assert isinstance(level1, ShearLevel1)
    # section_number has to be set by the user
    assert (level1.section_number == 0).all()
    assert "set section_number before processing" in caplog.text


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
    data = mrd_calibrate(mss_conf, data_raw)
    assert "SHEAR" in data and "SHE1" not in data

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
