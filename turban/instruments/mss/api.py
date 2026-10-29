from typing import cast, Literal
from pathlib import Path
import numpy as np

from turban.process.utemp.api import UTempLevel1
from turban.process.utemp.config import UTempConfig

from turban.instruments.mss import mss_mrd

from turban.instruments.mss.config import MssDeviceConfig
from turban.process.shear.config import ShearConfig
from turban.process.shear.api import ShearLevel1
from turban.utils.logging import get_logger
from turban.variables import STANDARD_CTD_TYPE

logger = get_logger(__name__)

# Sensors added as aux data under their own name, if present
# TODO: handle other sensors (Chl-a etc)
AUX_SENSORNAMES = ["ACC", "Chl_A", "ACCx", "ACCy"]


def mrd_to_level1(
    fname: str | Path,
    target: Literal["shear", "utemp"],
    proc_cfg: ShearConfig | UTempConfig,
    mss_cfg: MssDeviceConfig | None = None,
    shear_sensitivities: dict[str, float] | None = None,
    sensornames_ctd: dict[STANDARD_CTD_TYPE, str] | None = None,
    missing: Literal["nan", "interpolate"] | None = None,
) -> ShearLevel1 | UTempLevel1:
    """
    Read a binary .MRD file and convert to level1.

    If no MssDeviceConfig object is given, will read from MRD header. In this case,
    shear_sensitivities must not be None, and sensornames_ctd should map the
    standard ctd names to the MRD sensor names (guessing them when it is None is
    deprecated). If mss_cfg is given, it takes precedence and sensornames_ctd is
    ignored.

    Missing data records (detected from the COUNT channel) are handled according
    to `missing`, see `mss_mrd.mrd_calibrate`.

    section_number is set to 0 (discard) everywhere and has to be set by the user
    before processing.
    """
    if mss_cfg is None:
        if shear_sensitivities is None:
            raise ValueError("shear_sensitivities is required if mss_cfg is not given")
        mss_cfg = MssDeviceConfig.from_mrd(
            filename=fname,
            shear_sensitivities=shear_sensitivities,
            sensornames_ctd=sensornames_ctd,
        )
    elif sensornames_ctd is not None:
        logger.warning(
            "mss_cfg given, ignoring sensornames_ctd; using %s", mss_cfg.sensornames_ctd
        )

    with open(fname, "rb") as f:
        data_raw = mss_mrd.read_mrd(f)
    data = mss_mrd.mrd_calibrate(mss_cfg, data_raw, missing=missing)

    # physical time if the MRD file contains a time packet, else seconds
    time = data["time"].values

    # Sections are user-defined and should not be set here. All 0 (discard) for now,
    # setting them will be removed from mrd_to_level1.
    logger.warning(
        "Deprecated: mrd_to_level1 sets section_number to 0 (discard) everywhere, "
        "set section_number before processing."
    )
    section_number = np.zeros(len(time), dtype=int)

    if target == "shear":
        level1 = ShearLevel1(
            time=time,
            senspeed=data["PSPD_REL"].values,
            shear=data["SHEAR"].values,
            section_number=section_number,
            cfg=cast(ShearConfig, proc_cfg),
        )

    elif target == "utemp":
        if "utemp" not in data:
            raise ValueError("utemp needs the sensors NTC and NTCHP")
        level1 = UTempLevel1(
            time=time,
            senspeed=data["PSPD_REL"].values,
            dtempdt=data["utemp"].values,
            section_number=section_number,
            cfg=cast(UTempConfig, proc_cfg),
        )

    for k, v in mss_cfg.sensornames_ctd.items():
        level1.add_aux_data(data[v].values, name=k, name_out=k)

    absent = [v for v in AUX_SENSORNAMES if v not in data]
    if absent:
        logger.info("Sensors %s not present in data", absent)
    for v in AUX_SENSORNAMES:
        if v in data:
            level1.add_aux_data(data[v].values, name=v, name_out=v)

    return level1
