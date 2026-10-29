import re
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from turban.variables import STANDARD_CTD_TYPE
from turban.utils.logging import get_logger

logger = get_logger(__name__)


# Standard names for sensors, only used for (deprecated) guessing of sensornames_ctd
# in MssDeviceConfig.from_mrd
mss_standard_ctd_sensornames: dict[STANDARD_CTD_TYPE, list[str]] = {}
mss_standard_ctd_sensornames["press"] = ["PRESS", "P250", "P1000"]
mss_standard_ctd_sensornames["temp"] = ["TEMP", "NTC"]
mss_standard_ctd_sensornames["cond"] = ["COND"]

# Shear sensors must be named SHE<n>
MSS_SENSORNAME_SHEAR = r"^SHE\d+$"


class MssSensor(BaseModel):
    name: str = Field(
        description="Sensor name. Sensors with a fixed role are accessed by these "
        "exact names: COUNT, PRESS, TEMP, NTC, NTCHP, ACC, ACCx, ACCy"
    )
    coefficients: list[float]
    channel: int
    unit: str = Field(default="")
    calibration_type: Literal[None]  # ["N", "SHE", "P", "SHH", "NFC", "V04", "N24"]


class MssSensorPoly(MssSensor):
    """Polynomial calibration with all coefficients."""

    calibration_type: Literal["N"] = Field(default="N")
    _p: np.polynomial.Polynomial = PrivateAttr()

    def model_post_init(self, context: Any, /) -> None:
        self._p = np.polynomial.Polynomial(self.coefficients)

    def raw_to_units(self, rawdata, offset=0):
        """Apply polynomial calibration to raw data.

        Parameters
        ----------
        rawdata : array_like
            Raw sensor counts.
        offset : float, optional
            Additive offset applied before evaluating the polynomial.

        Returns
        -------
        ndarray
            Calibrated data in physical units.
        """
        return self._p(rawdata + offset)


class MssShearSensor(MssSensor):
    """Shear sensor, the coefficients are derived from `sensitivity`."""

    name: Annotated[str, Field(pattern=MSS_SENSORNAME_SHEAR)]
    sensitivity: float
    serial_number: str = Field(default="")
    reference_temperature: float | None = None
    calibration_date: str = Field(default="")
    calibration_type: Literal["SHE"] = Field(default="SHE")
    _p: np.polynomial.Polynomial = PrivateAttr()

    def model_post_init(self, context: Any, /) -> None:
        self.coefficients = [
            1.47133e-6 / self.sensitivity,
            2.94266e-6 / self.sensitivity,
        ]
        self._p = np.polynomial.Polynomial(self.coefficients)

    def raw_to_units(self, rawdata, offset=0):
        """Apply shear sensor calibration to raw data.

        Parameters
        ----------
        rawdata : array_like
            Raw sensor counts.
        offset : float, optional
            Subtractive offset (shear sensors use negative offset convention).

        Returns
        -------
        ndarray
            Calibrated shear in physical units.
        """
        return self._p(rawdata - offset)  # The shear sensors have the negative offset


class MssSensorPressure(MssSensor):
    """Pressure, polynomial from all but the last coefficient, which is a zero offset."""

    calibration_type: Literal["P"] = Field(default="P")
    _p: np.polynomial.Polynomial = PrivateAttr()

    def model_post_init(self, context: Any, /) -> None:
        self._p = np.polynomial.Polynomial(self.coefficients[:-1])

    def raw_to_units(self, rawdata, offset=0):
        """Apply pressure calibration to raw data.

        Parameters
        ----------
        rawdata : array_like
            Raw sensor counts.
        offset : float, optional
            Additive offset applied before evaluating the polynomial.

        Returns
        -------
        ndarray
            Calibrated pressure; the last coefficient is subtracted as a zero-offset.
        """
        return self._p(rawdata + offset) - self.coefficients[-1]


class MssSensorNTC(MssSensor):
    """Steinhart/Hart NTC polynomial from all but the last coefficient."""

    calibration_type: Literal["SHH"] = Field(default="SHH")
    _p: np.polynomial.Polynomial = PrivateAttr()

    def model_post_init(self, context: Any, /) -> None:
        self._p = np.polynomial.Polynomial(self.coefficients[:-1])

    def raw_to_units(self, rawdata, offset=0):
        """Apply Steinhart-Hart NTC calibration to raw data.

        Parameters
        ----------
        rawdata : array_like
            Raw resistance counts.
        offset : float, optional
            Additive offset applied before taking the natural log.

        Returns
        -------
        ndarray
            Temperature in degrees Celsius.
        """
        data = self._p(np.log(rawdata + offset))
        return 1 / data - 273.15  # Kelvin to degC


class MssSensorTurb(MssSensor):
    """Turbidity (NFC), polynomial from all but the last two coefficients, which
    are an offset and a factor."""

    calibration_type: Literal["NFC"] = Field(default="NFC")
    _p: np.polynomial.Polynomial = PrivateAttr()

    def model_post_init(self, context: Any, /) -> None:
        self._p = np.polynomial.Polynomial(self.coefficients[:-2])

    def raw_to_units(self, rawdata, offset=0):
        """Apply turbidity (NFC) calibration to raw data.

        Parameters
        ----------
        rawdata : array_like
            Raw sensor counts.
        offset : float, optional
            Additive offset applied before polynomial evaluation.

        Returns
        -------
        ndarray
            Calibrated turbidity in physical units.
        """
        return self._p(rawdata + offset) * self.coefficients[-1] + self.coefficients[-2]


class MssSensorOptode(MssSensor):
    """Oxygen optode, raw to mV with coefficients[0:2], then mV to O2 with the zero
    point correction coefficients[-2:]."""

    calibration_type: Literal["V04"] = Field(default="V04")
    _p1: np.polynomial.Polynomial = PrivateAttr()
    _p2: np.polynomial.Polynomial = PrivateAttr()

    def model_post_init(self, context: Any, /) -> None:
        self._p1 = np.polynomial.Polynomial(self.coefficients[0:2])
        self._p2 = np.polynomial.Polynomial(self.coefficients[-2:])

    def raw_to_units(self, rawdata, offset=0):
        """Apply two-stage oxygen optode calibration to raw data.

        Parameters
        ----------
        rawdata : array_like
            Raw sensor counts.
        offset : float, optional
            Additive offset applied in the first conversion stage.

        Returns
        -------
        ndarray
            Calibrated oxygen concentration in physical units.
        """
        return self._p2(self._p1(rawdata + offset))


class MssSensorOptodeInternalTemp(MssSensor):
    """Internal temperature of the oxygen optode, polynomial with all coefficients."""

    calibration_type: Literal["N24"] = Field(default="N24")
    _p: np.polynomial.Polynomial = PrivateAttr()

    def model_post_init(self, context: Any, /) -> None:
        self._p = np.polynomial.Polynomial(self.coefficients)

    def raw_to_units(self, rawdata, offset=0):
        """Apply polynomial calibration to raw optode internal temperature data.

        Parameters
        ----------
        rawdata : array_like
            Raw sensor counts.
        offset : float, optional
            Additive offset applied before evaluating the polynomial.

        Returns
        -------
        ndarray
            Calibrated temperature in physical units.
        """
        return self._p(rawdata + offset)


# Sensor class per calibration type in the MRD header ("N" is MssShearSensor for SHE<n>)
mss_sensor_classes: dict[str, type[MssSensor]] = {
    "N": MssSensorPoly,
    "SHH": MssSensorNTC,
    "P": MssSensorPressure,
    "V04": MssSensorOptode,
    "N24": MssSensorOptodeInternalTemp,
    "NFC": MssSensorTurb,
}


class MssDeviceConfig(BaseModel):
    offset: int = Field(
        default=0, description="16bit offset, typically 0, older devices have -32768"
    )
    sampling_freq: float = Field(
        default=1024.0,
        description="The sampling frequency [Hz] of the microstructure probe",
    )
    gain_utemp: float = Field(
        default=1.5,
        description="Gain of the NTC highpass pre-emphasis differentiator",
    )
    sensors: dict[
        str,
        Annotated[
            MssSensor
            | MssSensorPoly
            | MssShearSensor
            | MssSensorPressure
            | MssSensorNTC
            | MssSensorTurb
            | MssSensorOptode
            | MssSensorOptodeInternalTemp,
            Field(discriminator="calibration_type"),
        ],
    ] = Field(
        default_factory=dict,
        description="A dictionary of the sensors mounted to the probe",
    )
    sensornames_ctd: dict[STANDARD_CTD_TYPE, str] = Field(
        default_factory=lambda: {"cond": "", "temp": "", "press": ""},
        description="A dictionary to link standard ctd names to the names of the MSS config",
    )
    pspd_rel_method: Literal["pressure", "constant", "external"] = Field(
        default="pressure",
        description="Method for the platform speed relative to the seawater, this is needed to calculate wavenumbers from the sampled data",
    )
    pspd_rel_constant_vel: float | None = Field(
        default=None,
        description='Constant velocity [m/s] used as pspd_rel, if defined by "pspd_rel_method"',
    )

    @model_validator(mode="after")
    def _check_sensornames(self) -> "MssDeviceConfig":
        for key, sensor in self.sensors.items():
            if key != sensor.name:
                raise ValueError(
                    f"Sensor key {key!r} differs from its name {sensor.name!r}"
                )
        for ctd_sensor, name in self.sensornames_ctd.items():
            if name and name not in self.sensors:
                raise ValueError(
                    f"sensornames_ctd[{ctd_sensor!r}] = {name!r} is not in sensors"
                )
        return self

    @classmethod
    def from_mrd(
        cls,
        filename: str | Path,
        shear_sensitivities: dict[str, float],
        sensornames_ctd: dict[STANDARD_CTD_TYPE, str] | None = None,
        offset: int = 0,
    ):
        """Create a MssDeviceConfig from an MRD file.

        Polynomial sensors named SHE<n> in the header become shear sensors.

        Parameters
        ----------
        filename : str or Path
            Path to the MRD file.
        shear_sensitivities : dict[str, float]
            Mapping of sensor name to sensitivity value for shear sensors.
        sensornames_ctd : dict[STANDARD_CTD_TYPE, str], optional
            Mapping of standard ctd names (see `turban.variables.STANDARD_CTD_TYPE`)
            to sensor names in the MRD header, e.g.
            ``{"cond": "COND", "temp": "TEMP", "press": "PRESS"}``. If None
            (deprecated), the sensor names are guessed from
            `mss_standard_ctd_sensornames`.
        offset : int, optional
            16-bit count offset, typically 0 or -32768 for older devices
            (default 0).

        Returns
        -------
        MssDeviceConfig
            Configuration object with sensors parsed from the MRD file header.
        """
        # local import, mss_mrd imports from this module
        from turban.instruments.mss import mss_mrd

        logger.debug("Reading header of %s", filename)
        with open(filename, "rb") as f:
            header_raw = mss_mrd.read_mrd(filestream=f, header_only=True)["header"]
        channels = mss_mrd.parse_header(header_raw)["mss"]["channels"]

        sensors: dict[str, MssSensor] = {}
        for ch, sensor_dict in channels.items():
            sensorname = sensor_dict["name"]
            caltype = sensor_dict["caltype"].upper()
            kwargs = dict(
                channel=ch,
                name=sensorname,
                coefficients=sensor_dict["coeff"],
                unit=sensor_dict["unit"],
            )
            logger.debug(
                "Channel %s, sensorname %s, caltype %s", ch, sensorname, caltype
            )
            if caltype == "N" and re.fullmatch(MSS_SENSORNAME_SHEAR, sensorname):
                sensors[sensorname] = MssShearSensor(
                    sensitivity=shear_sensitivities[sensorname], **kwargs
                )
            elif caltype in mss_sensor_classes:
                sensors[sensorname] = mss_sensor_classes[caltype](**kwargs)
            else:
                logger.info(
                    "Skipping sensor %s with unknown calibration type %s",
                    sensorname,
                    caltype,
                )

        sensornames = [c["name"] for c in channels.values()]
        if sensornames_ctd is not None:
            missing = set(sensornames_ctd.values()) - set(sensornames)
            if missing:
                raise ValueError(
                    f"Sensors {sorted(missing)} given in sensornames_ctd not found in "
                    f"MRD header of {filename}, available: {sensornames}"
                )
        else:
            logger.warning(
                "Deprecated: guessing sensornames_ctd from standard sensor names, "
                "pass sensornames_ctd explicitly."
            )
            sensornames_ctd = {"cond": "", "temp": "", "press": ""}
            sensornames_casefold = [s.casefold() for s in sensornames]
            for ctd_sensor in sensornames_ctd:
                for k in mss_standard_ctd_sensornames.get(ctd_sensor, []):
                    if k.casefold() in sensornames_casefold:
                        sensorname = sensornames[
                            sensornames_casefold.index(k.casefold())
                        ]
                        logger.debug(
                            "Found MSS sensor %s for %s", sensorname, ctd_sensor
                        )
                        sensornames_ctd[ctd_sensor] = sensorname
                        break

        return cls(offset=offset, sensors=sensors, sensornames_ctd=sensornames_ctd)

    @classmethod
    def from_prb(cls, filename, shear_sensitivities=None, offset=0):
        """Create a MssDeviceConfig from a PRB file.

        Parameters
        ----------
        filename : str or Path
            Path to the PRB file.
        shear_sensitivities : dict[str, float], optional
            Shear sensitivity values.
        offset : int, optional
            16-bit count offset (default 0).

        Raises
        ------
        NotImplementedError
            This method is not yet implemented.
        """
        raise NotImplementedError
