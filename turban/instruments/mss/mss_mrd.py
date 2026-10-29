import datetime
import math
import re
from typing import Literal

import gsw
import numpy as np
import xarray as xr

from turban.instruments.mss import mss_utils
from turban.instruments.mss.config import MssDeviceConfig, MssShearSensor
from turban.utils.logging import get_logger

logger = get_logger(__name__)

try:
    from dateparser import parse as parse_date
except ImportError:
    logger.error("dateparser not available, date_pc of the MRD header will be wrong")

    def parse_date(*argv, **kwarg):
        return datetime.datetime(1970, 1, 1)


def _decode_coordinate(bdegmin: bytes, bmindec: bytes) -> tuple[float, int]:
    """Decode a GPS coordinate of an MRD position packet.

    Returns the coordinate in decimal degrees and the raw value degrees * 100 +
    integer minutes.
    """
    raw = int.from_bytes(bdegmin, byteorder="little")
    sign = 1 if raw & 0x8000 else -1
    degmin = raw & 0x1FFF
    mindec = int.from_bytes(bmindec, byteorder="little")
    digits = int(math.log10(mindec)) + 1 if mindec > 0 else 0
    minutes = degmin % 100 + mindec / 10**digits
    return sign * (degmin // 100 + minutes / 60), degmin


def _packet_words(b: bytes) -> list[int]:
    """The 8 little endian words of a 17 byte data packet."""
    return [int.from_bytes(b[i : i + 2], byteorder="little") for i in range(1, 16, 2)]


def read_mrd(filestream, header_only=False, pos_time_only=False) -> dict:
    """Read a binary SST MRD (Microstructure Raw Data) file.

    The ASCII header is padded with 0x1A to a multiple of 17 bytes. It is followed
    by 17 byte data packets: packet type + 8 little endian words. Type 1 is a GPS
    time packet, type 3 a GPS position packet, types 7 and 8 hold the ADC channels
    0-7 and 8-15.

    Parameters
    ----------
    filestream : binary file object
        Opened MRD file.
    header_only : bool, optional
        Stop after the header.
    pos_time_only : bool, optional
        Stop after the first GPS time and position packets.

    Returns
    -------
    dict
        "header" (str), "end_of_header" (int), "valid_mrd" (bool),
        "date" (list of [packet number, datetime, number of ADC samples before the
        packet]), "gps" (list of [packet number, lon, lat, lon degmin, lat degmin,
        hour, minute, second]) and, unless header_only, "numsamples" (int) and
        "channels" (raw counts, numsamples x 16).
    """
    f = filestream

    n = 0  # Number of bytes
    npackets = 0  # Number of 17 byte data packets
    in_header = True
    header = b""
    have_time = False
    have_pos = False

    data = {"date": [], "gps": []}
    adc = {7: [], 8: []}
    nread = 4096
    btmp = f.read(nread)
    logger.info("Start reading file")
    while True:
        if have_time and have_pos and pos_time_only:
            break

        if len(btmp) < 17:
            btmp = btmp + f.read(nread)

        if in_header:
            b = btmp[0:1]
            btmp = btmp[1:]
            n += 1
        else:
            b = btmp[0:17]
            btmp = btmp[17:]
            n += 17
            npackets += 1

        if len(b) == 0:  # Nothing left anymore
            break

        if in_header:
            header += b
            # Header is filled up with 0x1A until it is dividable by 17
            if b == b"\x1a":
                data["end_of_header"] = n
                if n % 17 == 0:
                    data["valid_mrd"] = True
                    try:
                        data["header"] = header.decode("UTF-8")
                    except UnicodeDecodeError:
                        logger.info(
                            "UTF-8 did not work, trying to decode using Windows charset"
                        )
                        data["header"] = header.decode("Windows-1252")
                    in_header = False
                    if header_only:
                        return data
            continue

        if b[0] in adc:  # ADC channels 0-7 (type 7) or 8-15 (type 8)
            adc[b[0]].append(np.frombuffer(b[1:], dtype="<u2"))

        elif b[0] == 1 and not have_time:  # Time packet of gps
            words = _packet_words(b)
            date = datetime.datetime(
                words[0],  # year
                words[1],  # month
                words[2],  # day
                words[4],  # hour
                words[5],  # minute
                words[6],  # second
                tzinfo=datetime.timezone.utc,
            )
            data["date"].append([npackets, date, len(adc[7])])
            have_time = True

        elif b[0] == 3 and not have_pos:  # Position packet of gps
            latdec, lat = _decode_coordinate(b[1:3], b[3:5])
            londec, lon = _decode_coordinate(b[5:7], b[7:9])
            # Daytime
            words = _packet_words(b)
            hhmmss = words[6] + (words[4] & 0x000F) * 100000
            hour = hhmmss // 10000
            minute = hhmmss % 10000 // 100
            second = hhmmss % 100
            data["gps"].append(
                [npackets, londec, latdec, lon, lat, hour, minute, second]
            )
            have_pos = True

    numsamples = min(len(adc[7]), len(adc[8]))
    data["numsamples"] = numsamples
    data["channels"] = np.hstack(
        [np.asarray(adc[7][:numsamples]), np.asarray(adc[8][:numsamples])]
    )
    return data


def parse_header(header):
    """Parse an ASCII MRD file header into a configuration dictionary.

    Extracts metadata from the ASCII header of a Microstructure Raw Data (MRD)
    file, including ship, cruise, date, and channel calibration coefficients.

    Parameters
    ----------
    header : str
        ASCII header text from the MRD file.

    Returns
    -------
    dict
        Configuration dictionary with keys 'ship', 'cruise', 'date_pc', and
        'mss' (which contains a nested dict with channel information).
    """
    config = {"mss": {"channels": {}}}
    ind_ship = header.find("Ship    :   ") + len("Ship    :   ")
    config["ship"] = header[ind_ship : ind_ship + 8].rstrip("_")

    ind_cruise = header.find("Cruise:") + len("Cruise:")
    config["cruise"] = header[ind_cruise + 1 : ind_cruise + 18].rstrip("_")

    lines = header.splitlines()
    date_pc = parse_date(lines[2]) if len(lines) > 2 else None
    if date_pc is None:
        logger.warning("Could not infer date_pc from the third header line")
    else:
        config["date_pc"] = date_pc.isoformat()

    for line in lines:
        # Sensor lines: device number, device name, channel, calibration type, name,
        # unit and 6 coefficients
        fields = re.sub(r"\s+", " ", line).split(" ")
        if len(fields) < 12:
            continue
        try:
            devicenum = int(fields[0])
            channelnum = int(fields[2])
        except ValueError:
            continue

        devicename = fields[1]
        if "MSS" in devicename:  # Only the MSS is treated here
            config["mss"]["name"] = devicename
            config["mss"]["devicenum"] = devicenum
            config["mss"]["channels"][channelnum] = {
                "unit": fields[5],
                "name": fields[4],
                "caltype": fields[3],
                "coeff": np.asarray([float(val) for val in fields[6:12]]),
            }

    return config


def count_to_time(count, sampling_freq: float) -> np.ndarray:
    """Convert the MSS sample counter into time since the first sample.

    The counter is a 16 bit integer that increases by one per sample and wraps
    around at 65536. Missing data records show up as increments larger than one
    and are accounted for in the returned time. Gaps of 65536 samples or more
    cannot be detected.

    Parameters
    ----------
    count : array_like
        Sample counter (raw or calibrated with slope 1, i.e. only offset).
    sampling_freq : float
        Sampling frequency in Hz.

    Returns
    -------
    ndarray
        Time since the first sample in s.
    """
    count = np.round(np.asarray(count)).astype(np.int64)
    dcount = np.diff(count) % 2**16
    nmissing = int(np.sum(dcount[dcount > 1] - 1))
    if nmissing > 0:
        logger.warning(
            "COUNT has %d gaps, %d data records are missing",
            np.sum(dcount > 1),
            nmissing,
        )
    if np.any(dcount == 0):
        logger.warning(
            "COUNT has %d repeated values (duplicate data records)",
            np.sum(dcount == 0),
        )
    return np.concatenate([[0], np.cumsum(dcount)]) / sampling_freq


def _add_derived(
    mss_config: MssDeviceConfig, ds: xr.Dataset, pspd_rel: np.ndarray | None
) -> xr.Dataset:
    """Add PSAL, DENS, PSPD_REL, SHEAR and utemp to the calibrated sensors (dim
    "index"), and drop the shear sensor channels. All input must be free of NaN at
    gaps, as the filters spread NaN."""
    fs = mss_config.sampling_freq
    nsamples = ds.sizes["index"]

    # Practical salinity and density
    cond = np.clip(ds[mss_config.sensornames_ctd["cond"]].values, 0, None)
    temp = ds[mss_config.sensornames_ctd["temp"]].values
    press = ds[mss_config.sensornames_ctd["press"]].values
    SP = gsw.SP_from_C(cond, temp, press)
    # Density from reference salinity, which (unlike absolute salinity) needs no
    # position
    SR = gsw.SR_from_SP(SP)
    dens = gsw.rho(SR, gsw.CT_from_t(SR, temp, press), press)
    ds["PSAL"] = (["index"], SP, {"units": "1"})
    ds["DENS"] = (["index"], dens, {"units": "kg m-3"})

    # Platform speed relative to the water
    if mss_config.pspd_rel_method == "constant":
        logger.debug("Using constant velocity %s", mss_config.pspd_rel_constant_vel)
        vsink = np.full(nsamples, mss_config.pspd_rel_constant_vel, dtype=float)
    elif mss_config.pspd_rel_method == "external" and pspd_rel is not None:
        logger.debug("Using external velocity")
        vsink = np.asarray(pspd_rel, dtype=float)
    else:
        if mss_config.pspd_rel_method == "external":
            logger.warning("No external pspd_rel given, using change of pressure")
        logger.debug("Using change of pressure to calculate velocity")
        vsink = mss_utils.calc_vsink(press=press, fs=fs)
    ds["PSPD_REL"] = (["index"], vsink, {"units": "m s-1"})

    # Shear of all shear sensors SHE<n>, sorted by n
    shearsensors = sorted(
        (
            name
            for name, sensor in mss_config.sensors.items()
            if isinstance(sensor, MssShearSensor)
        ),
        key=lambda name: int(name[3:]),
    )
    if shearsensors:
        shear = np.asarray(
            [
                mss_utils.calc_shear(ds[k].values, vsink, dens, fs=fs)
                for k in shearsensors
            ]
        )
        shear[np.isinf(shear)] = np.nan
    else:  # no shear data exists, dummy dimension of NaN shear values
        shear = np.full((1, nsamples), np.nan)
    ds = ds.drop_vars(shearsensors).assign_coords(n_shear=np.arange(len(shear)))
    ds["SHEAR"] = (["n_shear", "index"], shear, {"units": "s-1"})

    if "NTC" in ds and "NTCHP" in ds:
        utemp = mss_utils.deconvolve_mss_ntchp(
            x=ds["NTC"].values,
            x_emph=ds["NTCHP"].values,
            sampfreq=fs,
            gain=mss_config.gain_utemp,
        )
        ds["utemp"] = (["n_temp", "index"], utemp[np.newaxis, :])
    else:
        logger.info("No NTC and NTCHP channels, cannot derive utemp")

    return ds


def mrd_calibrate(
    mss_config: MssDeviceConfig,
    rawdata: dict,
    missing: Literal["nan", "interpolate"] | None = None,
    pspd_rel: np.ndarray | None = None,
) -> xr.Dataset:
    """Convert raw MRD data to physical units and derived quantities on a time axis.

    The calibration is taken from `mss_config` only, the MRD header is not used.

    Parameters
    ----------
    mss_config : MssDeviceConfig
        Device configuration.
    rawdata : dict
        Raw data from `read_mrd`.
    missing : {"nan", "interpolate"} or None, optional
        Handling of missing data records detected from the COUNT channel. None
        (default) keeps the records as they are, so that time is irregular at gaps.
        "nan" inserts time steps filled with NaN, "interpolate" inserts linearly
        interpolated time steps. Both also drop duplicate records and add the
        boolean variable "inserted".
    pspd_rel : ndarray, optional
        External platform speed on the output time axis, used if
        `mss_config.pspd_rel_method` is "external".

    Returns
    -------
    xr.Dataset
        All non-shear sensors (named as in `mss_config.sensors`), PSAL, DENS,
        PSPD_REL, SHEAR (n_shear, time), utemp (n_temp, time; only with NTC and
        NTCHP) and time_count, with dimension "time". The time coordinate is the
        physical time (UTC) if the file has a GPS time packet, else time since the
        first sample in s. Attributes longitude, latitude (NaN without GPS
        position) and date_gps (first time packet, if any).
    """
    if rawdata.get("numsamples", 0) == 0:
        raise ValueError("No samples found for conversion.")
    channels = rawdata["channels"]
    fs = mss_config.sampling_freq
    ds = xr.Dataset()

    for sensorname, sensor in mss_config.sensors.items():
        logger.debug("Converting %s", sensorname)
        # Channels are numbered from 1; float avoids uint16 overflow with the offset
        counts = channels[:, sensor.channel - 1].astype(np.float64)
        ds[sensorname] = (["index"], sensor.raw_to_units(counts, mss_config.offset))
        ds[sensorname].attrs["units"] = sensor.unit
        if isinstance(sensor, MssShearSensor):
            ds[sensorname].attrs["sensitivity"] = sensor.sensitivity

    # Time of each record since the first sample from the COUNT channel, accounting
    # for missing records
    if "COUNT" in ds:
        time_record = count_to_time(ds["COUNT"].values, fs)
    else:
        logger.warning("No COUNT channel found, time assumes no missing data records")
        time_record = np.arange(len(channels)) / fs
    sample = np.round(time_record * fs).astype(int)

    # Optionally insert time steps for missing records (dropping duplicate records).
    # They are interpolated also for "nan", as the filters of the derived quantities
    # spread NaN; they are set to NaN afterwards.
    inserted = None
    if missing is not None:
        _, keep = np.unique(sample, return_index=True)
        ds = ds.isel(index=keep).assign_coords(index=sample[keep])
        sample_all = np.arange(sample[-1] + 1)
        ds = ds.interp(index=sample_all, method="linear")
        inserted = ~np.isin(sample_all, sample[keep])
        sample = sample_all

    ds = _add_derived(mss_config, ds, pspd_rel)

    if inserted is not None:
        if missing == "nan":
            with xr.set_options(keep_attrs=True):
                ds = ds.where(xr.DataArray(~inserted, dims="index"))
        ds["inserted"] = (["index"], inserted)
        ds["inserted"].attrs["description"] = "record inserted for missing data"
    time_count = sample / fs
    ds["time_count"] = (["index"], time_count)
    ds["time_count"].attrs = {"units": "s", "description": "time since first sample"}

    # Physical time: start time of the file (time packet, 1 s resolution) + time_count
    gps_date = rawdata.get("date")
    if gps_date:
        date, isample = gps_date[0][1], gps_date[0][2]
        ds.attrs["date_gps"] = date.isoformat()
        # The time packet belongs to the record following it
        isample = min(isample, len(time_record) - 1)
        t0 = date.astimezone(datetime.timezone.utc).replace(
            tzinfo=None
        ) - datetime.timedelta(seconds=float(time_record[isample]))
        time = np.datetime64(t0, "ns") + np.round(time_count * 1e9).astype(
            "timedelta64[ns]"
        )
        time_attrs = {
            "description": "UTC, start time from MRD time packet "
            "(1 s resolution) plus time since first sample"
        }
    else:
        logger.warning("No time packet available, physical time cannot be derived.")
        time = time_count
        time_attrs = {"units": "s", "description": "time since first sample"}

    gps = rawdata.get("gps")
    if gps:
        ds.attrs["longitude"] = gps[0][1]
        ds.attrs["latitude"] = gps[0][2]
    else:
        logger.info("No GPS data available.")
        ds.attrs["longitude"] = np.nan
        ds.attrs["latitude"] = np.nan

    return (
        ds.drop_vars("index", errors="ignore")
        .rename_dims(index="time")
        .assign_coords(time=(["time"], time, time_attrs))
    )
