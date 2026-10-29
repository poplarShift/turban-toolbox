# TODO — MSS MRD reader cleanup

Scope: `turban/instruments/mss/api.py`, `mss_mrd.py`, `config.py` (context: `mss_utils.py`).

## ⚠ GUIDING PRINCIPLE: CHANGE THE HIGH-LEVEL API AS LITTLE AS POSSIBLE

- **`mrd_to_level1(...)` keeps its exact signature, defaults and return type.**
- **`MssDeviceConfig` (fields, `from_mrd(...)` signature) and the sensor classes keep
  their public interface.**
- **Tests should change minimally.** If a test has to change, list the reason here.

## Current structure

```
api.mrd_to_level1(...)               # signature unchanged; builds Shear/UTempLevel1
  ├─ MssDeviceConfig.from_mrd()      # only if mss_cfg is None (header -> calibration)
  ├─ mss_mrd.read_mrd(f)             # binary decode
  └─ mss_mrd.mrd_calibrate(cfg, raw, missing, pspd_rel) -> one xr.Dataset
        calibration (cfg only, no header) -> gaps (interpolated) -> time
        -> PSAL, DENS, PSPD_REL, SHEAR, utemp -> NaN at inserted records ("nan")
```

`raw_to_level0`/`level0_to_level1` are removed. Gap handling: see `HANDOVER_gap_filling.md`.

Test and doc changes (necessary, because those functions were removed):
- `tests/instruments/mss/test_mss.py`: `raw_to_level0`/`level0_to_level1` replaced by
  `mrd_calibrate`. The gap test now deletes raw records instead of level0 records.
  New checks: `inserted`, no NaN spread, `section_number` all 0 plus the warning,
  shear sensor name.
- `docs/mss.md`: the example uses `mrd_to_level1` and then sets `section_number`.
  Note: time is now physical time (datetime64) instead of `time_count`.

## Open

- [ ] **api.py TODO**: handle other sensors (Chl-a etc.). Decision so far: no generic
  "all sensors as aux". The explicit list `AUX_SENSORNAMES` stays.
- [ ] **section_number**: `mrd_to_level1` sets all 0 (discard) and logs a deprecation
  warning. Next step: stop setting it in `mrd_to_level1` altogether.
  `process.py` in tt_mss (outside this repo) must set sections now, otherwise it
  gets no estimates.
- [ ] **molvisc**: we don't want molvisc as an aux variable (deprecated). The unused
  `viscosity_kinematic` import is removed. Follow up on where molvisc aux is still
  used or expected (L4).
- [ ] **Reserved sensor names** (COUNT, PRESS, TEMP, NTC, NTCHP, ACC, ACCx, ACCy) are
  only documented (`MssSensor.name` description), not enforced: those sensors are
  all `MssSensorPoly`, so a type hint can't constrain them. A misspelled name is
  treated as absent. Only shear sensors are enforced via the type hint
  (`name: Annotated[str, Field(pattern=r"^SHE\d+$")]`).
- [ ] `UTempLevel1(dtempdt=utemp)`: utemp is the deconvolved temperature, not dT/dt.
  Leave as is for now.
- [ ] Not yet checked: uses of `raw_to_level0`/`level0_to_level1` elsewhere in
  `turban/` (e.g. `mss_hhl.py`), and the full test suite (`python -m pytest tests/ docs/`).
  Only `tests/instruments/mss` and `docs/mss.md` were run.
- [ ] `parse_header`: ship/cruise parsing uses `find()` without checking for -1
  (garbage if the field is missing). Unchanged.
