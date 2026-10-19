import pytest

from turban.instruments.mss.config import (
    MssDeviceConfig,
    MssSensorPoly,
    MssShearSensor,
)


def make_config(value: float) -> MssDeviceConfig:
    """Config with `value` in a list[float], a required float and an optional float."""
    return MssDeviceConfig(
        sensors={
            "PRESS": MssSensorPoly(name="PRESS", channel=1, coefficients=[value, 1.0]),
            "SHE1": MssShearSensor(
                name="SHE1", channel=2, coefficients=[], sensitivity=value
            ),
        },
        pspd_rel_method="constant",
        pspd_rel_constant_vel=value,
    )


@pytest.mark.parametrize("value", [4e-4, float("nan"), float("inf"), -float("inf")])
def test_json_roundtrip(value):
    cfg = make_config(value)
    dumped = cfg.model_dump_json()

    cfg2 = MssDeviceConfig.model_validate_json(dumped)

    # compare serialized forms, since NaN != NaN makes `cfg == cfg2` fail;
    # this also catches values silently lost to None (e.g. inf -> null -> None)
    assert cfg2.model_dump_json() == dumped
