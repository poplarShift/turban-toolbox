import numpy as np
import pytest

from turban.process.generic.api import Level2, Level3
from turban.process.generic.config import SegmentConfig
from turban.utils.util import agg_fast_to_slow

CHUNK_LENGTH = 100
CHUNK_OVERLAP = 50
N = 1000


def make_level2():
    cfg = SegmentConfig(
        sampfreq=100.0,
        segment_length=50,
        segment_overlap=25,
        chunk_length=CHUNK_LENGTH,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return Level2(
        time=np.arange(N, dtype=float),
        senspeed=np.ones(N),
        section_number=np.ones(N, dtype=int),
        cfg=cfg,
    )


def expected_agg(arr, agg_method):
    return agg_fast_to_slow(
        arr,
        section_number_or_data_len=np.ones(N, dtype=int),
        chunk_length=CHUNK_LENGTH,
        chunk_overlap=CHUNK_OVERLAP,
        agg_method=agg_method,
    )


@pytest.mark.parametrize(
    "agg_method, name_out, expected_name",
    [
        ("mean", "foo", "foo"),
        ("median", "foo", "foo"),
        ("mean", None, "pres_mean"),
        ("median", None, "pres_median"),
    ],
)
def test_add_aux_data_name_out_level2_to_level3(agg_method, name_out, expected_name):
    pres = np.linspace(0.0, 100.0, N)
    level2 = make_level2()
    level2.add_aux_data(pres, name="pres", agg_method=agg_method, name_out=name_out)

    aux3 = Level3._from_level_below_kwarg(level2)["_aux_data"]

    assert list(aux3) == [expected_name]
    dims, arr = aux3[expected_name]
    assert dims == ["time"]
    np.testing.assert_allclose(arr, expected_agg(pres, agg_method))
