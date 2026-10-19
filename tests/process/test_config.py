import xarray as xr

from turban.process.generic.config import SegmentConfig
from turban.process.shear.config import ShearConfig
from turban.utils.filepaths import (
    atomix_benchmark_baltic_fpath,
    atomix_benchmark_faroe_fpath,
)

def test_config():
    for path in [atomix_benchmark_baltic_fpath, atomix_benchmark_faroe_fpath]:
        for Config in [SegmentConfig, ShearConfig]:
            Config.from_atomix_netcdf(path)


def test_netcdf_roundtrip(tmp_path):
    """None cannot be stored as a netCDF attribute, so it must not be written, but be recovered upon loading."""
    cfg = ShearConfig(
        sampfreq=512.0,
        segment_length=1024,
        segment_overlap=512,
        chunk_length=4096,
        chunk_overlap=2048,
        freq_cutoff_antialias=None,
        freq_cutoff_corrupt=98.0,
    )
    ds = xr.Dataset()
    cfg.add_to_xarray(ds)
    assert "freq_cutoff_antialias" not in ds.attrs
    assert ds.attrs["freq_cutoff_corrupt"] == 98.0
    assert all(v is not None for v in ds.attrs.values())
    fpath = tmp_path / "config.nc"
    ds.to_netcdf(fpath)
    with xr.open_dataset(fpath) as ds_read:
        cfg_read = ShearConfig.from_xarray(ds_read)
    assert "freq_cutoff_antialias" not in ds_read.attrs
    assert cfg_read.freq_cutoff_antialias is None
    assert cfg_read == cfg
