import numpy as np
import pytest

from turban.utils.seawater import viscosity_kinematic, viscosity_kinematic_full


def test_viscosity_kinematic_simple_vs_full():
    # Polynomial fit vs. full Sharqawy et al. (2010) at S=35 over the oceanic
    # temperature range; they agree within the +-1.5 % accuracy of the
    # correlation (slightly more near the freezing point)
    temp = np.linspace(-2, 35, 75)
    nu_poly = viscosity_kinematic(temp)
    nu_full = viscosity_kinematic_full(temp, 35.0)
    assert nu_full.shape == temp.shape
    np.testing.assert_allclose(nu_poly, nu_full, rtol=0.02)
    np.testing.assert_allclose(nu_poly[temp >= 0], nu_full[temp >= 0], rtol=0.015)


def test_viscosity_kinematic_full_salinity_deviation():
    # nu increases with salinity by 0.11-0.16 % per psu (largest in warm water),
    # so for S=29..34 the deviation from S=35 stays below 1 %, i.e. within the
    # accuracy of the correlation
    temp = np.linspace(-2, 35, 75)[:, np.newaxis]
    sal = np.arange(29, 35, dtype=float)
    nu = viscosity_kinematic_full(temp, sal)
    nu_35 = viscosity_kinematic_full(temp, 35.0)
    rel_dev = nu / nu_35 - 1
    assert rel_dev.shape == (75, 6)
    assert np.all(np.diff(nu, axis=1) > 0)
    np.testing.assert_array_less(np.abs(rel_dev), 0.01)
    dev_per_psu = rel_dev / (sal - 35)
    assert np.all((dev_per_psu > 0.0011) & (dev_per_psu < 0.0017))
    # using the S=35 polynomial in fresher water stays within 2 %
    nu_poly = viscosity_kinematic(temp)
    np.testing.assert_allclose(nu_poly * np.ones_like(nu), nu, rtol=0.02)
