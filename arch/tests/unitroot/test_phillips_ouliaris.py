import pytest

from arch.unitroot.cointegration import phillips_ouliaris


@pytest.mark.parametrize("trend", ["n", "c", "ct", "ctt"])
@pytest.mark.parametrize("test_type", ["Za", "Zt", "Pu", "Pz"])
@pytest.mark.parametrize("kernel", ["bartlett", "parzen", "quadratic-spectral"])
@pytest.mark.parametrize("bandwidth", [None, 10])
@pytest.mark.parametrize("force_int", [True, False])
def test_smoke(trivariate_data, trend, test_type, kernel, bandwidth, force_int):
    y, x = trivariate_data
    res = phillips_ouliaris(
        y,
        x,
        trend=trend,
        test_type=test_type,
        kernel=kernel,
        bandwidth=bandwidth,
        force_int=force_int,
    )
    assert isinstance(res.stat, float)


def test_errors(trivariate_data):
    y, x = trivariate_data
    with pytest.raises(ValueError, match="kernel is not a known estimator."):
        phillips_ouliaris(y, x, kernel="fancy-kernel")
    with pytest.raises(ValueError, match="Unknown test_type: z-alpha."):
        phillips_ouliaris(y, x, test_type="z-alpha")
