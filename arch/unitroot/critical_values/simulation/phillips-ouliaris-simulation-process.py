from collections import defaultdict
import glob
import os

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS

from phillips_ouliaris import FILE_TYPES, ROOT, TRENDS

META = {"z_a": "negative", "z_t": "negative", "p_u": "positive", "p_z": "positive"}
CRITICAL_VALUES = (1, 5, 10)
# 1. Load data
# 2. Compute critical values


def xval(lhs, rhs, log=True, folds=5):
    lhs = np.asarray(lhs)
    rhs = np.asarray(rhs)
    pcg = np.random.PCG64(849756746597530743027509)
    gen = np.random.Generator(pcg)
    nobs = lhs.shape[0]
    idx = gen.permutation(nobs)
    predictions = np.empty((nobs, 6))
    for fold in range(folds):
        right = int(fold * nobs / folds)
        left = int((fold + 1) * nobs / folds)
        locs = idx[np.r_[np.arange(0, right), np.arange(left, nobs)]]
        end = nobs if fold == folds - 1 else left
        pred_loc = idx[np.arange(right, end)]
        pred_rhs = rhs[pred_loc]
        sm = OLS(lhs[locs], rhs[locs, :3]).fit()
        predictions[pred_loc, 0] = pred_rhs[:, :3] @ sm.params
        lg = OLS(lhs[locs], rhs[locs]).fit()
        predictions[pred_loc, 1] = pred_rhs @ lg.params
        if log and np.all(np.sign(lhs) == np.sign(lhs)[0]):
            log_lhs = np.log(np.abs(lhs))
            sgn = np.sign(lhs[0])
            sm_log = OLS(log_lhs[locs], rhs[locs, :3]).fit()
            sigma2 = (sm_log.resid ** 2).mean()
            predictions[pred_loc, 2] = sgn * np.exp(pred_rhs[:, :3] @ sm_log.params)
            predictions[pred_loc, 3] = sgn * np.exp(
                pred_rhs[:, :3] @ sm_log.params + sigma2 / 2
            )

            lg_log = OLS(log_lhs[locs], rhs[locs]).fit()
            sigma2 = (lg_log.resid ** 2).mean()
            predictions[pred_loc, 4] = sgn * np.exp(pred_rhs @ lg_log.params)
            predictions[pred_loc, 5] = sgn * np.exp(
                pred_rhs @ lg_log.params + sigma2 / 2
            )
    errors = lhs[:, None] - predictions
    best = np.argmin(errors.var(0))
    print(best)


def estimate_cv_regression(results, statistic):
    # For percentiles 1, 5 and 10, regress on a constant, and powers of 1/T
    out = {}
    quantiles = np.asarray(results.index)
    tau = np.array(results.columns).reshape((1, -1)).T
    rhs = (1.0 / tau) ** np.arange(4)
    for cv in CRITICAL_VALUES:
        if META[statistic] == "negative":
            loc = np.argmin(np.abs(100 * quantiles - cv))
        else:
            loc = np.argmin(np.abs(100 * quantiles - (100 - cv)))
        lhs = np.squeeze(np.asarray(results.iloc[loc]))
        xval(lhs, rhs)
        res = OLS(lhs, rhs).fit()
        params = res.params.copy()
        if res.pvalues[-1] > 0.05:
            params[-1] = 0.00
        out[cv] = [round(val, 5) for val in params]
    return out, tau.min()


results = defaultdict(list)
for file_type in FILE_TYPES:
    for trend in TRENDS:
        pattern = f"*-statistic-{file_type}-trend-{trend}*.hdf"
        print(pattern)
        result_files = glob.glob(os.path.join(ROOT, pattern))
        print(result_files)
        for rf in result_files:
            temp = pd.DataFrame(pd.read_hdf(rf, "results"))
            statistics = temp.columns.levels[2]
            for stat in statistics:
                single = temp.loc[:, pd.IndexSlice[:, :, stat]]
                single.columns = single.columns.droplevel(2)
                results[(stat, trend)].append(single)

joined = defaultdict(list)
for key in results:
    temp = results[key]
    stoch_trends = temp[0].columns.levels[1]
    quantiles = np.array(temp[0].index)
    for st in stoch_trends:
        for df in temp:
            single = df.loc[:, pd.IndexSlice[:, st]]
            single.columns = single.columns.droplevel(1)
            single = single.dropna(axis=1, how="all")
            joined[key + (st,)].append(single)

final = {key: pd.concat(joined[key], axis=1) for key in joined}
params = {}
for key in final:
    print(key)
    params[key] = estimate_cv_regression(final[key], key[0])
