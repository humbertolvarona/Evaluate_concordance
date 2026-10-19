import numpy as np
import scipy.stats as stats
from sklearn.metrics import mean_squared_error
import statsmodels.stats.weightstats as smw

def evaluate_concordance(observed, simulated, file_name="concordance_report.txt"):
    observed = np.array(observed)
    simulated = np.array(simulated)
    n = len(observed)
    u_stat, u_p = stats.mannwhitneyu(observed, simulated, alternative='two-sided')
    mrd = np.median(np.abs(simulated - observed) / np.abs(observed))
    diff = simulated - observed
    n_boot = 1000
    boot_diff = []
    for _ in range(n_boot):
        sample = np.random.choice(diff, size=n, replace=True)
        boot_diff.append(np.mean(sample))
    boot_ci_low = np.percentile(boot_diff, 2.5)
    boot_ci_high = np.percentile(boot_diff, 97.5)
    rmse = np.sqrt(mean_squared_error(observed, simulated))
    obs_range = np.max(observed) - np.min(observed)
    nrmse = rmse / obs_range if obs_range != 0 else np.nan
    obs_mean = np.mean(observed)
    si = rmse / obs_mean if obs_mean != 0 else np.nan
    spearman_rho, spearman_p = stats.spearmanr(observed, simulated)
    kendall_tau, kendall_p = stats.kendalltau(observed, simulated)
    bound = 0.1 * obs_mean
    tost_p, tost_t1, tost_t2 = smw.ttost_paired(observed, simulated, -bound, bound)
    bias = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    loa_lower = bias - 1.96 * std_diff
    loa_upper = bias + 1.96 * std_diff
    pearson_r = np.corrcoef(observed, simulated)[0, 1]
    var_obs = np.var(observed)
    var_sim = np.var(simulated)
    ccc = (2 * pearson_r * np.std(observed) * np.std(simulated)) / (var_obs + var_sim + (obs_mean - np.mean(simulated))**2)
    lines = []
    lines.append("Concordance and Agreement Statistical Report")
    lines.append("=" * 50)
    lines.append(f"Sample Size: {n}")
    lines.append("")
    lines.append("1. Mann-Whitney U Test")
    lines.append("Parameters: alternative='two-sided'")
    lines.append(f"U-statistic: {u_stat:.4f}")
    lines.append(f"p-value: {u_p:.4e}")
    lines.append("")
    lines.append("2. Median Relative Difference")
    lines.append("Parameters: absolute relative difference median")
    lines.append(f"MRD: {mrd:.4f}")
    lines.append("")
    lines.append("3. Bootstrap Confidence Interval")
    lines.append("Parameters: iterations=1000, alpha=0.05, metric=mean difference")
    lines.append(f"95% CI: [{boot_ci_low:.4f}, {boot_ci_high:.4f}]")
    lines.append("")
    lines.append("4. Normalized Root Mean Square Error")
    lines.append("Parameters: normalized by range max-min")
    lines.append(f"NRMSE: {nrmse:.4f}")
    lines.append("")
    lines.append("5. Scatter Index")
    lines.append("Parameters: RMSE normalized by observed mean")
    lines.append(f"SI: {si:.4f}")
    lines.append("")
    lines.append("6. Spearman Rank Correlation")
    lines.append("Parameters: two-sided")
    lines.append(f"Rho: {spearman_rho:.4f}")
    lines.append(f"p-value: {spearman_p:.4e}")
    lines.append("")
    lines.append("7. Kendall Tau Correlation")
    lines.append("Parameters: two-sided")
    lines.append(f"Tau: {kendall_tau:.4f}")
    lines.append(f"p-value: {kendall_p:.4e}")
    lines.append("")
    lines.append("8. TOST Equivalence Test")
    lines.append(f"Parameters: paired, bounds +/- 10% of mean ({bound:.4f})")
    lines.append(f"p-value: {tost_p:.4e}")
    lines.append("")
    lines.append("9. Bland-Altman Analysis")
    lines.append("Parameters: 95% limits of agreement")
    lines.append(f"Bias: {bias:.4f}")
    lines.append(f"Lower Limit: {loa_lower:.4f}")
    lines.append(f"Upper Limit: {loa_upper:.4f}")
    lines.append("")
    lines.append("10. Lin Concordance Correlation Coefficient")
    lines.append("Parameters: sample estimates")
    lines.append(f"CCC: {ccc:.4f}")
    with open(file_name, 'w') as f:
        f.write(chr(10).join(lines))
    return file_name
