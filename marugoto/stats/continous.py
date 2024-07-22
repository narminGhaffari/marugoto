#!/usr/bin/env python3
"""Calculate statistics for deployments on continous targets."""

from pathlib import Path
import pandas as pd
from sklearn import metrics
import scipy.stats as st
import scipy
import numpy as np
import seaborn as sns
import os


__all__ = ['continous', 'aggregate_continous_stats',
           'continous_aggregated_']


score_labels = ['R^2', 'Pearsons R', 'p-value']

def bootstrap_resample(data, n=None):
    """Resample the data with replacement."""
    if n is None:
        n = len(data)
    resample_i = np.floor(np.random.rand(n) * len(data)).astype(int)
    return data.iloc[resample_i]

def bootstrap_statistics(data, target_label, pred_label, n_iterations=1000):
    """Compute bootstrap statistics for R², Pearson's r, and p-value."""
    boot_r2 = []
    boot_pearson_r = []
    boot_pval = []

    for _ in range(n_iterations):
        sample = bootstrap_resample(data)
        r_value, pval = scipy.stats.pearsonr(sample[target_label], sample[pred_label])
        slope, intercept, r_value_lrg, p_value, std_err = scipy.stats.linregress(sample[target_label], sample[pred_label])
        boot_r2.append(r_value_lrg**2)
        boot_pearson_r.append(r_value)
        boot_pval.append(pval)

    return np.array(boot_r2), np.array(boot_pearson_r), np.array(boot_pval)

def continous(preds_df: pd.DataFrame, target_label: str) -> pd.DataFrame:
    plot_pearsr_df = preds_df[[target_label, "pred"]]
    plot_pearsr_df[target_label] = pd.to_numeric(plot_pearsr_df[target_label], errors='coerce')
    plot_pearsr_df['pred'] = pd.to_numeric(plot_pearsr_df['pred'], errors='coerce')
    pears = scipy.stats.pearsonr(plot_pearsr_df[target_label], plot_pearsr_df['pred'])[0]
    pval = scipy.stats.pearsonr(plot_pearsr_df[target_label], plot_pearsr_df['pred'])[1]
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(plot_pearsr_df[target_label], plot_pearsr_df['pred'])
    # Correct approach to create the DataFrame
    stats_data = {
        'R^2': [np.round(r_value**2, 2)],
        'Pearsons R': [np.round(pears, 2)],
        'p-value': [np.round(pval, 7)]
    }

    # Create DataFrame from the dictionary
    stats_df = pd.DataFrame(stats_data)
    return stats_df


def aggregate_continous_stats(df):
    scores_df = df[score_labels]
    means, sems = scores_df.mean(), scores_df.sem()
    l, h = st.t.interval(confidence=.95, df=len(scores_df)-1, loc=means, scale=sems)
    stats_df = pd.DataFrame.from_dict({'mean': means, '95% conf': (h-l)/2}).transpose().unstack()
    return stats_df

def continous_aggregated_(*preds_csvs: str, outpath: str, target_label: str) -> None:

    outpath = Path(outpath)
    preds_dfs = {
        Path(p).parent.name: continous(
            pd.read_csv(p, dtype=str), target_label)
        for p in preds_csvs}
    preds_df = pd.concat(preds_dfs, ignore_index=True)
    preds_df.index.name = None
    preds_df.to_csv(outpath/f'{target_label}-continous-stats-individual.csv')

    patient_preds_df = []
    for p in preds_csvs:
        patient_preds_df.append(pd.read_csv(p))
    aggregated_df = pd.concat(patient_preds_df, ignore_index=True)

    #Plot Aggregated Correlation Plot
    aggregated_df.to_csv(outpath/f'{target_label}-patient-preds-aggregated.csv')


    boot_r2, boot_pearson_r, boot_pval = bootstrap_statistics(aggregated_df, target_label, "pred")

    r2_conf_int = np.percentile(boot_r2, [2.5, 97.5])
    pearson_r_conf_int = np.percentile(boot_pearson_r, [2.5, 97.5])
    pval_conf_int = np.percentile(boot_pval, [2.5, 97.5])

    print("95% Confidence Interval for R²:", r2_conf_int)
    print("95% Confidence Interval for Pearson's r:", pearson_r_conf_int)
    print("95% Confidence Interval for p-value:", pval_conf_int)

    # Plot the aggregated correlation plot with confidence intervals
    pears = scipy.stats.pearsonr(aggregated_df[target_label], aggregated_df['pred'])[0]
    pval = scipy.stats.pearsonr(aggregated_df[target_label], aggregated_df['pred'])[1]
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(aggregated_df[target_label], aggregated_df['pred'])

    ax = sns.lmplot(x=target_label, y='pred', data=aggregated_df)
    ax.set(title=f"Aggregated Correlation Plot\nR^2: {np.round(r_value**2, 2)} | Pearson's R: {np.round(pears, 2)} | p-value: {np.round(pval, 7)}\n"
                f"95% CI R²: [{r2_conf_int[0]:.2f}, {r2_conf_int[1]:.2f}] | 95% CI Pearson's R: [{pearson_r_conf_int[0]:.2f}, {pearson_r_conf_int[1]:.2f}] | 95% CI p-value: [{pval_conf_int[0]:.7f}, {pval_conf_int[1]:.7f}]")
    ax.savefig(outpath/"aggregated_correlation_plot.png")

if __name__ == '__main__':
    from fire import Fire
    Fire(continous_aggregated_)
