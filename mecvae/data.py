import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

def simulate_brain_measures(n_samples, n_features, n_batches, random_state=None, sex_proportion=0.5,
                             age_mean=40, age_std=10, scanner_mean=0, scanner_std=0.05):
    """Simulate synthetic FreeSurfer brain measures with batch effects and covariates (sex, age, scanner) following the ComBat model.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    n_features : int
        Number of brain measure features to generate.
    n_batches : int
        Number of batches in the data.
    sex_proportion : float, optional, default: 0.5
        Proportion of samples with sex equal to 1.
    age_mean : float, optional, default: 40
        Mean of the age distribution.
    age_std : float, optional, default: 10
        Standard deviation of the age distribution.
    scanner_mean : float, optional, default: 0
        Mean of the scanner distribution.
    scanner_std : float, optional, default: 0.05
        Standard deviation of the scanner distribution.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing the simulated brain measures and covariates (sex, age, scanner, batch).

    """
    # Set random seed
    rng = np.random.default_rng(random_state)

    # Simulate sex, age, and scanner covariates
    sex = rng.choice([0, 1], n_samples, p=[1 - sex_proportion, sex_proportion])
    age = rng.normal(age_mean, age_std, n_samples)
    scanner = rng.normal(scanner_mean, scanner_std, n_samples)

    # Simulate batch effects
    batch = rng.choice(np.arange(n_batches), n_samples)

    # Initialize brain measures with random values
    brain_measures = rng.random((n_samples, n_features))
    brain_measures_unbiased = brain_measures.copy()

    # Apply batch effects
    for b in range(n_batches):
        batch_indices = np.where(batch == b)
        batch_size = len(batch_indices[0])

        # Batch effect scaling factors
        scaling_factors = 1 + rng.normal(0, 0.1, (batch_size, n_features))

        # Apply scaling factors to brain measures within the batch
        brain_measures[batch_indices] *= scaling_factors

    # Apply covariate effects
    for i in range(n_features):
        # Randomly set the effect of each covariate on the brain measure
        sex_effect = rng.normal(0, 0.1)
        age_effect = rng.normal(0, 0.01)
        scanner_effect = rng.normal(0, 0.1)
        batch_effect = rng.normal(0, 0.1)

        # Apply the effects to the brain measures
        brain_measures[:, i] += sex_effect * sex + age_effect * age + scanner_effect * scanner + batch_effect * batch
        brain_measures_unbiased[:, i] += sex_effect * sex + age_effect * age + scanner_effect * scanner

    # Combine brain measures and covariates into a pandas DataFrame
    columns = [f'Feature_{i + 1}' for i in range(n_features)] + ['Sex', 'Age', 'Scanner', 'Batch']
    data = np.column_stack((brain_measures, sex, age, scanner, batch))
    data_unbiased = np.column_stack((brain_measures_unbiased, sex, age, scanner, batch))
    df = pd.DataFrame(data, columns=columns)
    df_unbiased = pd.DataFrame(data_unbiased, columns=columns)

    return df, df_unbiased

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.stats import f_oneway

def plot_batch_effects(df):
    n_batches = len(df['Batch'].unique())
    # Separate features and confounders
    feature_cols = [c for c in df.columns if c.lower().startswith('feature')]
    X = df[feature_cols].values
    confounders = df[['Sex', 'Age', 'Scanner']].values

    # Correct for confounders using linear regression
    lr = LinearRegression()
    lr.fit(confounders, X)
    residuals = X #- lr.predict(confounders)

    # Perform PCA on residuals
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(residuals)
    principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

    # Add batch information to principal components dataframe
    principal_df['Batch'] = df['Batch']

    # Perform ANOVA for PC1
    model_pc1 = sm.formula.ols('PC1 ~ C(Batch)', data=principal_df).fit()
    anova_results_pc1 = sm.stats.anova_lm(model_pc1)

    # Perform ANOVA for PC2
    model_pc2 = sm.formula.ols('PC2 ~ C(Batch)', data=principal_df).fit()
    anova_results_pc2 = sm.stats.anova_lm(model_pc2)

    # Visualize PCA and boxplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))

    # PCA plot
    sns.scatterplot(ax=axes[0], x='PC1', y='PC2', hue='Batch', data=principal_df, palette='colorblind')
    axes[0].set_title('PCA plot \n(corrected for age and sex and scanner)')

    # Boxplots
    sns.boxplot(ax=axes[1], x='Batch', y='PC1', data=principal_df, palette='colorblind')
    sns.boxplot(ax=axes[2], x='Batch', y='PC2', data=principal_df, palette='colorblind')
    axes[1].set_title('Boxplots of Principal Components (corrected for age and sex)')

    # Report p-values for PC1
    pvalue_pc1 = anova_results_pc1['PR(>F)'][0]
    axes[1].text(n_batches - 0.5, principal_df['PC1'].max(), f"p-value PC1: {pvalue_pc1:.5f}",
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'), 
                horizontalalignment='right', verticalalignment='top')

    # Report p-values for PC2
    pvalue_pc2 = anova_results_pc2['PR(>F)'][0]
    axes[2].text(n_batches - 0.5, principal_df['PC2'].max(), f"p-value PC2: {pvalue_pc2:.5f}",
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'), 
                horizontalalignment='right', verticalalignment='top')

    return fig

if __name__ == "__main__":
    # Simulate brain measures
    n_samples = 100
    n_features = 10
    n_batches = 4
    df = simulate_brain_measures(n_samples, n_features, n_batches)
    g = plot_batch_effects(df)
    plt.show()