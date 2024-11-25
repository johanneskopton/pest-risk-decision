from functools import cached_property

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics

class DataPreprocessor:
    def __init__(
        self,
        df,
        covariate_cols,
        y_col,
        location_min
        ):
        self.df = df
        self.covariate_cols = covariate_cols
        self.y_col = y_col
        self.location_min = location_min

    @cached_property
    def normalization_bounds(self):
        normalization_bounds = dict()
        for column in self.df.columns:
            if self.df[column].dtype.kind in 'iuf':
                col_min = self.df[column].min()
                col_max = self.df[column].max()
                normalization_bounds[column] = col_min, col_max
        return normalization_bounds

    def normalize(self, array, col_name):
        norm_min, norm_max = self.normalization_bounds[col_name]
        return (array - norm_min) / (norm_max - norm_min) * 2 - 1

    def prepare_space_coords(self, df):
        X = df[["latitude", "longitude"]] - self.location_min
        return X

    def prepare_time_coords(self, df):
        X = df["date"] / 365
        return X

    def prepare_covariates(self, df):
        X = pd.DataFrame()
        for i, covariate_col in enumerate(self.covariate_cols):
            col_array = df[covariate_col]
            col_array = self.normalize(col_array, covariate_col)
            X[covariate_col] = col_array
        return X

    def get_X(self, df):
        X = pd.DataFrame()
        X[["t_lat", "t_lon"]] = self.prepare_space_coords(df)
        X["time"] = self.prepare_time_coords(df)
        X[self.covariate_cols] = self.prepare_covariates(df)
        return X

    def get_XY(self, df):
        XY = self.get_X(df)
        XY[self.y_col] = df[self.y_col]
        return XY

    def get_X_numpy(self, df):
        return self.get_X(df).to_numpy(copy=True)

    def get_Y_numpy(self, df):
        return df[self.y_col].to_numpy(copy=True)


def destandardize_date(days, reference_day):
    return reference_day + pd.Timedelta(days=days)

def beauty_print_date(date):
    return date.strftime("%d %B %Y")

def tech_print_date(date):
    return date.strftime("%Y_%m_%d")

def standardize_location(location, location_min):
    return location - location_min


def plot_cross_validation_roc(pred_y_list, test_y_list, ax=None, target="screen"):
    n_cv_splits = len(pred_y_list)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    for fold in range(n_cv_splits):
        viz = sklearn.metrics.RocCurveDisplay.from_predictions(
            test_y_list[fold],
            pred_y_list[fold],
            name=f"        {fold}",
            alpha=0.3,
            lw=1,
            ax=ax,
            # plot_manually for compatibility with older sklearn and cuml
            # plot_chance_level=(fold == self._cv_splits - 1),
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], color="k")

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = sklearn.metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean (AUC = %0.2f)" % (
            mean_auc,
        ),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Mean ROC curve with variability\n(TimeSeriesPrediction)",
    )
    ax.axis("square")
    ax.legend(loc="lower right")

    if ax is None:
        if target == "screen":
            plt.show()
        else:
            fig.savefig(target)
