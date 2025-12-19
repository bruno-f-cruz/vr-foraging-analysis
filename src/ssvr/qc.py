import matplotlib.pyplot as plt
import numpy as np
from contraqctor.qc import ContextExportableObj, Suite

from .dataset import SessionDataset
from .visualization import plot_ethogram, plot_session_trials


class SingleSiteBehaviorSuite(Suite):
    def __init__(self, session_dataset: SessionDataset):
        self.session_dataset = session_dataset

    def test_metrics(self):
        metrics = self.session_dataset.session_metrics
        metrics_str = "\n".join(f"{field}: {getattr(metrics, field)}" for field in metrics.__dataclass_fields__)
        return self.pass_test(True, f"Session metrics:\n{metrics_str}")

    def test_visualize_raw_ethogram(self):
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax_velocity, ax_events = plot_ethogram(
                self.session_dataset,
                t_start=self.session_dataset.trials["odor_onset_time"][30],
                t_end=self.session_dataset.trials["odor_onset_time"][40],
                ax=ax,
            )
        except Exception as e:
            return self.fail_test(False, f"Failed to plot ethogram: {e}")
        context = ContextExportableObj.as_context(fig)
        return self.pass_test(True, "Ethogram plotted successfully", context=context)

    def test_visualize_summary_behavior(self):
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax = plot_session_trials(self.session_dataset, alpha=0.33, ax=ax)

            time_of_trial = self.session_dataset.trials["odor_onset_time"]

            blocks = self.session_dataset.dataset["Behavior"]["SoftwareEvents"]["Block"].load().data.copy()
            block_times = blocks.index.values
            trial_indices = time_of_trial.searchsorted(block_times, side="right") - 1
            trial_indices = np.maximum(trial_indices, 0)
            blocks["trial_idx"] = time_of_trial.iloc[trial_indices].index.values
            ax.vlines(
                blocks["trial_idx"].values,
                ymin=ax.get_ylim()[0],
                ymax=ax.get_ylim()[1],
                colors="k",
                linestyles="dashed",
                label="Block Change",
            )
        except Exception as e:
            return self.fail_test(False, f"Failed to plot summary behavior: {e}")

        context = ContextExportableObj.as_context(fig)
        return self.pass_test(True, "Summary behavior plotted successfully", context=context)
