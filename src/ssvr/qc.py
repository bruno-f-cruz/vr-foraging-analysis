import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from aind_behavior_vr_foraging.data_qc.data_qc import make_qc_runner
from contraqctor.qc import ContextExportableObj, HtmlReporter, Suite

from .dataset import SessionDataset
from .visualization import plot_ethogram, plot_session_trials

logger = logging.getLogger(__name__)


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


def run_qc(session_datasets: list[SessionDataset], path: Path = Path("./derived") / "qc_reports"):
    plt.ioff()
    path.mkdir(parents=True, exist_ok=True)

    for session in session_datasets:
        print(f"{session.session_info.session_id}", end="")
        if (path / f"{session.session_info.session_id}.html").exists():
            print("  - Skipping existing report")
            continue
        try:
            print("  - Running QC...")
            runner = make_qc_runner(session.dataset)
            runner.add_suite(SingleSiteBehaviorSuite(session_dataset=session), group="SingleSiteBehaviorSuite")
            qc_path = path / f"{session.session_info.session_id}.html"
            reporter = HtmlReporter(output_path=qc_path)
            results = runner.run_all()
            reporter.report_results(results)
        except Exception as e:
            logging.error(f"Failed to run QC for session {session.session_info.session_id}: {e}")

    files = sorted(p for p in path.glob("*.html") if p.name != "index.html")

    with open(path / "index.html", "w", encoding="utf-8") as f:
        f.write("<!doctype html><html><body><ul>\n")
        for p in files:
            f.write(f'<li><a href="{p.name}">{p.stem}</a></li>\n')
        f.write("</ul></body></html>\n")
    plt.ion()
