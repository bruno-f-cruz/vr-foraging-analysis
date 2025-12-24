import numpy as np
import pandas as pd
from aind_behavior_vr_foraging import task_logic

from ssvr.dataset import SessionDataset

## These all operate "in-place"


def determine_patch_probabilities(row) -> tuple[task_logic.Block, np.ndarray]:
    block_obj = task_logic.Block.model_validate(row["data"])
    patches = block_obj.environment_statistics.patches
    arr = np.ndarray(shape=(len(patches),), dtype=float)
    for patch in patches:
        arr[patch.state_index] = patch.reward_specification.probability.distribution_parameters.value
    return block_obj, arr


def enrich_with_block_info(session: SessionDataset) -> pd.DataFrame:
    trials = session.trials
    ## Get trials since block start
    block = session.dataset["Behavior"]["SoftwareEvents"]["Block"].data

    block["block"], block["block_patch_probabilities"] = zip(*block.apply(determine_patch_probabilities, axis=1))
    block["block_index"] = range(len(block))

    def find_closest_past_block_index(trial_time):
        past_blocks = block[block.index <= trial_time]
        if len(past_blocks) == 0:
            return None
        return past_blocks.index[-1]

    timestamps = trials["odor_onset_time"].apply(find_closest_past_block_index)
    trials["block"] = timestamps.apply(lambda idx: block.loc[idx, "block"] if idx is not None else None)
    trials["block_patch_probabilities"] = timestamps.apply(
        lambda idx: block.loc[idx, "block_patch_probabilities"] if idx is not None else None
    )
    trials["block_index"] = timestamps.apply(lambda idx: block.loc[idx, "block_index"] if idx is not None else None)
    trials["high_patch_index"] = trials["block_patch_probabilities"].apply(np.argmax)
    return trials


def enrich_with_relative_to_block(session: SessionDataset) -> pd.DataFrame:
    trials = session.trials

    block_indices = trials["block_index"].values
    patch_indices = trials["patch_index"].values
    n = len(trials)
    transitions = [0] + list((block_indices[1:] != block_indices[:-1]).nonzero()[0] + 1) + [n]

    trials_from_last_block = [
        i - transitions[block]
        for block in range(len(transitions) - 1)
        for i in range(transitions[block], transitions[block + 1])
    ]
    trials_to_next_block = [
        transitions[block + 1] - i - 1
        for block in range(len(transitions) - 1)
        for i in range(transitions[block], transitions[block + 1])
    ]

    # By trial type (patch_index)
    trials_from_last_block_by_trial_type = [0] * n
    trials_to_next_block_by_trial_type = [0] * n

    for block in range(len(transitions) - 1):
        start = transitions[block]
        end = transitions[block + 1]
        for i in range(start, end):
            # Count trials of the same patch_index since last block
            count = 0
            for j in range(start, i + 1):
                if patch_indices[j] == patch_indices[i]:
                    count += 1
            trials_from_last_block_by_trial_type[i] = count - 1  # exclude current trial

        for i in range(start, end):
            count = 0
            for j in range(i + 1, end):
                if patch_indices[j] == patch_indices[i]:
                    count += 1
            trials_to_next_block_by_trial_type[i] = count

    trials["trials_from_last_block"] = trials_from_last_block
    trials["trials_to_next_block"] = trials_to_next_block
    trials["trials_from_last_block_by_trial_type"] = trials_from_last_block_by_trial_type
    trials["trials_to_next_block_by_trial_type"] = trials_to_next_block_by_trial_type
    return trials


def enrich_with_previous_trial(session: SessionDataset, n_previous: int = 5) -> pd.DataFrame:
    trials = session.trials
    columns_to_add = ["is_rewarded", "is_choice", "patch_index"]

    n = len(trials)
    for col in columns_to_add:
        for i in range(1, n_previous + 1):
            new_col = f"{col}_past_{i}"
            trials[new_col] = [trials[col][idx - i] if idx - i >= 0 else None for idx in range(n)]
    return trials


def enrich_with_block_probability(session: SessionDataset) -> pd.DataFrame:
    trials = session.trials

    def get_block_reward_probability(row):
        patch_index = row["patch_index"]
        block_probabilities = row["block_patch_probabilities"]
        if block_probabilities is None:
            return None
        return block_probabilities[patch_index]

    trials["block_reward_probability"] = trials.apply(get_block_reward_probability, axis=1)
    return trials
