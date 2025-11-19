# %% [markdown]
# # This Notebook Runs Baselines (BPOSD, etc.)

# %%
import json
from functools import partial
from io import StringIO
from operator import itemgetter

from surface_sim.setups.setup import SetupDict

from mccd.random_clifford_circuit import *
from surface_sim.setups import CircuitNoiseSetup
from surface_sim.models import CircuitNoiseModel, BiasedCircuitNoiseModel
from surface_sim import Detectors, Setup
from surface_sim.experiments import schedule_from_circuit, experiment_from_schedule
import time
import stim

from pathlib import Path
import stim
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed
import itertools
import shelve
from surface_sim.layouts import rot_surface_codes

from pymatching import Matching as MWPM
from mle_decoder import MLEDecoder as MLE
from stimbposd import BPOSD
from sklearn.metrics import accuracy_score

import os

os.environ['GRB_LICENSE_FILE'] = '/Users/fengcong/.gurobi/gurobi.lic'

# %% [markdown]
# ## Baseline Decoders
# 
# - MWPM. We use the open-source library PyMatching with the noise model used for data generation as detailed in the ‘Experimentally motivated noise model’ subsection.
# 
# - BP-OSD. We use the open-source library stimbposd. We use the exact noise model used for data generation and set the maximal belief propagation iterations to 20.
# 
# - MLE. We use the algorithm developed and implemented as in ref. 14.
# 
# ### Notes
# 
# All baselines have PyPI packages.
# 
# ```
# pymatching
# mle-decoder
# stimbposd
# ```

# %%
DECODER_BASELINES = {
    'BPOSD': partial(BPOSD, max_bp_iters=20),
    'MLE': MLE, # TODO: model too large. acacdemic license.
    'MWPM': MWPM,
}

# %% [markdown]
# ## Basic Gates & Surface Code
# 
# MCCD uses I, X, Y, Z, H (single qubit gates) and CX (two qubit gates).
# 
# MCCD uses Rotated Surface Code.
# 
# surface-sim supports I, X, Z for rotated gates and I, H, X, Z for unrotated gates.
# 
# ### Notes
# 
# We use the gates which `surface-sim` supports.

# %%
def print_random_circuit(c: RandomCliffordCircuit):
    return list(c)

def dict_product(input_dict):
    keys = input_dict.keys()
    value_lists = input_dict.values()

    # 使用itertools.product生成所有值的组合
    value_combinations = itertools.product(*value_lists)

    # 将每个值的组合与键配对，生成字典列表
    for combo in value_combinations:
        yield dict(zip(keys, combo))

def run_decoder(name: str, circuit: stim.Circuit, shots: int):
    """Runs decoder on the given circuit

    Args:
        name: decoder name
        circuit: circuit to run
        shots: number of shots

    Returns:
        A dict containing the decoder metrics.
    """
    method = DECODER_BASELINES[name](circuit.detector_error_model())
    sampler = circuit.compile_detector_sampler()
    syndrome, labels = sampler.sample(shots=shots, separate_observables=True)
    begin = time.time_ns()
    predictions = method.decode_batch(syndrome)
    end = time.time_ns()
    logical_accuracy = accuracy_score(labels, predictions)
    walltime_seconds = (end - begin) / 1e9
    return dict(
        decoder=name,
        logical_accuracy=logical_accuracy,
        walltime_seconds=walltime_seconds,
    )

def run_decoder_tasks(root_dir, bench_circuits, bench_decoders, df_name):
    """Run all the baseline decoders on the benchmark circuit.

    Args:
        bench_circuits: Benchmark circuits.
        df_name: Name of the dataframe file.

    Returns:
        The result dataframe.
    """
    root_dir = Path(root_dir) / df_name
    root_dir.mkdir(parents=True, exist_ok=True)

    def run_decoder_plus(config, cir_str, i, **kwargs):
        res = config.copy()
        res.update(kwargs)
        save_dir = root_dir / f'res{i}.json'
        save_dir.parent.mkdir(parents=True, exist_ok=True)
        if save_dir.is_file():
            return

        kwargs['circuit'] = stim.Circuit.from_file(StringIO(cir_str))
        try:
            res.update(run_decoder(**kwargs))
            save_dir.write_text(json.dumps(res))
        except Exception as e:
            print(e)

    def tasks():
        i = 0
        for phy_cir, config in bench_circuits:
            for decoder in bench_decoders:
                i += 1
                yield delayed(run_decoder_plus)(config, phy_cir, i-1,
                                                name=decoder, shots=num_shots)

    Parallel(n_jobs=-1, verbose=1)(tasks())

    bench_result = list(root_dir.glob('res*.json'))

    df = pd.DataFrame.from_records(bench_result)
    df = pd.melt(df, id_vars=['decoder', 'distance', 'depth', 'circuit_type_index'],
             value_vars=['walltime_seconds', 'logical_accuracy'],
             var_name='metric',
             value_name='value')

    filename = root_dir / '..' / f'{df_name}.csv'
    filename.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filename, index=False)
    print('done')
    return df


def load_circuits(root_dir: Path):
    circuits = []
    for config_path in sorted(root_dir.glob('*_config.json')):
        config = json.loads(config_path.read_text())

        for cir_path in sorted(root_dir.glob(config_path.stem.replace('_config', '_phy_trial*.stim'))):
            phy_cir = stim.Circuit.from_file(cir_path)
            circuits.append((str(phy_cir), config))

    return circuits
    

# %% [markdown]
# ## Circuit Depths
# 
# From released source_data.zip, Type I circuits have depths:
# 
# array([ 2,  4,  6,  8, 10, 12, 14, 16, 18])
# 
# Type II circuits have depths:
# 
# array([ 4,  8, 12, 16, 20, 24, 28, 32, 36])
# 

# %% [markdown]
# ### Shots & Repeat
# We evaluate each decoder over 20 independent runs. In each run, we randomly sample 1,000 syndrome trajectories from Type I/II circuits and average them to obtain a run-level performance estimate. We report the mean across the 20 runs, with error bars showing s.e.m.
# 

# %% [markdown]
# Load circuits.

# %%
noise_model = 'average_depolarizing_noise'
root_dir = Path('./data/bench') / noise_model
num_shots = 1000

# %%
fig4_circuits = load_circuits(root_dir / 'fig4/circuits')
fig5_circuits = load_circuits(root_dir / 'fig5/circuits')

# %%
len(fig4_circuits), len(fig5_circuits)

# %%
df4 = run_decoder_tasks(root_dir / 'fig4/result',
    fig4_circuits, DECODER_BASELINES.keys(), 'fig4-baselines')


# %%
df5 = run_decoder_tasks(root_dir / 'fig5/result',
    fig5_circuits, DECODER_BASELINES.keys(), 'fig5-baselines')



