import marimo

__generated_with = "0.10.15"
app = marimo.App(width="full", app_title="ODE Evaluation")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    import os
    os.environ["OMP_NUM_THREADS"] = "6" # export OMP_NUM_THREADS=4
    os.environ["OPENBLAS_NUM_THREADS"] = "6" # export OPENBLAS_NUM_THREADS=4
    os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6

    os.chdir(mo.notebook_dir())
    return (os,)


@app.cell
def _():
    from matplotlib import pyplot as plt

    import torch
    from torch import nn

    from omegaconf import OmegaConf
    from ase.visualize.plot import plot_atoms
    from ase import Atoms
    from schnetpack import properties
    from schnetpack.model import NeuralNetworkPotential
    import schnetpack.transform as trn
    from schnetpack.datasets import QM9
    from tqdm import tqdm
    import ase

    from morered.datasets import QM9Filtered, QM7X
    from morered.noise_schedules import PolynomialSchedule, CosineSchedule
    from morered.processes import VPGaussianDDPM
    from morered.utils import scatter_mean, check_validity, generate_bonds_data, batch_center_systems, batch_rmsd
    from morered.sampling import DDPM, MoreRedJT, MoreRedAS, MoreRedITP, ConsistencySampler
    from morered.reverse_process import ReverseODE, ReverseODEHeun
    return (
        Atoms,
        ConsistencySampler,
        CosineSchedule,
        DDPM,
        MoreRedAS,
        MoreRedITP,
        MoreRedJT,
        NeuralNetworkPotential,
        OmegaConf,
        PolynomialSchedule,
        QM7X,
        QM9,
        QM9Filtered,
        ReverseODE,
        ReverseODEHeun,
        VPGaussianDDPM,
        ase,
        batch_center_systems,
        batch_rmsd,
        check_validity,
        generate_bonds_data,
        nn,
        plot_atoms,
        plt,
        properties,
        scatter_mean,
        torch,
        tqdm,
        trn,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Paths""")
    return


@app.cell
def _():
    # path to store the dataset as ASE '.db' files
    split_file_path = "../split.npz"

    # model path
    models_path = "../models"
    return models_path, split_file_path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Load Model""")
    return


@app.cell
def _(models_path, os, torch):
    # time predictor
    denoiser = torch.load(os.path.join(models_path, "qm7x_ddpm.pt"), map_location="cpu")
    return (denoiser,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Define Sampler""")
    return


@app.cell
def _(PolynomialSchedule, ReverseODE, VPGaussianDDPM, denoiser, torch):
    # define the noise schedule
    T = 500
    noise_schedule = PolynomialSchedule(T=T, s=1e-5, dtype=torch.float64, variance_type="lower_bound")

    # define the forward diffusion process
    diff_proc = VPGaussianDDPM(noise_schedule, noise_key="eps", invariant=True, dtype=torch.float64)

    # define reverse ODE
    reverse_ode = ReverseODE(diffusion_process=diff_proc, denoiser = denoiser, recompute_neighbors=False)
    return T, diff_proc, noise_schedule, reverse_ode


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Define Data Loader""")
    return


@app.cell
def _(os):
    tut_path = "./tut"
    os.makedirs(tut_path, exist_ok=True)
    return (tut_path,)


@app.cell
def _(trn):
    transforms=[
        trn.CastTo64(),
        trn.SubtractCenterOfGeometry(),
    ]
    return (transforms,)


@app.cell
def _(os, tut_path):
    # path to store the dataset as ASE '.db' files
    datapath = os.path.join(tut_path, "qm7x.db")
    return (datapath,)


@app.cell
def _(QM7X, datapath, split_file_path, transforms):
    data = QM7X(
        datapath=datapath,
        only_equilibrium=True,
        batch_size=10,
        split_file=split_file_path,
        transforms=transforms,
        num_workers=2,
        pin_memory=False,
        load_properties=["rmsd"],
    )
    return (data,)


@app.cell
def _(data, generate_bonds_data):
    # prepare and setup the dataset
    data.prepare_data()
    data.setup()
    bonds_data = generate_bonds_data()
    return (bonds_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Evaluate Reverse ODE""")
    return


@app.cell
def _():
    # time step for diffusion
    t = 300
    return (t,)


@app.cell
def _(
    bonds_data,
    check_validity,
    data,
    diff_proc,
    properties,
    reverse_ode,
    t,
    torch,
):
    data_iter = iter(data.train_dataloader())
    n_batches = len(data_iter)

    limit_batches = 10

    # initialize counters
    total_atoms = 0
    total_molecules = 0
    molecule_stats = {'stable_molecules': 0, 'stable_molecules_wo_h': 0}
    atom_stats = {'stable_atoms': 0, 'stable_atoms_wo_h': 0}

    for batch_idx, target in enumerate(data_iter):
        # track total atoms and molecules
        n_atoms = target[properties.n_atoms].sum()
        n_molecules = len(target[properties.n_atoms])
        total_atoms += n_atoms
        total_molecules += n_molecules


        # diffuse batch
        batch = {k: v.clone() for k, v in target.items()}
        x_t, _ = diff_proc.diffuse(
            batch[properties.R], idx_m=batch[properties.idx_m], t=torch.tensor(t)
        )
        batch[properties.R] = x_t

        # denoise batch and check validity
        denoised,_,_ = reverse_ode.denoise(batch, t=t)
        batch.update(denoised)
        validity_res = check_validity(batch, *bonds_data.values())
        for m_key in molecule_stats.keys():
            molecule_stats[m_key] += sum(validity_res[m_key])
        for a_key in atom_stats.keys():
            for i in range(n_molecules):
                atom_stats[a_key] += sum(validity_res[a_key][i])

        print(f"completed: {batch_idx + 1}/{n_batches}")

        if limit_batches and batch_idx >= limit_batches:
            break
    return (
        a_key,
        atom_stats,
        batch,
        batch_idx,
        data_iter,
        denoised,
        i,
        limit_batches,
        m_key,
        molecule_stats,
        n_atoms,
        n_batches,
        n_molecules,
        target,
        total_atoms,
        total_molecules,
        validity_res,
        x_t,
    )


@app.cell
def _(atom_stats, molecule_stats, total_atoms, total_molecules):
    # normalize molecule stats
    for k,v,in molecule_stats.items():
       print(f"{k}: {v / total_molecules}")

    # normalize atom stats
    for k,v,in atom_stats.items():
        print(f"{k}: {v / total_atoms}")
    return k, v


if __name__ == "__main__":
    app.run()
