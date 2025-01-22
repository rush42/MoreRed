import marimo

__generated_with = "0.10.15"
app = marimo.App()


@app.cell
def _():
    import os
    os.environ["OMP_NUM_THREADS"] = "6" # export OMP_NUM_THREADS=4
    os.environ["OPENBLAS_NUM_THREADS"] = "6" # export OPENBLAS_NUM_THREADS=4
    os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
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
    from morered import ReverseODE
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
    split_file_path = "./split.npz"

    # model path
    models_path = "../models"
    return models_path, split_file_path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Load model""")
    return


@app.cell
def _(models_path, os, torch):
    # time predictor
    time_predictor = torch.load(os.path.join(models_path, "qm7x_time_predictor.pt"), map_location="cpu")
    return (time_predictor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Define sampler""")
    return


@app.cell
def _(PolynomialSchedule, VPGaussianDDPM, torch):
    # define the noise schedule
    T = 200
    noise_schedule = PolynomialSchedule(T=T, s=1e-5, dtype=torch.float64, variance_type="lower_bound")

    # define the forward diffusion process
    diff_proc = VPGaussianDDPM(noise_schedule, noise_key="eps", invariant=True, dtype=torch.float64)
    return T, diff_proc, noise_schedule


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Define data loader""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Define paths""")
    return


@app.cell
def _():
    tut_path = "./tut"
    return (tut_path,)


@app.cell
def _(os, tut_path):
    os.makedirs(tut_path, exist_ok=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Define data input transformations""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""QM7X""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""NOTE! QM7-X is large. Therefore, it takes time to download the dataset and prepare it""")
    return


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
        batch_size=1,
        split_file=split_file_path,
        transforms=transforms,
        num_workers=2,
        pin_memory=False,
        load_properties=["rmsd"],
    )
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Prepare dataset""")
    return


@app.cell
def _(data):
    # prepare and setup the dataset
    data.prepare_data()
    data.setup()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Representation Loss""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Load a batch""")
    return


@app.cell
def _(data):
    # train split here is not the same as during training
    target = next(iter(data.test_dataloader()))
    return (target,)


@app.cell
def _(
    NeuralNetworkPotential,
    ReverseODE,
    diff_proc,
    nn,
    time_predictor,
    trn,
):
    model_output = nn.Identity()
    model_output.model_outputs = ['vector_representation', 'scalar_representation']
    representation = NeuralNetworkPotential(input_modules=time_predictor.input_modules, representation=time_predictor.representation, output_modules=nn.ModuleList([model_output]))

    rode = ReverseODE(diff_proc, time_predictor)
    caster = trn.CastTo32()
    def prepare_batch(batch):
        return caster(rode.prepare_batch(batch))
    return caster, model_output, prepare_batch, representation, rode


@app.cell
def _(representation):
    for param in representation.parameters():
        param.requires_grad = False
    return (param,)


@app.cell
def _(T, diff_proc, properties, target, torch):
    timesteps = torch.arange(T).unsqueeze(-1).unsqueeze(-1)
    traj, _ = diff_proc.diffuse(target[properties.R], idx_m = None, t=timesteps)
    return timesteps, traj


@app.cell
def _(properties, target):
    _n = target[properties.n_atoms][0]
    power = 2
    def vector_norm_loss(target, pred):
        diff = pred['vector_representation'] - target['vector_representation']
        return diff.detach().norm(dim=1).pow(power).mean() / _n

    def vector_loss(target, pred):
        diff = pred['vector_representation'] - target['vector_representation']
        return diff.detach().norm().pow(power) / _n

    def scalar_loss(target, pred):
        diff = pred['scalar_representation'] - target['scalar_representation']
        return diff.detach().norm().pow(power) / _n
    return power, scalar_loss, vector_loss, vector_norm_loss


@app.cell
def _(
    T,
    prepare_batch,
    properties,
    representation,
    scalar_loss,
    target,
    traj,
    vector_loss,
    vector_norm_loss,
):
    _batch = {key: val.clone() for key, val in target.items()}
    target_hat = prepare_batch(target)
    target_reprs = representation(target_hat)
    vector_norm_losses = []
    vector_losses = []
    scalar_losses = []
    for i in range(T):
        _batch[properties.R] = traj[i]
        batch_hat = prepare_batch(_batch)
        batch_reprs = representation(batch_hat)
        vector_norm_losses.append(vector_norm_loss(target_reprs, batch_reprs))
        vector_losses.append(vector_loss(target_reprs, batch_reprs))
        scalar_losses.append(scalar_loss(target_reprs, batch_reprs))
    return (
        batch_hat,
        batch_reprs,
        i,
        scalar_losses,
        target_hat,
        target_reprs,
        vector_losses,
        vector_norm_losses,
    )


@app.cell
def _(plt, scalar_losses, vector_losses, vector_norm_losses):
    plt.figure(figsize=(8, 4))
    plt.xlabel(r"t")
    plt.ylabel(r"loss")

    plt.plot(scalar_losses, '.', label='scalar')
    plt.plot(vector_losses, '.', label='vector')
    plt.plot(vector_norm_losses, '.', label='vector (norm)')

    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Tinkering""")
    return


@app.cell
def _(diff_proc, properties, target, torch):
    _batch = {key: val.clone() for key, val in target.items()}
    _batch[properties.R], _ = diff_proc.diffuse(_batch[properties.R], _batch[properties.idx_m], t=torch.tensor(1))
    _n = _batch[properties.R].shape[0]
    return


@app.cell
def _(torch):
    alpha = torch.tensor(torch.pi)
    rot_m = torch.Tensor(
        [
        [torch.cos(alpha), - torch.sin(alpha), 0],
        [torch.sin(alpha),    torch.cos(alpha), 0],
        [0,                   0,                1]
        ]
    ).double()
    return alpha, rot_m


@app.cell
def _(prepare_batch, properties, rot_m, target):
    rotated = {key: val.clone() for key, val in target.items()}
    rotated[properties.R] = target[properties.R] @ rot_m
    rotated_hat = prepare_batch(rotated)
    return rotated, rotated_hat


@app.cell
def _(representation, rotated_hat):
    rotated_reprs = representation(rotated_hat)
    return (rotated_reprs,)


@app.cell
def _(rotated_reprs, scalar_loss, target_reprs, vector_loss):
    print(f"vector loss: {vector_loss(target_reprs, rotated_reprs)}, scalar loss: {scalar_loss(target_reprs, rotated_reprs)}")
    return


@app.cell
def _(prepare_batch, target):
    permuted = {key: val.clone() for key, val in target.items()}
    # for i in range(n):
    #     for j in range(i):
    #         if permuted[properties.Z][i] == permuted[properties.Z][j]:
    #             permuted[properties.R][i, :], permuted[properties.R][j, :] = permuted[properties.R][j, :], permuted[properties.R][i, :]
    # permuted[properties.R][-2, :], permuted[properties.R][-1, :] = permuted[properties.R][-1, :], permuted[properties.R][-2, :]
    permuted_hat = prepare_batch(permuted)
    return permuted, permuted_hat


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
