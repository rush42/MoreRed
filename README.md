# MoreRed: Molecular Relaxation by Reverse Diffusion with Time Step Prediction

[MoreRed (Molecular Relaxation by Reverse Diffusion)](https://iopscience.iop.org/article/10.1088/2632-2153/ad652c) is a generative diffusion model that can generate new structures or denoise arbitrarily noisy ones. Unlike previous geometry relaxation methods, which require labeled equilibrium and non-equilibrium structures, MoreRed is trained using **exclusively unlabeled equilibrium structures**. Despite this, it effectively relaxes non-equilibrium structures, achieving competitive results with **much less data**, exhibiting better robustness to the noise level in the input and reducing computation time during relaxation.

<table align="center", border=0>
  <tr>
    <td rowspan="2">
      <img src="https://github.com/khaledkah/MoreRed/assets/56682622/5f7a680e-7fd2-434e-b3a8-abc2aad6d39f" width="550" height="420">
    </td>
    <td>
      <img src="https://github.com/khaledkah/MoreRed/assets/56682622/a02032ba-a3a2-4b20-9658-faada1cbdd73" width="300" height="200">
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/khaledkah/MoreRed/assets/56682622/dc18a881-8abc-48c8-a704-e10dc528998c" width="300" height="200">
    </td>
  </tr>
</table>

MoreRed is built on top of [SchNetPack 2.0](https://github.com/atomistic-machine-learning/schnetpack/tree/master), an easily configurable and extendible library for constructing and training neural network models for atomistic systems like molecules. SchNetPack utilizes [PyTorch Lightning](https://www.pytorchlightning.ai/) for model building and [Hydra](https://hydra.cc/) for straightforward management of experimental configurations. While high level usage of the `morered` package to train and use the models described in the original paper does not require knowledge of its underlying dependencies, we recommend users familiarize themselves with Hydra to be able to customize their experimental configurations. Additionally, the tutorials and the documentation provided in [SchNetPack 2.0](https://github.com/atomistic-machine-learning/schnetpack/tree/master) can be helpful. Below, we explain how to use the `morered` package.

**_NOTE: while the current documentation in the README file and the source code should be sufficient to use the package easily, we will continually enhance it._**

#### Content

+ [News](/README.md##News)
+ [Installation](/README.md##Installation)
+ [Using pre-trained models](/README.md##Using-pre-trained-models)
+ [Molecular relaxation and structure generation](/README.md##Molecular-relaxation-relaxation-and-structure-generation)
+ [Tutorials](/README.md##Tutorials)
+ [Training your own models](/README.md##Training-your-own-models)
+ [How to cite](/README.md##How-to-cite)

## News
- Under the folder `morered`, we uploaded the final models trained on QM9 and QM7-X datasets
- We updated the notebook `denoising_tutorial.ipynb` with more details on using the trained models.

## Installation
Requirements:
- python >= 3.8
- SchNetPack 2.0

You can install `morered` from the source code using pip, which will also install all its required dependencies including SchNetPack:

Download this repository. e.g. by cloning it using:
```
git clone git@github.com:khaledkah/MoreRed.git
cd MoreRed
```
We recommend creating a new Python environment or using conda to avoid incompatibilities with previously installed packages. E.g. if using conda:
```
conda create -n morered python=3.12
conda activate morered
```
Now to install the package, inside the folder `MoreRed` run:
```
pip install .
```
## Using pre-trained models
Under the folder `models`, you can find the final models trained on QM9 and Qm7-X datasets until complete convergence. You can load the models using `torch.load()` command. Besides, the tutorial notebooks provide details on how to use the models.

## Molecular relaxation and structure generation
The notebook `notebooks/denoising_tutorial.ipynb` explains how the trained models can be used for denoising and generation from scratch using different samplers.
Under `src/morered/sampling`, you can find ready-to-use Python classes implementing the different samplers: `MoreRed-ITP`, `MoreRed-JT`, `MoreRed-AS`, `DDPM`. The same classes can be used for denoising/relaxation of noisy structures as well as for new structure generation.

## Tutorials
Under `notebooks`, we provide different tutorials in the form of Jupyter notebooks, that will be continually updated:
  - `diffusion_tutorial.ipynb`: explains how to use the diffusion processes implemented in `morered`.
  - `denoising_tutorial.ipynb`: explains how to use the trained models with the different samplers implemented in `morered` for noisy structure relaxation or generation from scratch. Under the folder `models`, we provide final models trained on QM9 and QM7-X datasets until complete convergence.

## Training your own models
The human-readable and customizable YAML configuration files under `src/morered/configs` are all you need to train and run customizable experiments with `morered`. They follow the configuration structure used in [SchNetPack 2.0](https://github.com/atomistic-machine-learning/schnetpack/tree/master). Here, we explain how to train and use the different models.

Installing `morered` using pip adds the new CLI command `mrdtrain`, which can be used to train the different models by running the command:
```
mrdtrain experiment=<my-experiemnt>
```
where `<my-experiment>` specifies the experimental configurations to be used. It can either be one of the pre-installed experiments within the package, under `src/morered/configs/experiments`, or a path to a new YAML file created by the user. Detailed instructions on creating custom configurations can be found in the documentation of [SchNetPack 2.0](https://github.com/atomistic-machine-learning/schnetpack/tree/master).

In the original paper, three variants of MoreRed were introduced:

#### MoreRed-JT:
You can train the `MoreRed-JT` variant on QM7-X with the default configuration by simply running:
```
mrdtrain experiment=vp_gauss_morered_jt
```

#### MoreRed-AS/ITP:
Both variants, `MoreRed-AS` and `MoreRed-ITP`, require separately trained time and noise predictors. The noise predictor here is also the usual DDPM model and can be trained using:
```
mrdtrain experiment=vp_gauss_ddpm
```
The time predictor can be trained by running:
```
mrdtrain experiment=vp_gauss_time_predictor
```

#### Train on QM9
To train the models on QM9 instead of QM7-X you can append the suffix `_qm9` to the experiment name, for instance by running:
```
mrdtrain experiment=vp_gauss_morered_jt_qm9
```
Otherwise, you can use the CLI to overwrite the Hydra configurations of the data set by running:
```
mrdtrain experiment=vp_gauss_morered_jt data=qm9_filtered
```

#### Use your config files
To use your config files, you can define the configurations in YAML files and refer to the directory containing these files:

```
mrdtrain --config-dir=<path/to/my_configs> experiment=<my_experiment>

```
More about overwriting configurations in the CLI can be found in the [SchNetPack 2.0](https://github.com/atomistic-machine-learning/schnetpack/tree/master) documentation. 

## How to cite
if you use MoreRed in your research, please cite the corresponding publication:

Kahouli, K., Hessmann, S. S. P., Müller, K.-R., Nakajima, S., Gugler, S., & Gebauer, N. W. A. (2024). Molecular relaxation by reverse diffusion with time step prediction. Machine Learning: Science and Technology, 5(3), 035038. [doi:10.1088/2632-2153/ad652c](https://doi.org/10.1088/2632-2153/ad652c)

    @article{kahouli2024morered,
      doi = {10.1088/2632-2153/ad652c},
      url = {https://dx.doi.org/10.1088/2632-2153/ad652c},
      year = {2024},
      month = {aug},
      publisher = {IOP Publishing},
      volume = {5},
      number = {3},
      pages = {035038},
      author = {Khaled Kahouli and Stefaan Simon Pierre Hessmann and Klaus-Robert Müller and Shinichi Nakajima and Stefan Gugler and Niklas Wolf Andreas Gebauer},
      title = {Molecular relaxation by reverse diffusion with time step prediction},
      journal = {Machine Learning: Science and Technology},
    }
