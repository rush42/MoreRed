import copy
import inspect
import logging
from typing import Dict, Literal, Optional, Union

from morered.diffusion_schedule import DiffusionSchedule
from morered.noise_schedules import NoiseSchedule
from morered.utils import compute_neighbors, find_optimal_permutation, rotate_optimally
import torch
from torch_ema import ExponentialMovingAverage as EMA
from schnetpack import properties
from schnetpack.task import AtomisticTask, ModelOutput, UnsupervisedModelOutput
from schnetpack.transform import CastTo32
from schnetpack.model import NeuralNetworkPotential
from torch import nn
from torchmetrics import Metric

from morered.reverse_process import ReverseODE, ReverseProcess

log = logging.getLogger(__name__)


class DiffModelOutput(ModelOutput):
    """
    define diffusion output head.
    """

    def __init__(
        self,
        name: str,
        nll_metric: Optional[Metric] = None,
        **kwargs,
    ):
        """
        Args:
            name: name of the output.
            nll_metric: metric to compute the NLL.
        """
        super().__init__(name=name, **kwargs)

        if nll_metric is not None:
            self.nll_metrics = {
                "train": nll_metric,
                "val": nll_metric.clone(),
                "test": nll_metric.clone(),
            }
        else:
            self.nll_metrics = None

    def update_nll(
        self, inputs: Dict[str, torch.Tensor], subset: str
    ) -> Dict[str, torch.Tensor]:
        """
        update the NLL metric.

        Args:
            inputs: input batch.
            subset: the dataset split used.
        """
        if self.nll_metrics is not None:
            return self.nll_metrics[subset](inputs)
        else:
            return {}

    def calculate_loss(
        self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        calculate the loss.

        Args:
            pred: outputs.
            target: target values.
        """
        if self.loss_weight == 0 or self.loss_fn is None:
            return torch.tensor(0.0)

        # extract the extra arguments of the loss function if needed
        args_ = inspect.getfullargspec(self.loss_fn).args[2:]
        kwargs = {k: pred[k] for k in args_ if k in pred}

        # calculate the loss using the extra arguments if needed
        if kwargs:
            loss = self.loss_weight * self.loss_fn(
                pred[self.name], target[self.target_property], **kwargs
            )
        else:
            loss = self.loss_weight * self.loss_fn(
                pred[self.name], target[self.target_property]
            )

        return loss


class DiffusionTask(AtomisticTask):
    """
    Defines the diffusion task for pytorch lightning.
    Subclasses the atomistic task and adds the diffusion NLL.
    """

    def __init__(
        self,
        diffuse_property: str,
        skip_exploding_batches: bool = True,
        include_l0: bool = False,
        time_key: str = "t",
        noise_key: str = "eps",
        noise_pred_key: str = "eps_pred",
        t0_noise_key: str = "eps_init",
        t0_noise_pred_key: str = "eps_init_pred",
        **kwargs,
    ):
        """
        Args:
            diffuse_property: property to diffuse.
            skip_exploding_batches: ignore exploding batches during training.
            include_l0: always add one extra forward pass to compute the noise at t=0.
                         Used to compute the l_0 term of the NLL., but slows training.
            time_key: key of the true diffusion time step in the input dictionary.
            noise_key: key of the true noise in the input dictionary.
            noise_pred_key: key of the predicted noise in the output dictionary.
            t0_noise_key: key of the true noise at t=0 in the input dictionary.
            t0_noise_pred_key: key of the predicted noise at t=0 in the output.
        """
        super().__init__(**kwargs)

        self.diffuse_property = diffuse_property
        self.skip_exploding_batches = skip_exploding_batches
        self.include_l0 = include_l0
        self.time_key = time_key
        self.noise_key = noise_key
        self.noise_pred_key = noise_pred_key
        self.t0_noise_key = t0_noise_key
        self.t0_noise_pred_key = t0_noise_pred_key

    def setup(self, stage=None):
        """
        overwrite the pytorch lightning task setup function.
        """
        # call the parent atomistic task setup
        AtomisticTask.setup(self, stage=stage)  # type: ignore

        # force some post-processing transforms during training
        forced_postprocessors = []
        for pp in self.model.postprocessors:
            if hasattr(pp, "force_apply"):
                if pp.force_apply:
                    forced_postprocessors.append(pp)
        self.model.forced_postprocessors = nn.ModuleList(forced_postprocessors)

    def predict_without_postprocessing(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        predict without post-processing transforms.
        Note: forced post-processing transforms will still be applied.

        Args:
            batch: input batch.
        """
        tmp_postprocessors = self.model.postprocessors
        self.model.postprocessors = self.model.forced_postprocessors
        pred = self(batch)
        self.model.postprocessors = tmp_postprocessors

        return pred

    def forward_t0(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Extra forward pass to compute the noise at t=0.
        Usually to compute the l_0 term of the NLL but slows training.

        Args:
            batch: input batch.
        """
        # temporarly save the relavant quantities for L_t.
        tmp_diff_step = batch[self.time_key].clone()
        tmp_noise_pred = batch[self.noise_pred_key].clone()
        tmp_prop = batch[self.diffuse_property].clone()

        # load the initial noise and set the time step to 0.
        idx_t0 = torch.where((batch["ts"] == 0).all(0))[0][0]
        batch[self.diffuse_property] = batch[f"all_{self.diffuse_property}"][:, idx_t0]
        batch[self.time_key] = torch.zeros_like(batch[self.time_key])

        # feed-forward pass for t=0.
        # This will override the neighbors and distances computed for previous t.
        pred = self.predict_without_postprocessing(batch)

        # restore the initial (relevant) predictions for L_t for gradient backprop.
        batch[self.t0_noise_key] = batch[f"all_{self.noise_key}"][:, idx_t0]
        batch[self.t0_noise_pred_key] = pred[self.noise_pred_key]
        batch[self.time_key] = tmp_diff_step
        batch[self.noise_pred_key] = tmp_noise_pred
        batch[self.diffuse_property] = tmp_prop

        return batch

    def log_nll(self, inputs: Dict[str, torch.Tensor], subset: str):
        """
        log the diffusion  NLL.

        Args:
            inputs: input batch.
            subset: the dataset split used.
        """
        for output in self.outputs:
            if hasattr(output, "nll_metrics") and output.nll_metrics is not None:
                metrics = output.update_nll(inputs, subset)
                for key, val in metrics.items():
                    self.log(
                        f"{subset}_{output.name}_{key}",
                        val,
                        on_step=(subset == "train"),
                        on_epoch=(subset != "train"),
                        prog_bar=(subset != "train"),
                        metric_attribute=f"{subset}_{output.name}_{key}",
                    )

    def _step(self, batch: Dict[str, torch.Tensor], subset: str) -> torch.FloatTensor:
        """
        perform one forward pass and calculate the loss and log metrics.

        Args:
            batch: input batch.
            subset: the dataset split used.
        """
        # extract the target values from the batch
        targets = {
            output.target_property: batch[output.target_property]
            for output in self.outputs
            if not isinstance(output, UnsupervisedModelOutput)
        }
        try:
            targets["considered_atoms"] = batch["considered_atoms"]
        except Exception:
            pass

        # predict output quantity
        pred = self.predict_without_postprocessing(batch)

        # apply constraints
        pred, targets = self.apply_constraints(pred, targets)

        # calculate the loss
        loss = self.loss_fn(pred, targets)

        # log loss and metrics
        self.log(
            f"{subset}_loss",
            loss,
            on_step=(subset == "train"),
            on_epoch=(subset != "train"),
            prog_bar=(subset != "train"),
        )
        self.log_metrics(pred, targets, subset)

        # pefrom an extra forward pass for t=0 and log the NLL
        if self.include_l0:
            batch = self.forward_t0(batch)
        self.log_nll(batch, subset)

        return loss  # type: ignore

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Optional[torch.FloatTensor]:
        """
        define the training step for pytorch lightning.

        Args:
            batch: input batch.
            batch_idx: batch index.
        """
        # perform forward pass
        loss = self._step(batch, "train")

        # skip exploding batches in backward pass
        if self.skip_exploding_batches and (
            torch.isnan(loss) or torch.isinf(loss) or loss > 1e10
        ):
            log.warning(
                f"Loss is {loss} for train batch_idx {batch_idx} and training step "
                f"{self.global_step}, training step will be skipped!"
            )
            return None

        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.FloatTensor]:
        """
        define the validation step for pytorch lightning.

        Args:
            batch: input batch.
            batch_idx: batch index.
        """
        # enable non-training gradients with respect to specific quanitites if needed.
        torch.set_grad_enabled(self.grad_enabled)

        # forward pass
        loss = self._step(batch, "val")

        return {"val_loss": loss}

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.FloatTensor]:
        """
        define the test step for pytorch lightning.

        Args:
            batch: input batch.
            batch_idx: batch index.
        """
        # enable non-training gradients with respect to specific quanitites if needed.
        torch.set_grad_enabled(self.grad_enabled)

        # forward pass
        loss = self._step(batch, "test")

        return {"test_loss": loss}


class ConsistencyModelOutput(ModelOutput):
    """
    define consistency output head.
    """

    def __init__(
        self,
        permutation_invariant: bool = True,
        rotation_invariant: bool = False,
        noise_schedule: Optional[NoiseSchedule] = None,
        weight_property: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            name: name of the output.
            weight_fn: the weight function to weight each sample based on target[weight_property].
            weight_property: the property name to use for the weight function.
        """
        super().__init__(**kwargs)

        self.weight_property = weight_property
        self.permutation_invariant = permutation_invariant
        self.rotation_invariant = rotation_invariant
        self.weight_property = weight_property
        self.noise_schedule = noise_schedule
        if weight_property is not None and noise_schedule is None:
            raise Exception("weight_property given but noise_schedule is None")

    def calculate_loss(
        self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        calculate the loss.

        Args:
            pred: outputs.
            target: target values.
        """
        if self.loss_weight == 0 or self.loss_fn is None:
            return torch.tensor(0.0)

        # extract the extra arguments of the loss function if needed
        args_ = inspect.getfullargspec(self.loss_fn).args[2:]
        kwargs = {k: pred[k] for k in args_ if k in pred}

        if self.permutation_invariant:
            permutation = find_optimal_permutation(pred, target)
            target[self.target_property] = target[self.target_property][permutation]

        if self.rotation_invariant:
            target[properties.R] = rotate_optimally(pred, target)

        loss_samplewise = self.loss_fn(
            pred[self.name], target[self.target_property], **kwargs
        )

        if self.weight_property is not None:
            weights = self.noise_schedule(target.t, [self.weight_property])[
                self.weight_property
            ]
            loss_samplewise *= weights

        loss = torch.mean(loss_samplewise)

        return self.loss_weight * loss


class LRSOutput(ModelOutput):
    """
    define "Learned Representation Similarity" output.
    Inspired  by "LPIPS" this output works with molecule representations of a pretrained model.
    """

    def __init__(
        self,
        source_model: Union[NeuralNetworkPotential, str],
        name: str,
        representation_name: Literal[
            "vector_representation", "scalar_representation"
        ] = "scalar_representation",
        **kwargs,
    ):
        """
        Args:
            model: the model to use for the representation.
        """
        super().__init__(name=name, **kwargs)
        self.representation_name = representation_name

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load the model if given as a path
        if isinstance(source_model, str):
            source_model = torch.load(source_model, map_location=device)

        # construct the representation for loss calulation by replacing the output_modules with an identity layer
        model_output = nn.Identity()
        model_output.to(device)
        model_output.model_outputs = ["vector_representation", "scalar_representation"]

        self.representation = NeuralNetworkPotential(
            input_modules=source_model.input_modules,
            representation=source_model.representation,
            output_modules=nn.ModuleList([model_output]),
        )

        # freeze the representation
        for param in self.representation.parameters():
            param.requires_grad = False

    def calculate_loss(
        self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        calculate the loss.

        Args:
            pred: outputs.
            target: target values.
        """
        # map model output to positions
        pred[properties.R] = pred[self.name]
        target[properties.R] = target[self.name]

        # compute pred and target representation
        pred_representation = self.representation.forward(pred)
        with torch.no_grad():
            target_representation = self.representation.forward(target)

        # calculate the loss
        loss = self.loss_fn(
            pred_representation[self.representation_name],
            target_representation[self.representation_name],
        )

        return loss



class NormRegularizer(UnsupervisedModelOutput):
    def __init__(
        self,
        limit: float = 0.0,
        metrics: Optional[dict] = None,
        **kwargs,
    ):
        """
        Args:
            name: name of the output.
            weight_fn: the weight function to weight each sample based on target[weight_property].
            weight_property: the property name to use for the weight function.
        """

        self.limit = limit

        if metrics is None:
            metrics = {}

        super().__init__(metrics=metrics, **kwargs)

    def calculate_loss(self, pred, target=None):
        if self.limit == 0.0:
            return 0.0
        # average the norm over all atoms
        pred_magnitude = pred[self.name].norm(dim=1).mean()
        # truncate the loss at lower_limit
        penalty = nn.functional.relu(self.limit - pred_magnitude)

        if self.loss_fn is None:
            return penalty

        return self.loss_fn(penalty)


class ConsitencyTask(AtomisticTask):
    """
    Defines the consitency task for pytorch lightning as proposed by Song et al 2021.
    """

    def __init__(
        self,
        model: nn.Module,
        reverse_process: ReverseProcess,
        time_key: str = "t",
        x_t_key: str = "_positions",
        ema_decay=0.9999,
        caster=CastTo32(),
        cutoff: float = 500.0,
        recompute_neighbors: bool = True,
        skip_exploding_batches: bool = True,
        loss_limit: float = 5e-2,
        skip_referenceless_batches: bool = True,
        initialize_with_denoiser: bool = False,
        diffusion_schedule: Optional[DiffusionSchedule] = None,
        **kwargs,
    ):
        """
        Args:
            time_key: key of the true diffusion time step in the input dictionary.
            x_t_key: key of the sampled positions at time t.
        """
        super().__init__(model=model, **kwargs)

        self.time_key = time_key
        self.x_t_key = x_t_key
        self.ema_decay = ema_decay
        self.reverse_process = reverse_process
        self.caster = caster
        self.cutoff = cutoff
        self.loss_limit = loss_limit
        self.recompute_neighbors = recompute_neighbors
        self.skip_exploding_batches = skip_exploding_batches
        self.skip_referenceless_batches = skip_referenceless_batches
        self.diffusion_schedule = diffusion_schedule

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if initialize_with_denoiser:
            model.source_model.load_state_dict(self.reverse_process.denoiser.state_dict())

        # create target and online model
        self.model = model
        self.model.to(device=device)
        self.ema = EMA(model.parameters(), decay=ema_decay)

        self.normalize_time = self.reverse_process.diffusion_process.normalize_time
        self.unnormalize_time = self.reverse_process.diffusion_process.unnormalize_time

    def setup(self, stage=None):
        """
        overwrite the pytorch lightning task setup function.
        """
        # call the parent atomistic task setup
        AtomisticTask.setup(self, stage=stage)  # type: ignore

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        forward pass through the target model

        Args:
            inputs: input batch.
        """
        # calculate the forward pass for the target model
        with torch.no_grad(), self.ema.average_parameters():
            pred = self.model(batch)

        return {k: v.detach() for k, v in pred.items()}

    def forward_online(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        forward pass through the online model

        Args:
            inputs: input batch.
        """
        # calculate the forward pass for the online model
        pred = self.model(batch)

        return pred

    def _batch_hat(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        generate the batch hat for the online model by appyling the reverse ODE.

        Args:
            batch: input batch.
        """
        input = {k: v.clone() for k, v in batch.items()}
        batch_hat = {k: v.clone() for k, v in batch.items()}

        dtype = batch_hat[self.x_t_key].dtype

        # calculate t_next in normalized space
        t_1 = self.normalize_time(torch.tensor(1))
        t_next = batch_hat[self.time_key] - t_1

        # take one step of the reverse ODE for every t > 1
        x_t_next = self.reverse_process(input).to(dtype=dtype)

        # for all t=0 use the original positions
        x_t_next[t_next == 0] = batch_hat[f"original_{self.x_t_key}"][t_next == 0]

        # update batch_hat
        batch_hat[self.x_t_key] = x_t_next
        batch_hat[self.time_key] = t_next

        if self.recompute_neighbors:
            batch_hat.update(
                compute_neighbors(batch_hat, cutoff=self.cutoff, device=self.device)
            )

        return batch_hat

    def _step(
        self, batch: Dict[str, torch.Tensor], subset
    ) -> Optional[torch.FloatTensor]:
        """
        evaluate the target model.

        Args:
            batch: input batch.
            batch_idx: batch index.
        """
        batch_size = batch[properties.n_atoms].shape[0]
        is_train = subset == "train"

        # get batch_hat which is one step down the probability flow
        batch_hat = self._batch_hat(batch)

        # forward pass of x_t-1 with the target model
        target = self.forward(batch_hat)

        # forward pass of x_t with either online or target model depending on the step
        if is_train:
            pred = self.forward_online(batch)
        else:
            pred = self.forward(batch)

        # calculate the loss between online and target prediction
        loss = self.loss_fn(pred, target)

        # logging
        self.log(
            f"{subset}_loss",
            loss,
            on_step=is_train,
            on_epoch=not is_train,
            prog_bar=not is_train,
            batch_size=batch_size,
        )
        self.log_metrics(pred, target, subset)

        with torch.no_grad():
            target_magnitude = target["x_0_pred"].norm(dim=1).mean()
            self.log(
                "target_magnitude",
                target_magnitude,
                on_epoch=True,
                batch_size=batch_size,
            )

            pred_magnitude = pred["x_0_pred"].norm(dim=1).mean()
            self.log(
                "pred_magnitude", pred_magnitude, on_epoch=True, batch_size=batch_size
            )

            x_delta = batch[properties.R] - batch_hat[properties.R]
            x_delta_magnitude = x_delta.norm(dim=1).mean()
            self.log("x_delta_magnitude", x_delta_magnitude, on_epoch=False, on_step=True, batch_size=batch_size)

            # data_magnitude = batch["original__positions"].norm(dim=1).mean()
            # self.log("data_magnitude", data_magnitude, on_epoch=True, batch_size=batch_size)

        return loss

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Optional[torch.FloatTensor]:
        """
        define the training step for pytorch lightning.

        Args:
            batch: input batch.
            batch_idx: batch index.
        """
        t = self.unnormalize_time(batch[self.time_key])
        if self.skip_referenceless_batches and not (t == 1).any().item():
            log.warning(
                f"Training batch is without original reference batch_idx {batch_idx} and training step "
                f"{self.global_step}, training step will be skipped!"
            )
            return None

        loss = self._step(batch, "train")

        # skip exploding batches in backward pass
        if self.skip_exploding_batches and (
            torch.isnan(loss) or torch.isinf(loss) or loss > self.loss_limit
        ):
            log.warning(
                f"Loss is {loss} for train batch_idx {batch_idx} and training step "
                f"{self.global_step}, training step will be skipped!"
            )
            return None
        
        if self.diffusion_schedule is not None:
            self.diffusion_schedule.increase_step()

        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Optional[torch.FloatTensor]:
        """
        define the validation step for pytorch lightning.

        Args:
            batch: input batch.
            batch_idx: batch index.
        """
        loss = self._step(batch, "val")

        return {"val_loss": loss}

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Optional[torch.FloatTensor]:
        """
        define the test step for pytorch lightning.

        Args:
            batch: input batch.
            batch_idx: batch index.
        """
        loss = self._step(batch, "test")

        return {"test_loss": loss}

    def on_train_batch_end(self, *args):
        # update the target model with the new online model parameters
        self.ema.update()
        return super().on_train_batch_end(*args)
