"""
This script contains the general implementation of the Task Arithmetic method.

http://arxiv.org/abs/2212.04089
"""

import logging
import os
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, List, Mapping, TypeVar, Union, Dict
from copy import deepcopy
from collections import defaultdict
from functools import partial
import functools

from .utils import *

import torch
from lightning.fabric.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from fusion_bench.compat.method import ModelFusionAlgorithm
from fusion_bench.dataset.clip_dataset import CLIPDataset
from fusion_bench.mixins import CLIPClassificationMixin
from fusion_bench.compat.modelpool import ModelPool, HuggingFaceClipVisionPool
from fusion_bench.mixins.lightning_fabric import LightningFabricMixin
from fusion_bench.mixins.simple_profiler import SimpleProfilerMixin
from fusion_bench.models.wrappers.layer_wise_fusion_fw_am import (
    LayerWiseMergedModel,
    get_layer_wise_weights,
)
from fusion_bench.utils.data import load_tensor_from_file
from fusion_bench.utils.type import TorchModelType

if TYPE_CHECKING:
    from fusion_bench.programs.fabric_fusion_program import FabricModelFusionProgram

from fusion_bench.mixins.simple_profiler import SimpleProfilerMixin
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.utils.data import InfiniteDataLoader
from fusion_bench.utils.state_dict_arithmetic import (
    state_dict_add,
    state_dict_mul,
    state_dict_sub,
)
from fusion_bench.utils.type import StateDictType
from fusion_bench.utils import instantiate

log = logging.getLogger(__name__)


# @torch.no_grad()
# def task_arithmetic_merge(
#     pretrained_model: nn.Module,
#     finetuned_models: List[Dict[str, Tensor]],
#     scaling_factor: float,
#     inplace: bool = True,
# ) -> nn.Module:
#     """
#     Merges the task vectors from multiple fine-tuned models into a single pre-trained model.

#     Args:
#         pretrained_model (nn.Module): The pre-trained model to which the task vectors will be added.
#         finetuned_models (List[nn.Module]): A list of fine-tuned models from which task vectors will be calculated.
#         scaling_factor (float): A factor by which the task vectors will be scaled before merging.
#         inplace (bool, optional): If True, the pre-trained model will be modified in place.
#                                   If False, a copy of the pre-trained model will be modified. Defaults to True.

#     Returns:
#         nn.Module: The pre-trained model with the merged task vectors.
#     """
#     if not inplace:
#         pretrained_model = deepcopy(pretrained_model)
#     if isinstance(finetuned_models[0], nn.Module):
#         finetuned_models = [deepcopy(model.state_dict(keep_vars=True)) for model in finetuned_models]
#     task_vector: StateDictType = None
#     # Calculate the total task vector
#     for model in finetuned_models:
#         if task_vector is None:
#             task_vector = state_dict_sub(
#                 model,
#                 pretrained_model.state_dict(keep_vars=True),
#             )
#         else:
#             task_vector = state_dict_add(
#                 task_vector,
#                 state_dict_sub(
#                     model,
#                     pretrained_model.state_dict(keep_vars=True),
#                 ),
#             )
#     # scale the task vector
#     task_vector = state_dict_mul(task_vector, scaling_factor)
#     # add the task vector to the pretrained model
#     state_dict = state_dict_add(
#         pretrained_model.state_dict(keep_vars=True), task_vector
#     )
#     pretrained_model.load_state_dict(state_dict)
#     return pretrained_model


# @torch.no_grad()
# def ties_merge(
#     pretrained_model: nn.Module,
#     finetuned_models: List[Dict[str, Tensor]],
#     scaling_factor: float,
#     threshold: float,
# ) -> nn.Module:    
#     remove_keys = []
#     merge_func = "sum"
#     if isinstance(finetuned_models[0], nn.Module):
#         finetuned_models = [deepcopy(model.state_dict(keep_vars=True)) for model in finetuned_models]

#     ptm_check = pretrained_model.state_dict(keep_vars=True)

#     # Compute the task vectors
#     flat_ft = torch.vstack(
#             [state_dict_to_vector(check, remove_keys) for check in finetuned_models]
#         )
#     flat_ptm = state_dict_to_vector(ptm_check, remove_keys)
#     tv_flat_checks = flat_ft - flat_ptm

#     # Perform TIES Merging
#     merged_tv = ties_merging(
#         tv_flat_checks,
#         reset_thresh=threshold,
#         merge_func=merge_func,
#     )
#     merged_check = flat_ptm + scaling_factor * merged_tv
#     merged_state_dict = vector_to_state_dict(
#         merged_check, ptm_check, remove_keys=remove_keys
#     )

#     # Load the merged state dict into the pretrained model
#     pretrained_model.load_state_dict(merged_state_dict)
#     return pretrained_model

@torch.no_grad()
def task_arithmetic_merge(
    merged_model: LayerWiseMergedModel,
    finetuned_models: List[Dict[str, Tensor]],
    indices: List[int] = None,
    scaling_factor: float = 1.0,
) -> nn.Module:
    """
    Merges the task vectors from multiple fine-tuned models into a single pre-trained model.

    Args:
        pretrained_model (nn.Module): The pre-trained model to which the task vectors will be added.
        finetuned_models (List[nn.Module]): A list of fine-tuned models from which task vectors will be calculated.
        scaling_factor (float): A factor by which the task vectors will be scaled before merging.
        inplace (bool, optional): If True, the pre-trained model will be modified in place.
                                  If False, a copy of the pre-trained model will be modified. Defaults to True.

    Returns:
        nn.Module: The pre-trained model with the merged task vectors.
    """
    print("shapes: ", merged_model.merge_weight.shape, len(indices))
    print(indices)
    # Directly edit merge_weight of the merged model
    for l, model_index in enumerate(indices):
        merged_model.merge_weight[model_index, l] += scaling_factor
    merged_model.merge_weights()
    return merged_model

def entropy_loss(logits: Tensor, pred = None, eps: float = 1e-8) -> Tensor:
    """
    Compute the entropy loss of a set of logits.

    Args:
        logits (Tensor): The logits to compute the entropy loss of.
        eps (float): A small value to avoid log(0). Default is 1e-8.

    Returns:
        Tensor: The entropy loss of the logits.
    """
    # Ensure the logits tensor has 2 dimensions
    assert (
        logits.dim() == 2
    ), f"Expected logits to have 2 dimensions, found {logits.dim()}, {logits.size()=}"

    # Compute the softmax probabilities
    probs = torch.softmax(logits, dim=-1)

    # Compute the entropy loss
    return -torch.sum(probs * torch.log(probs + eps), dim=-1).mean()


class FrankWolfeHardAdamergingLossApproxAlgorithm(
    CLIPClassificationMixin,
    ModelFusionAlgorithm,
    SimpleProfilerMixin,
):


    def __init__(self, 
                 merge_fn: str,
                 step_size: float,
                 max_iters: int,
                 dataset_size:int,
                 tasks: List[str] = [],
                 granularity: str = 'task',
                 max_num_models: int = 100,
                 loss_fn: str = "cross_entropy",
                 init_weight: str = "",
                 scaling_factor: float = 1.,
                 threshold: int = 20,
                 init_layer_weights: float = None,
                 ada_merge: bool = True,
                 ada_max_steps: int = 500,
                 ada_optimizer: str = "adam",
                 ada_lr: float = 1e-3,
                 **kwargs):
        """
        Initializes the TaskArithmeticAlgorithm with the given scaling factor.

        Args:
            scaling_factor (int): The factor by which the task vectors will be scaled before merging.
        """
        # self.merger = merge_fn
        # if merge_fn == "task_arithmetic":
        #     self.merge_fn = task_arithmetic_merge
        # elif merge_fn == "ties":
        #     self.merge_fn = partial(ties_merge, threshold=threshold)
        # # elif merge_fn == "concrete_ta":
        # #     self.merge_fn = ConcreteTaskArithmeticAlgorithmForCLIP(
        # #         instantiate(OmegaConf.load("config/method/concrete_subspace/clip_concrete_task_arithmetic.yaml"))
        # #     )
        # else:
        #     raise ValueError(f"Unsupported merge_fn: {merge_fn}")
        self.scaling_factor = scaling_factor
        
        self.init_weight = init_weight
        self.step_size = step_size
        self.max_iters = max_iters
        self.granularity = granularity
        self.loss_fn = loss_fn
        self.tasks = tasks
        self.dataset_size = dataset_size
        self.max_num_models = max_num_models

        self.ada_merge = ada_merge
        self.init_layer_weights = init_layer_weights
        self.ada_max_steps = ada_max_steps
        self.ada_optimizer = ada_optimizer
        self.ada_lr = ada_lr
        super().__init__(**kwargs)


    def on_frank_wolfe_iteration_start(self):
        self.setup_zero_shot_classification_head()
    
    def calculate_projection(self, pretrained_model: nn.Module, finetuned_models: List[nn.Module]):
        # Compute the svd and projection here
        pretrained_sd = pretrained_model.state_dict(keep_vars=True)
        filtered_keys = [
            k
            for k in pretrained_sd.keys()
            if ("encoder" in k and "layer_norm" not in k and "weight" in k)
        ]
        task_vectors = []
        for m in finetuned_models:
            m.requires_grad_(False)
        pretrained_model = pretrained_model.requires_grad_(False)
        for model in finetuned_models:
            model_sd = model.state_dict(keep_vars=True)
            filtered_task_vector = {
                k: (model_sd[k].to("cpu") - pretrained_sd[k].to("cpu")) for k in filtered_keys
            }
            task_vectors.append(filtered_task_vector)

        projection = {}
        for layer_name in task_vectors[0].keys():
            for i, vector in enumerate(task_vectors):
                layer_vector = vector[layer_name]
                u, s, v = torch.linalg.svd(layer_vector, full_matrices=False)
                if i == 0:
                    print(f"Computed SVD for {layer_name}...")
                    sum_u = torch.zeros_like(u, device=layer_vector.device)
                    sum_s = torch.zeros_like(s, device=layer_vector.device)
                    sum_v = torch.zeros_like(v, device=layer_vector.device)

                reduced_index_s = int(s.shape[0] / len(task_vectors))

                # select only the first reduced_index_s columns of u and place them
                sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[
                    :, :reduced_index_s
                ]
                sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[
                    :reduced_index_s
                ]
                # select only the first reduced_index_s rows of v and place them
                sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[
                    :reduced_index_s, :
                ]
            # SVD of shared subspace to avoid overlapping task vectors
            u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
            # u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)
            layer_proj = torch.matmul(
                u_u[:, : int(s.shape[0] / len(task_vectors))],
                u_u[:, : int(s.shape[0] / len(task_vectors))].T,
            )
            projection[layer_name] = layer_proj.to("cpu") # Projection matrix for each layer

        for m in finetuned_models:
            m.requires_grad_(True)
        pretrained_model = pretrained_model.requires_grad_(True)
        return projection, task_vectors

    @functools.cache
    def get_shuffled_loader_iter(self, task: str):
        if self.loss_fn == "cross_entropy":
            # get dataloader kwargs
            dataloader_kwargs = self._dataloader_kwargs.copy()
            dataloader_kwargs["shuffle"] = True
            dataloader_kwargs["batch_size"] = 1

            # get the test dataset
            clip_dataset = CLIPDataset(
                self.modelpool.load_train_dataset(task), self.clip_processor
            )
            # create the dataloader
            loader = DataLoader(clip_dataset, **dataloader_kwargs)
            loader = self.fabric.setup_dataloaders(loader)
            return iter(InfiniteDataLoader(loader))
        elif self.loss_fn == "entropy":
            return super().get_shuffled_test_loader_iter(
                task,
                batch_size=1,
            )
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_fn}")

    
    def frank_wolfe_iteration(self, merged_model):
        
        merged_model.train()
        # zero the gradients
        for name, param in merged_model.named_parameters():
            param.requires_grad = True
            param.grad = None
        sd = merged_model.state_dict(keep_vars=True)

        losses = defaultdict(list)
        gradients = {} 

        for layer_name in self.task_vectors[0].keys():
            task_layer_vectors = torch.stack([vec[layer_name] for vec in self.task_vectors])
            merged_model_layer_vector = sd[layer_name].to("cpu")
            initial_model_layer_vector = self.initial_model.state_dict(keep_vars=True)[layer_name].to("cpu")
            losses[layer_name] = 0.0
            for task_layer_vector in task_layer_vectors:
                # -layer_vector
                part_1 = -task_layer_vector
                # merged_model - layer_vector
                part_2 = merged_model_layer_vector - initial_model_layer_vector - task_layer_vector
                # dot product between part_1 and part_2
                inner_product = torch.sum(part_1 * part_2)
                result = inner_product * inner_product
                losses[layer_name] += result

            print(f"Layer: {layer_name}, DoGE Loss: {losses[layer_name].item()}")
            # calculate the gradients
            losses[layer_name].backward()
            g = sd[layer_name].grad.clone().to("cpu")
            g = g - self.projection[layer_name].to("cpu") @ g
            gradients[layer_name] = g
            sd[layer_name].grad = None
            
        
        # calculate the loss
        avg_loss = sum(losses.values()) / len(self.task_vectors)
        log.info(f"Average Loss: {avg_loss}, Total Loss: {sum(losses.values())}")

        for name, param in merged_model.named_parameters():
            param.grad = None
        merged_model.eval()
        
        return gradients


    def frank_wolfe_selection(self, gradients, checkpoints, model_to_merge_names={}, type='task'):
        assert type in ['task', 'layer'], f"Unsupported FW selection type: {type}, supported types are ['task', 'layer']"
        min_inner_product = float("inf")
        min_model = None 
        min_model_name = None
        log_dict = {}
        if type == 'task':
            for model_name, model_to_merge in checkpoints.items():
                model_to_merge = model_to_merge.to('cuda').state_dict()
                inner_product_sum = 0
                for param_name, param_value in model_to_merge.items():
                    # caclulate consine similarity
                    grad = (gradients[param_name] if param_name in gradients else torch.zeros_like(param_value)).to("cpu")
                    ckpt = model_to_merge[param_name].to("cpu")
                    param_alignment = torch.dot(grad.flatten(), ckpt.flatten()) / (torch.norm(grad) * torch.norm(ckpt))
                    inner_product_sum += param_alignment
                log_dict[model_name] = inner_product_sum.item()
                if inner_product_sum < min_inner_product and model_name not in model_to_merge_names:
                    min_inner_product = inner_product_sum
                    min_model = deepcopy(model_to_merge)
                    min_model_name = model_name
        else:
            min_model = {}
            min_inner_product = {}
            min_idx = {}
            min_model_name = {}
            for model_name, model_to_merge in checkpoints.items():
                model_to_merge = model_to_merge.to('cuda').state_dict()
                for param_name, param_value in model_to_merge.items():
                    # caclulate consine similarity
                    grad = (gradients[param_name] if param_name in gradients else torch.zeros_like(param_value)).to("cpu")
                    ckpt = model_to_merge[param_name].to("cpu")
                    param_alignment = torch.dot(grad.flatten(), ckpt.flatten()) / (torch.norm(grad) * torch.norm(ckpt))
                    if (param_name not in min_inner_product or param_alignment < min_inner_product[param_name]) and \
                            model_name not in model_to_merge_names[param_name]:
                        min_inner_product[param_name] = param_alignment
                        # if min_inner_product[param_name] < 0:
                        min_model[param_name] = param_value
                        min_idx[param_name] = model_name
                        min_model_name[param_name] = model_name
                        # else:
                            # min_model[param_name] = torch.zeros_like(param_value)
            min_inner_product = sum(min_inner_product.values())
            log_dict = {model_name: 0 for model_name in checkpoints.keys()}
            for k in min_idx.values():
                log_dict[k] += 1 
        
        return min_model, min_model_name, min_inner_product, log_dict


    def run_adamerging(self, module: "LayerWiseMergedModel[TorchModelType]"):
        # configure optimizer
        if self.ada_optimizer == "adam":
            optimizer = torch.optim.Adam([module.merge_weight], lr=self.ada_lr)
            print(f"{optimizer=}")
            module, optimizer = self.fabric.setup(module, optimizer)
        else:
            raise ValueError(f"Unsupported optimizer: {self.ada_optimizer}")

        module.train()
        module.merge_weights()
        for step_idx in (
            pbar := tqdm(
                range(self.ada_max_steps),
                "AdaMerging Test-time adaptation",
                dynamic_ncols=True,
            )
        ):
            # default behavior for first-order optimizers
            for task in self.tasks:
                with self.profile("data loading"):
                    batch = next(self.get_shuffled_test_loader_iter(task))
                with self.profile("forward pass"):
                    logits = self.compute_logits(module, batch[0], task)
                    loss = entropy_loss(logits)
                with self.profile("backward pass"):
                    self.fabric.backward(loss, retain_graph=True)

            with self.profile("optimizer step"):
                optimizer.step()
                optimizer.zero_grad()
            with self.profile("merging weights"):
                module.merge_weights()

            metrics = {
                "train/loss": loss.item(),
                "train/weight_max": module.merge_weight.max().item(),
                "train/weight_min": module.merge_weight.min().item(),
                "train/weight_mean": module.merge_weight.mean().item(),
            }
            self.fabric.log_dict(metrics, step=step_idx)
            pbar.set_postfix(metrics)

        self.print_profile_summary()
        return module


    def run(self, modelpool: HuggingFaceClipVisionPool):
        log.info("Fusing models using FW merging.")
        self.modelpool = modelpool
        # self.log_hyperparams(self.config)
        self.on_frank_wolfe_iteration_start()

        assert modelpool.has_pretrained, "Pretrained model is required."
        finetuned_models = {name: modelpool.load_model(name) for name in modelpool.model_names[:self.max_num_models]}
        pretrained_model = modelpool.load_model("_pretrained_")
        
        if self.init_weight:
            if self.init_weight == 'base':
                log.info("Initializing the merged model with the base model")
                merged_model = pretrained_model
            else:
                log.info("Initializing the merged model with the initial weight")
                if isinstance(self.init_weight, str):
                    # self.config.weights is a path to a saved tensor
                    layer_wise_weight = load_tensor_from_file(self.init_weight)
                else:
                    raise ValueError(f"Unsupported weights format: {self.init_weight}")

                merged_model = LayerWiseMergedModel(
                    layer_wise_weight=layer_wise_weight,
                    pretrained_model=modelpool.load_model("_pretrained_"),
                    finetuned_models=list(finetuned_models.values()),
                    clamp_weights=False,
                    tie_weights=True,
                    strict=False,
                )
                merged_model = merged_model.merge_and_unload()

            # Construct the layer-wise merged model instead
            self.initial_model = modelpool.load_model("_pretrained_")
            self.initial_model.load_state_dict(merged_model.state_dict())
            finetuned_models['initial'] = self.initial_model
            # Create layer-wise weights
            layer_wise_weight = get_layer_wise_weights(
                num_models=len(finetuned_models),
                num_layers=len(
                    tuple(
                        filter(lambda p: p.requires_grad, pretrained_model.parameters())
                    )
                ),
                init_values= self.init_layer_weights if self.init_layer_weights is not None else 0.0,
            )
            # Change the last row (initial model) to 1.0
            layer_wise_weight[-1, :] = 1.0
            # Create the merged model with the layer-wise weights
            with torch.no_grad():
                self.set_requires_grad(merged_model, self.initial_model)
            merged_model = LayerWiseMergedModel(
                layer_wise_weight=layer_wise_weight,
                pretrained_model=merged_model,
                finetuned_models=list(finetuned_models.values()),
                clamp_weights=False,
                tie_weights=True,
                strict=False,
            ).cuda()
        else:
            raise ValueError("`init_weight` must be base or specified in the config file.")

        
        self.projection, self.task_vectors = self.calculate_projection(pretrained_model, finetuned_models.values())
        # FW iteration
        for step_idx in (
            pbar := tqdm(
                range(self.max_iters if not self.is_debug_mode else 1),
                ("[DEBUG MODE] " if self.is_debug_mode else "")
                + "Frank-Wolfe Merging",
                dynamic_ncols=True,
            )
        ):
            torch.cuda.empty_cache()
            merged_model_tmp = merged_model.merge_and_copy().cuda()
            with torch.no_grad():
                self.set_requires_grad(merged_model_tmp, self.initial_model)
            torch.set_grad_enabled(True)
            gradients = self.frank_wolfe_iteration(merged_model_tmp)
            torch.set_grad_enabled(False)
            grad_norm = torch.norm(torch.stack([torch.norm(g) for g in gradients.values()]))

            model_to_merge_names = [] if self.granularity == 'task' else {name: [] for name in merged_model.state_dict().keys()}
            min_model, min_model_name, min_alignment, chosen_model = self.frank_wolfe_selection(gradients, finetuned_models, model_to_merge_names=model_to_merge_names, type=self.granularity)

            # Determine step size
            step = 2 / (step_idx + 2) * self.step_size
            
            # print iteration information
            log.info(f"Iteration {step_idx+1}, Task Vector: {min_model_name}, Gradient Norm: {grad_norm:.6f}, Inner Products: {min_alignment:.6f}, Chosen Model: {chosen_model}")
            
            # Calculate position of each min_model in the finetuned model lists in each layer
            if self.granularity == 'task':
                indices = [list(finetuned_models.keys()).index(min_model_name) for _ in range(len(merged_model.state_dict()))]
            else:
                indices = [list(finetuned_models.keys()).index(min_model_name[param_name]) for param_name in merged_model.state_dict().keys()]
            merged_model = task_arithmetic_merge(
                    merged_model=merged_model,
                    finetuned_models=[min_model],
                    indices=indices,
                    scaling_factor=step * self.scaling_factor, 
                )
    
        # Merge model with Adamerging
        if self.ada_merge:
            print("number of models to merge: ", len(modelpool.model_names))
            torch.set_grad_enabled(True)
            merged_model = self.run_adamerging(merged_model)
            torch.set_grad_enabled(False)

        with torch.no_grad():
            merged_model = merged_model.merge_and_unload()
            self.set_requires_grad(merged_model, self.initial_model)
        # eval and return model
        merged_model = merged_model.cuda().eval()
        return merged_model

    def set_requires_grad(self, merged_model, initial_model):
        for name, param in initial_model.named_parameters():
            for n, p in merged_model.named_parameters():
                if name == n:
                    p.requires_grad = param.requires_grad
