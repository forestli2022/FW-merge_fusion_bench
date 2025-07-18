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
import numpy as np
import functools
import gc

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
from fusion_bench.models.wrappers.layer_wise_fusion import (
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


def projection_simplex_sort(v, z=1):
    # print(v.shape)
    n_features = v.shape[0]  # Get the number of elements in v
    u, _ = torch.sort(v, descending=True)  # Sort v in descending order
    cssv = torch.cumsum(u, dim=0) - z  # Compute cumulative sum and subtract z
    ind = torch.arange(1, n_features + 1, dtype=torch.long, device=v.device)  # Create index tensor (1 to n_features)
    cond = u - cssv / ind > 0  # Condition to find rho
    if cond.any():  # Ensure there is at least one valid rho
        rho = ind[cond][-1]  # Find the largest index satisfying the condition
        theta = cssv[rho - 1] / rho  # Compute the correct threshold theta
    else:
        theta = 0  # Default case when all values are zero or negative
    w = torch.clamp(v - theta, min=0)  # Compute the projected vector, ensuring non-negativity
    return w


@torch.no_grad()
def task_arithmetic_merge(
    pretrained_model: nn.Module,
    finetuned_models: List[Dict[str, Tensor]],
    scaling_factor: float,
    inplace: bool = True,
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
    if not inplace:
        pretrained_model = deepcopy(pretrained_model)
    if isinstance(finetuned_models[0], nn.Module):
        finetuned_models = [deepcopy(model.state_dict(keep_vars=True)) for model in finetuned_models]
    task_vector: StateDictType = None
    # Calculate the total task vector
    for model in finetuned_models:
        if task_vector is None:
            task_vector = state_dict_sub(
                model,
                pretrained_model.state_dict(keep_vars=True),
            )
        else:
            task_vector = state_dict_add(
                task_vector,
                state_dict_sub(
                    model,
                    pretrained_model.state_dict(keep_vars=True),
                ),
            )
    # scale the task vector
    task_vector = state_dict_mul(task_vector, scaling_factor)
    # add the task vector to the pretrained model
    state_dict = state_dict_add(
        pretrained_model.state_dict(keep_vars=True), task_vector
    )
    pretrained_model.load_state_dict(state_dict)
    return pretrained_model


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


class FrankWolfeSoftLossApproxAlgorithm(
    CLIPClassificationMixin,
    ModelFusionAlgorithm,
    SimpleProfilerMixin,
):
    def __init__(self, 
                 max_iters: int,
                 dataset_size:int,
                 ada_iters: int,
                 ada_coeff: float,
                 merge_fn: str,
                 granularity: str = "task",
                 max_num_models: int = 100,
                 step_size: float = 0.3,
                 tasks: List[str] = [],
                 init_weight: str = "",
                 ada_loss = "entropy_loss",
                 **kwargs):
        """
        Initializes the TaskArithmeticAlgorithm with the given scaling factor.

        Args:
            step_size (int): The factor by which the task vectors will be scaled before merging.
        """
        self.merge_fn = merge_fn
        
        self.init_weight = init_weight
        self.max_iters = max_iters
        self.ada_iters = ada_iters
        self.ada_coeff = ada_coeff
        self.granularity = granularity
        self.tasks = tasks
        self.step_size = step_size
        self.dataset_size = dataset_size
        self.max_num_models = max_num_models
        self.ada_loss = ada_loss
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
                k: (model_sd[k].to("cpu") - pretrained_sd[k].to("cpu")).detach() for k in filtered_keys
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
            projection[layer_name] = layer_proj # Projection matrix for each layer

        for m in finetuned_models:
            m.requires_grad_(True)
        pretrained_model = pretrained_model.requires_grad_(True)
        return projection, task_vectors

    @functools.cache
    def get_shuffled_train_loader_iter(self, task: str, batch_size: int = 1):
        # get dataloader kwargs
        dataloader_kwargs = self._dataloader_kwargs.copy()
        dataloader_kwargs["shuffle"] = True
        dataloader_kwargs["batch_size"] = batch_size

        # get the test dataset
        clip_dataset = CLIPDataset(
            self.modelpool.load_train_dataset(task), self.clip_processor
        )
        # create the dataloader
        loader = DataLoader(clip_dataset, **dataloader_kwargs)
        loader = self.fabric.setup_dataloaders(loader)
        return iter(InfiniteDataLoader(loader))
    

    @functools.cache
    def get_shuffled_test_loader_iter(self, task: str, batch_size: int = 1):
        return super().get_shuffled_test_loader_iter(
                task,
                batch_size=batch_size
        )
    
    
    def run_adamerging(self, module: LayerWiseMergedModel[TorchModelType]):
        use_entropy_loss = self.ada_loss == 'entropy_loss'
        
        optimizer = torch.optim.Adam(
            [module.merge_weight], lr=1e-3
        )
        module, optimizer = self.fabric.setup(module, optimizer)
        module.train()
        for step_idx in (
            pbar := tqdm(
                range(self.ada_iters),
                "AdaMerging (2/2)",
                dynamic_ncols=True,
                disable=not self.fabric.is_global_zero,
            )
        ):
            with self.profile("merge weights"):
                module.merge_weights()

            metrics = {}
            total_loss = None
            tasks = self.modelpool.model_names if self.tasks == [] else self.tasks
            if not use_entropy_loss:
                loss_fn = nn.CrossEntropyLoss()
            for task in tasks:
                with self.profile("data loading"):
                    if use_entropy_loss:
                        batch = next(self.get_shuffled_test_loader_iter(task, batch_size=16))
                    else:
                        batch = next(self.get_shuffled_train_loader_iter(task, batch_size=16))
                        # NOTE: The labels are not allowed to be used during test-time adaptation
                    images = batch[0]
                with self.profile("forward pass"):
                    logits = self.compute_logits(module, images, task)
                    if use_entropy_loss:
                        loss = entropy_loss(logits)
                    else:
                        loss = loss_fn(logits, batch[1])
                    total_loss = loss if total_loss is None else total_loss + loss
            optimizer.zero_grad()
            with self.profile("compute grad"):
                self.fabric.backward(total_loss)

            with self.profile("base optimizer step"):
                optimizer.step()

            metrics.update({"train/loss": loss.item()})
            self.fabric.log_dict(metrics, step=step_idx)
            pbar.set_postfix(metrics)
        return module
    
    
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

            # print(f"Layer: {layer_name}, DoGE Loss: {losses[layer_name].item()}")
            # calculate the gradients
            losses[layer_name].backward(retain_graph=False)
            g = sd[layer_name].grad.clone().to("cpu")
            g = (g - self.projection[layer_name] @ g)
            gradients[layer_name] = g.to("cpu")
            sd[layer_name].grad = None
            del part_1, part_2, inner_product, result
            torch.cuda.empty_cache()
            
        
        # calculate the loss
        avg_loss = sum(losses.values()) / len(self.task_vectors)
        log.info(f"Average Loss: {avg_loss}, Total Loss: {sum(losses.values())}")
        del losses

        for name, param in merged_model.named_parameters():
            param.grad = None
        merged_model.eval()
        
        return gradients

    def frank_wolfe_selection(self, gradients, checkpoints, model_to_merge_names=[], type='task', num_models=4):
        # min_models: list of min_model_dicts; min_model_names: list of model names; min_inner_products: list of inner products; log_dicts: dict of inner products per model
        assert type in ['task', 'layer'], f"Unsupported FW selection type: {type}, supported types are ['task', 'layer']"

        inner_products = []
        models = []
        model_names = []
        log_dict = {}
        if type == 'task':
            for model_name, model_to_merge in checkpoints.items():
                model_to_merge = model_to_merge.to('cpu').state_dict()
                inner_product_sum = 0
                for param_name, param_value in model_to_merge.items():
                    # caclulate consine similarity
                    if param_name not in gradients:
                        continue
                    grad = gradients[param_name]
                    ckpt = model_to_merge[param_name]
                    param_alignment = torch.dot(grad.flatten(), ckpt.flatten()) / (torch.norm(grad) * torch.norm(ckpt))
                    inner_product_sum += param_alignment

                inner_products.append(inner_product_sum)
                models.append(model_to_merge)
                model_names.append(model_name) 
                log_dict[model_name] = inner_product_sum.item()

                # if inner_product_sum < min_inner_product and model_name not in model_to_merge_names:
                #     min_inner_product = inner_product_sum
                #     min_model = deepcopy(model_to_merge)
                #     min_model_name = model_name
            # get smallest k model indices
            min_inner_products = []
            min_models = []
            min_model_names = []
            arr = np.array(inner_products)
            num_models = 4
            indices = np.argpartition(arr, num_models)[:num_models]
            for i in indices:
                min_inner_products.append(arr[i])
                min_models.append(deepcopy(models[i]))
                min_model_names.append(model_names[i])
            print("models: ", min_model_names)
        else:
            param_candidates = defaultdict(list)  # param_name -> list of (cos_sim, model_name, tensor)

            # Collect all cosine similarities for each layer
            for model_name, model_to_merge in checkpoints.items():
                model_to_merge = model_to_merge.to('cpu').state_dict()
                for param_name, param_value in model_to_merge.items():
                    if param_name not in gradients:
                        grad = torch.zeros_like(param_value)
                    else:
                        grad = gradients[param_name]
                    ckpt = param_value
                    denom = torch.norm(grad) * torch.norm(ckpt)
                    if denom == 0:
                        param_alignment = torch.tensor(float(0))
                    else:
                        param_alignment = torch.dot(grad.flatten(), ckpt.flatten()) / denom
                    param_candidates[param_name].append((param_alignment.item(), model_name, param_value))

            # Select top-k for each layer
            min_models = {} # list of dicts: one dict per selected model (for each param)
            min_model_names = defaultdict(list) # list of lists: one list per param, containing top-k model names
            min_inner_products = defaultdict(list) # list of lists: one list per param, containing top-k similarities
            log_dict = {model_name: 0 for model_name in checkpoints.keys()}

            for param_name, candidates in param_candidates.items():
                # Sort by lowest cosine similarity
                sorted_candidates = sorted(candidates, key=lambda x: x[0])
                top_k = sorted_candidates[:num_models]

                for cos_sim, model_name, param_tensor in top_k:
                    min_models.setdefault(param_name, []).append(param_tensor)
                    min_model_names[param_name].append(model_name)
                    min_inner_products[param_name].append(cos_sim)
                    log_dict[model_name] += 1
        
        return min_models, min_model_names, min_inner_products, log_dict
    


    def run(self, modelpool: HuggingFaceClipVisionPool):
        log.info("Fusing models using FW merging.")
        self.modelpool = modelpool
        tasks = self.tasks if self.tasks else self.modelpool.model_names
        self.log_hyperparams(self.config)
        self.on_frank_wolfe_iteration_start()

        assert modelpool.has_pretrained, "Pretrained model is required."
        finetuned_models = {name: modelpool.load_model(name) for name in modelpool.model_names[:self.max_num_models]}
        pretrained_model = modelpool.load_model("_pretrained_")
        
        if self.init_weight == 'base' or self.init_weight == '':
            merged_model = modelpool.load_model("_pretrained_")
        else:
            log.info("Initializing the merged model with the initial weight")
            if isinstance(self.init_weight, str):
                # self.config.weights is a path to a saved tensor
                layer_wise_weight = load_tensor_from_file(self.init_weight)
            else:
                raise ValueError(f"Unsupported weights format: {self.init_weight}")

            layerwise_merged_model = LayerWiseMergedModel(
                layer_wise_weight=layer_wise_weight,
                pretrained_model=pretrained_model,
                finetuned_models=list(finetuned_models.values())[:self.max_num_models],
                clamp_weights=False,
                tie_weights=True,
                strict=False,
            ).cuda()
            merged_model = layerwise_merged_model.merge_and_unload()

        self.initial_model = modelpool.load_model("_pretrained_")
        self.set_requires_grad(merged_model, self.initial_model)
        # initial_model.load_state_dict(deepcopy(merged_model.state_dict()))
        # finetuned_models['initial'] = initial_model

        # calculate projection
        task_models = finetuned_models.values()
        self.projection, self.task_vectors = self.calculate_projection(pretrained_model, task_models)
        for step_idx in (
            pbar := tqdm(
                range(self.max_iters if not self.is_debug_mode else 1),
                ("[DEBUG MODE] " if self.is_debug_mode else "")
                + "Frank-Wolfe Merging",
                dynamic_ncols=True,
            )
        ):
            torch.cuda.empty_cache()
            # Find the task vector with the most alignment to the gradient
            models_dict_to_merge = []
            model_to_merge_names = [] if self.granularity == 'task' else {name: [] for name in merged_model.state_dict().keys()}
            inner_products = []

            torch.set_grad_enabled(True)
            torch.cuda.empty_cache()
            # calculate gradient once, loss is global
            gradients = self.frank_wolfe_iteration(merged_model.cuda())
            torch.set_grad_enabled(False)
            grad_norm = torch.norm(torch.stack([torch.norm(g) for g in gradients.values()]))
            
            # select number of tasks of models
            # min_models: list of min_model_dicts; min_model_names: list of model names; min_inner_products: list of inner products; log_dict: dict of inner products per model
            min_models, min_model_names, min_inner_products, log_dict = self.frank_wolfe_selection(gradients, finetuned_models, model_to_merge_names, type=self.granularity, num_models=len(tasks))
            if self.granularity == 'task':
                model_to_merge_names = min_model_names
            else:
                for model_i in min_model_names:
                    for param_name, model_name in zip(gradients.keys(), model_i):
                        model_to_merge_names[param_name].append(model_name)
            models_dict_to_merge = min_models
            inner_products = min_inner_products

            for task in tasks:
                log.info(f"Task: {task}, Inner Products: {log_dict[task]}")

            
            # print iteration information
            log.info(f"Iteration {step_idx+1}, Task Vector: {model_to_merge_names}, Gradient Norm: {grad_norm:.6f}, Inner Products: {inner_products}")
            
            if self.merge_fn == 'adamerging':
                models_to_merge = [modelpool.load_model('_pretrained_').to("cpu") for _ in range(len(models_dict_to_merge))]
                layer_wise_weight = get_layer_wise_weights(
                    num_models=len(models_to_merge),
                    num_layers=len(
                        tuple(
                            filter(lambda p: p.requires_grad, models_to_merge[0].parameters())
                        )
                    ),
                    init_values=self.ada_coeff if step_idx > 0 else 0.3,
                )
                for model_to_merge, model_to_merge_dict in zip(models_to_merge, models_dict_to_merge):
                    model_to_merge.load_state_dict(model_to_merge_dict)
                layerwise_merged_model = LayerWiseMergedModel(
                    layer_wise_weight=layer_wise_weight.to("cpu"),
                    pretrained_model=merged_model.to("cpu"),
                    finetuned_models=models_to_merge,
                    clamp_weights=False,
                    tie_weights=True,
                    strict=False,
                ).cuda()
                torch.cuda.empty_cache()
                torch.set_grad_enabled(True)
                layerwise_merged_model = self.run_adamerging(layerwise_merged_model)
                torch.set_grad_enabled(False)
                with torch.no_grad():
                    merged_model = layerwise_merged_model.merge_and_unload()
                    self.set_requires_grad(merged_model, self.initial_model)
                del models_to_merge, layerwise_merged_model, layer_wise_weight, models_dict_to_merge
                torch.cuda.empty_cache()
            else:
                step = 2 / (step_idx + 2) * self.step_size if step_idx > 0 else 1
                merged_model = task_arithmetic_merge(merged_model.to('cpu'), models_dict_to_merge, 0.3*step)
                del models_dict_to_merge
                
        torch.set_grad_enabled(False)
        merged_model = merged_model.cuda().eval()
        return merged_model

    def set_requires_grad(self, merged_model, initial_model):
        for name, param in initial_model.named_parameters():
            for n, p in merged_model.named_parameters():
                if name == n:
                    p.requires_grad = param.requires_grad
