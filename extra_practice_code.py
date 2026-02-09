import torch

import torch.nn as nn


def initialize_tracking(model):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # Initialize a boolean tensor to track modified weights
            module.modified_weights = torch.zeros_like(module.weight, dtype=torch.bool)


# This Version Is Working On UnPruned Models
def modify_negative_weights(model):
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                weights = module.weight.data

                # Find all negative weights
                zero_weight_indices = (weights <= 0).nonzero(as_tuple=True)

                # Replace zero weights with the specified random weight
                if len(zero_weight_indices[0]) > 0:
                    random_weight = torch.tensor(50.0, device=weights.device).clone().detach()
                    weights.index_put_(zero_weight_indices, random_weight)


# It Modifies The Pruned Weights To 50
def modify_weights_zero(model):
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if hasattr(module, 'weight_mask') and hasattr(module, 'weight_orig'):
                    # Access weight_mask and weight_orig
                    weight_mask = module.weight_mask
                    weight_orig = module.weight_orig
                    weights = module.weight


                    # Find indices where weight_mask is zero (indicating pruned weights)
                    pruned_indices = (weight_mask == 0).nonzero(as_tuple=True)

                    # # Update the original weights at the pruned indices to 50
                    if len(pruned_indices[0]) > 0:
                        random_weight = torch.tensor(50.0, device=weights.device).clone().detach()
                        weight_orig.index_put_(pruned_indices, random_weight)
                        weights.index_put_(pruned_indices, random_weight)


def modify_retained_weights(model, percentage=0.2):
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if hasattr(module, 'weight_mask') and hasattr(module, 'weight_orig'):
                    # Access weight_mask and weight_orig
                    weight_mask = module.weight_mask
                    weight_orig = module.weight_orig
                    weights = module.weight

                    # Get indices where weight_mask is 1 (not pruned weights)
                    not_pruned_indices = (weight_mask == 1).nonzero(as_tuple=True)

                    # Get the values of not pruned weights
                    not_pruned_weights = weight_orig[not_pruned_indices]

                    # Separate positive and negative weights
                    p_weights = not_pruned_weights[not_pruned_weights > 0]
                    n_weights = not_pruned_weights[not_pruned_weights < 0]

                    # Calculate number of weights to select
                    num_weights_to_select = int(len(not_pruned_weights) * percentage / 2)

                    # Get top positive weights
                    top_positive_indices = torch.topk(not_pruned_weights, num_weights_to_select, largest=True).indices
                    # Get top negative weights
                    top_negative_indices = torch.topk(not_pruned_weights, num_weights_to_select, largest=False).indices

                    if len(top_positive_indices) > 0:
                        # sp_indices means smallest positive indices
                        sp_indices = torch.topk(p_weights, num_weights_to_select, largest=False).indices
                        # Ensure indices are within bounds
                        sp_indices = sp_indices[:min(num_weights_to_select, len(sp_indices))]
                        # Map back to original indices
                        sp_original_indices = (not_pruned_weights.unsqueeze(0) == p_weights[sp_indices].unsqueeze(1)).nonzero(as_tuple=True)
                        # Generate random positive values
                        random_positive_weights = torch.abs(torch.randn(num_weights_to_select, device=weights.device))

                        # Update the original positive weights using the random positive values
                        weight_orig.index_put_((not_pruned_indices[0][top_positive_indices], not_pruned_indices[1][top_positive_indices]), random_positive_weights)
                        weights.index_put_((not_pruned_indices[0][top_positive_indices], not_pruned_indices[1][top_positive_indices]), random_positive_weights)

                    if len(top_negative_indices) > 0:
                        # ln_indices means largest negative indices
                        ln_indices = torch.topk(n_weights, num_weights_to_select, largest=True).indices
                        # Ensure indices are within bounds
                        ln_indices = ln_indices[:min(num_weights_to_select, len(ln_indices))]
                        ln_original_indices = (not_pruned_weights.unsqueeze(0) == n_weights[ln_indices].unsqueeze(1)).nonzero(as_tuple=True)
                        # Generate random negative values
                        random_negative_weights = -torch.abs(torch.randn(num_weights_to_select, device=weights.device))

                        # Update the original negative weights using the random negative values
                        weight_orig.index_put_((not_pruned_indices[0][top_negative_indices], not_pruned_indices[1][top_negative_indices]), random_negative_weights)
                        weights.index_put_((not_pruned_indices[0][top_negative_indices], not_pruned_indices[1][top_negative_indices]),random_negative_weights)


def modify_pruned_weights(model, sparsity_amount):
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if hasattr(module, 'weight_mask') and hasattr(module, 'weight_orig'):
                    # Access weight_mask and weight_orig
                    weight_mask = module.weight_mask
                    weight_orig = module.weight_orig
                    weights = module.weight

                    # Step 1: Set random weights for the pruned weights and update the weight_mask
                    pruned_indices = (weight_mask == 0).nonzero(as_tuple=True)
                    if len(pruned_indices[0]) > 0:
                        random_weights = torch.randn(len(pruned_indices[0]), device=weights.device)
                        weight_orig.index_put_(pruned_indices, random_weights)
                        weights.index_put_(pruned_indices, random_weights)
                        weight_mask.index_put_(pruned_indices, torch.ones(len(pruned_indices[0]), device=weights.device))

                    # Step 2: Prune the smallest positive and largest negative weights
                    not_pruned_indices = (weight_mask == 1).nonzero(as_tuple=True)
                    not_pruned_weights = weight_orig[not_pruned_indices]

                    num_weights_to_prune = int(len(not_pruned_weights) * sparsity_amount)

                    if num_weights_to_prune > 0:
                        # Get indices of the smallest positive weights
                        positive_weights = not_pruned_weights[not_pruned_weights > 0]
                        if len(positive_weights) > 0:
                            num_positive_to_prune = min(num_weights_to_prune, len(positive_weights))
                            smallest_positive_indices = torch.topk(positive_weights, num_positive_to_prune, largest=False).indices
                            smallest_positive_original_indices = tuple(index[smallest_positive_indices] for index in not_pruned_indices)
                            weight_mask.index_put_(smallest_positive_original_indices, torch.zeros(len(smallest_positive_indices), device=weights.device))

                        # Get indices of the largest negative weights
                        negative_weights = not_pruned_weights[not_pruned_weights < 0]
                        if len(negative_weights) > 0:
                            num_negative_to_prune = min(num_weights_to_prune, len(negative_weights))
                            largest_negative_indices = torch.topk(negative_weights, num_negative_to_prune, largest=True).indices
                            largest_negative_original_indices = tuple(index[largest_negative_indices] for index in not_pruned_indices)
                            weight_mask.index_put_(largest_negative_original_indices, torch.zeros(len(largest_negative_indices), device=weights.device))


def calculate_sparsity(epoch, total_epochs, initial_sparsity, final_sparsity, pruning_epochs):
    # Linear sparsity schedule starting after `pruning_epochs` epochs
    if epoch >= pruning_epochs:
        return initial_sparsity + (final_sparsity - initial_sparsity) * ((epoch - pruning_epochs) / (total_epochs - pruning_epochs))
    else:
        return initial_sparsity
