import torch
import torch.nn.utils.prune as prune


class ErdosRenyiPruningMethod(prune.BasePruningMethod):
    def __init__(self, p):
        """
        Initialize the pruning method using a percentage.
        Args:
            prune_percent (float): The percentage of connections to prune (0 to 100).
        """
        self.prune_percent = p

    def compute_mask(self, t, default_mask):
        """
        Compute the mask for pruning using a probability-based method.
        Args:
            t (Tensor): The tensor containing the weights of the layer.
            default_mask (Tensor): The default binary mask.

        Returns:
            mask (Tensor): The mask to be applied, which has the same size as `t`.
        """
        # Get the shape of the weight tensor
        noRows, noCols = t.shape

        # Generate a probability mask based on the Erdos-Renyi algorithm
        mask_flat = self.createWeightsMask(self.prune_percent, noRows, noCols)
        mask_flat = mask_flat.to(t.device)

        # Combine with the default mask
        mask = mask_flat.view_as(t) * default_mask.to(t.device)

        return mask

    def createWeightsMask(self, prune_percent, noRows, noCols):
        """
        Generate an Erdos-Renyi sparse weights mask based on percentage.
        Args:
            prune_percent (float): The percentage of connections to prune (0 to 1).
            noRows (int): Number of rows (input neurons).
            noCols (int): Number of columns (output neurons).

        Returns:
            Tensor: The mask with 0s and 1s based on the computed probability.
        """
        # Calculate epsilon from the prune percentage
        epsilon = prune_percent * (noRows * noCols) / (noRows + noCols)

        # Calculate the probability for each connection to remain
        prob = (epsilon * (noRows + noCols)) / (noRows * noCols)

        # Generate random values and apply the probability to generate the mask
        mask_weights = torch.rand(noRows, noCols)
        mask_weights[mask_weights < prob] = 0
        mask_weights[mask_weights >= prob] = 1

        # Return the mask as a tensor
        return mask_weights.float()


# class ErdosRenyiPruningMethod(prune.BasePruningMethod):
#     def __init__(self, p):
#         """
#         Initialize the pruning method.
#         Args:
#             p (float): The target sparsity, the probability of pruning a given connection.
#         """
#         self.p = p
#
#     def compute_mask(self, t, default_mask):
#         """
#         Compute the mask for pruning.
#         Args:
#             t (Tensor): The tensor containing the weights of the layer.
#             default_mask (Tensor): The default binary mask.
#
#         Returns:
#             mask (Tensor): The mask to be applied, which has the same size as `t`.
#         """
#         # Calculate the number of elements that should be pruned
#         num_elements = t.numel()
#         num_pruned = int(self.p * num_elements)
#
#         # # Get the absolute values of the weights
#         # weights = t.abs()
#
#         # Flatten the weights and default mask
#         weights_flat = t.view(-1)
#         default_mask_flat = default_mask.view(-1)
#
#         # Sort weights and get the indices
#         # _, indices = torch.topk(weights_flat, num_pruned, largest=False)
#         indices = torch.randperm(weights_flat.numel())[:num_pruned]
#
#         # Create the mask based on Erdos-Renyi probability
#         mask_flat = torch.ones_like(weights_flat)
#         mask_flat[indices] = 0
#
#         # Reshape the mask back to the original shape
#         mask = mask_flat.view_as(t)
#
#         # Combine with the default mask
#         mask = mask * default_mask
#
#         return mask