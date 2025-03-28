import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):
    def __init__(self, input_dim, num_experts, hidden_dim, output_dim, k=2):
        """
        Mixture of Experts (MoE) Layer
        - input_dim: Input feature size
        - num_experts: Number of experts
        - hidden_dim: Size of hidden layer inside each expert
        - output_dim: Output feature size
        - k: Number of top experts to activate
        """
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.k = k

        # Define Experts (Small MLPs)
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        ) for _ in range(num_experts)])

        # Gating Network (Softmax over experts)
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        """
        Forward pass for MoE Layer
        - x: Input tensor of shape (batch_size, input_dim)
        """
        batch_size, _ = x.shape

        # Compute gating scores (batch_size, num_experts)
        gate_logits = self.gate(x)  # Raw scores for each expert
        gate_weights = F.softmax(gate_logits, dim=-1)  # Convert to probabilities

        # Select Top-k Experts (Sparse Routing)
        topk_weights, topk_indices = torch.topk(gate_weights, self.k, dim=-1)  # (batch_size, k)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)  # Normalize

        # Compute Expert Outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # (batch_size, num_experts, output_dim)

        # Gather Top-k Expert Outputs
        topk_expert_outputs = torch.gather(expert_outputs, 1, topk_indices.unsqueeze(-1).expand(-1, -1, expert_outputs.shape[-1]))  # (batch_size, k, output_dim)

        # Weighted Sum of Top-k Expert Outputs
        final_output = torch.sum(topk_expert_outputs * topk_weights.unsqueeze(-1), dim=1)  # (batch_size, output_dim)

        return final_output
    

