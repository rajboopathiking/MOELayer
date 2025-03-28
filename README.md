# MOELayer
*** This Layer Used For Mode Expert Model it help to weight for each features that will contribute to Model choose specific Task ***

Downloads :
   Downloading and Upadating >>>

   
  ```bash
   git clone https://github.com/rajboopathiking/MOELayer.git
  ```
  ```bash
   pip install -r requirements.txt
  ```

Usage :

 ```python
 moe_layer = MOE(input_dim, num_experts, hidden_dim, output_dim, k=2):
"""
        Mixture of Experts (MoE) Layer
        - input_dim: Input feature size
        - num_experts: Number of experts
        - hidden_dim: Size of hidden layer inside each expert
        - output_dim: Output feature size
        - k: Number of top experts to activate
 """
```

