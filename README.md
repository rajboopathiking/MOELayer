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

example :

```python
class BertMoEModel(nn.Module):
    def __init__(self, num_experts=4, k=2, num_classes=3):
        super(BertMoEModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.moe = MoELayer(input_dim=768, num_experts=num_experts, hidden_dim=1024, output_dim=768, k=k)

        # Task-specific heads
        self.classification_head = nn.Linear(768, num_classes)  # For classification
        self.regression_head = nn.Linear(768, 1)  # For regression

    def forward(self, input_ids, attention_mask, task="classification"):
        bert_output = self.bert(input_ids, attention_mask).last_hidden_state[:, 0, :]
        moe_output = self.moe(bert_output)

        if task == "classification":
            return self.classification_head(moe_output)
        elif task == "regression":
            return self.regression_head(moe_output)
        else:
            raise ValueError("Task must be 'classification' or 'regression'")

```
