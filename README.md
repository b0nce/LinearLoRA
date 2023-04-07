# LinearLoRA
Simple implementation of LoRA.

You can use `update_model_with_lora` on your torch Module and this will automatically update the whole thing. Then you can just save your model as usual and load state dict into original module.

For example:
```python
import torch
from lora import update_model_with_lora


saving_path = "model.pt"

# Create your pretrained model
model = MyComplexNNWithLotsOfParameters(...)
# Update it, such that each linear model will be augmented with LoRA
model = update_model_with_lora(model, rank=5)

# Fine-tune it
train(model)

# Save parameters
torch.save(model.state_dict(), saving_path)

# Now you can use these weights as usual with original model!
model = MyComplexNNWithLotsOfParameters(...)
model.load_state_dict(torch.load(saving_path))
```
