# Example: conceptual illustration (not exam code)
from peft import PromptTuningConfig, get_peft_model

config = PromptTuningConfig(task_type="CAUSAL_LM", num_virtual_tokens=20)
model = get_peft_model(base_model, config)
