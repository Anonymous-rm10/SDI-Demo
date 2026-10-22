import copy


REFERENCE_ADAPTER_NAME = "rl_reference"


def _single_active_adapter(llm):
    active = getattr(llm, "active_adapter", None)
    if active is None:
        active = getattr(llm, "active_adapters", None)
        if callable(active):
            active = active()
    if isinstance(active, (list, tuple)):
        if len(active) != 1:
            raise RuntimeError(f"Expected one active PEFT adapter, got {active}")
        active = active[0]
    if not isinstance(active, str) or not active:
        raise RuntimeError("Could not determine the active PEFT adapter")
    return active


def _set_trainable_policy(llm, policy_name):
    policy_marker = f".{policy_name}."
    count = 0
    for name, parameter in llm.named_parameters():
        if "lora_" not in name:
            continue
        parameter.requires_grad = policy_marker in name
        if parameter.requires_grad:
            count += parameter.numel()
    if count == 0:
        raise RuntimeError(f"No trainable LoRA weights for adapter {policy_name}")
    return count
 

def clone_reference_adapter(model, reference_name=REFERENCE_ADAPTER_NAME):
    """Clone the active policy LoRA without duplicating the Qwen base."""
    llm = model.llm
    if not hasattr(llm, "peft_config") or not hasattr(llm, "add_adapter"):
        raise RuntimeError("RL requires a PEFT LoRA model")
    policy_name = _single_active_adapter(llm)
    if reference_name in llm.peft_config:
        raise ValueError(f"Adapter already exists: {reference_name}")
    llm.add_adapter(reference_name, copy.deepcopy(llm.peft_config[policy_name]))

    parameters = dict(llm.named_parameters())
    reference_marker = f".{reference_name}."
    policy_marker = f".{policy_name}."
    copied = 0
    for name, parameter in parameters.items():
        if "lora_" not in name or reference_marker not in name:
            continue
        source_name = name.replace(reference_marker, policy_marker)
        if source_name not in parameters:
            raise RuntimeError(
                f"Cannot find policy parameter for reference weight {name}")
        parameter.data.copy_(parameters[source_name].data)
        copied += parameter.numel()
    if copied == 0:
        raise RuntimeError("Reference LoRA clone copied no parameters")

    llm.set_adapter(policy_name)
    trainable = _set_trainable_policy(llm, policy_name)
    return policy_name, reference_name, trainable


def activate_adapter(model, adapter_name, policy_name):
    """Activate one adapter while keeping only policy LoRA weights trainable."""
    model.llm.set_adapter(adapter_name)
    return _set_trainable_policy(model.llm, policy_name)

