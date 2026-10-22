import torch


def group_relative_policy_loss(policy_logprobs, reference_logprobs,
                               token_mask, advantages, kl_beta):
    """Token-normalized group-relative policy gradient with k3 KL.

    Each candidate receives a scalar group-relative advantage. Averaging the
    token log-probabilities prevents long transcripts from dominating a batch.
    The non-negative k3 estimator regularizes the policy toward the frozen SFT
    reference adapter.
    """
    if policy_logprobs.shape != reference_logprobs.shape:
        raise ValueError("Policy/reference log-probability shapes differ")
    if token_mask.shape != policy_logprobs.shape:
        raise ValueError("token_mask shape differs from log-probabilities")
    if advantages.ndim != 1 or advantages.size(0) != policy_logprobs.size(0):
        raise ValueError("advantages must contain one value per response")

    mask = token_mask.to(policy_logprobs.dtype)
    lengths = mask.sum(dim=1).clamp_min(1.0)
    sequence_logprobs = (policy_logprobs * mask).sum(dim=1) / lengths
    policy_loss = -(advantages.detach() * sequence_logprobs).mean()

    log_ratio = (reference_logprobs - policy_logprobs).clamp(-20.0, 20.0)
    token_kl = torch.exp(log_ratio) - log_ratio - 1.0
    sequence_kl = (token_kl * mask).sum(dim=1) / lengths
    kl = sequence_kl.mean()
    loss = policy_loss + kl_beta * kl
    return loss, {
        "policy_loss": policy_loss.detach(),
        "kl": kl.detach(),
        "mean_sequence_logprob": sequence_logprobs.detach().mean(),
        "mean_response_tokens": lengths.detach().mean(),
    }
