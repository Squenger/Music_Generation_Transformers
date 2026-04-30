import torch
import torch.nn.functional as F

def get_batch_logprobs(logits, labels):
    """
    extracts the exact log-probability of the played sequence.
    - logits: [batch_size, sequence_length, vocab_size]
    - labels: [batch_size, sequence_length]
    """
    log_probs = F.log_softmax(logits, dim=-1)
    
    per_token_logprobs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    
    # ignore padding tokens
    mask = (labels != 0)
    per_token_logprobs = per_token_logprobs * mask    
    return per_token_logprobs.sum(dim=-1)