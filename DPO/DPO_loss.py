import torch
import torch.nn.functional as F

def dpo_loss(pi_theta_logprobs_winner, pi_ref_logprobs_winner,pi_theta_logprobs_loser,
            pi_ref_logprobs_loser,
            beta=0.1):
    """
    Calculates the DPO Loss for a batch of data.
    """
    # improvement ratio for the winner sequence
    ratio_winner = pi_theta_logprobs_winner - pi_ref_logprobs_winner
    
    # ratio for the loser sequence
    ratio_loser = pi_theta_logprobs_loser - pi_ref_logprobs_loser
    
    # logits for the sigmoid (difference of ratios * beta)
    logits = beta * (ratio_winner - ratio_loser)
    
    # final loss: log-sigmoid
    loss = -F.logsigmoid(logits).mean()
    
    #  reward tracking 
    with torch.no_grad():
        reward_winner = beta * ratio_winner.mean().item()
        reward_loser = beta * ratio_loser.mean().item()
        
    return loss, reward_winner, reward_loser