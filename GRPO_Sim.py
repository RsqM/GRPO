# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 19:08:49 2025

@author: Rohan
"""
#%%package imports

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel

#%%Main HyperParams Mentioned in paper

model_lr = 2e-5
policy_lr = 1e-6
kl_coeff = 0.04
sample_size = 64
max_length = 1024
training_batch = 1024

#%%Main GRPO Config HyperParameters as per Equation

G = 5 #No of outputs sampled from policy
epsilon = 0.15 #clipping limit
beta = 5e-4 #KL Penalty weight
learning_rate = 1e-3

#%%Example Proof - Single Q from distribution P(Q) for simulation

question = "What is the main Function of GRPO?"
possible_answers = ["Training", "Testing", "Policy Optimization", "Hyperparameter Tuning", "Automation"]
correct_answer_idx = 2

#%%Initialising Bert-base-cased

model_local_path =r"google/bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(model_local_path)
model = BertModel.from_pretrained(model_local_path)

def get_embed(txt):
  inputs = tokenizer(
    txt, return_tensors = "pt", padding=True, truncation=True, max_length=512
  )
  outputs = model(**inputs)
  return outputs.last_hidden_state.mean(dim=1).detach()

#%%Initializing Main Policy (π_θ in the formula)

class PolicyModel(nn.Module):
  
  def _init_(self, num_actions, embedding_dim=768):
    super()._init_()
    self.fc = nn.Linear(embedding_dim, num_actions)
    
  def forward(self, x):
    # Returns π_θ (o|q) - all action probabilities given the state
    return torch.softmax(self.fc(x), dim=-1)

#%%create question embedding

question_embedding = get_embed(question)

#%%Inititalise Policy (π_θ) and store old policy in temp (π_θ_old)

num_actions = len(possible_answers)
policy = PolicyModel(num_actions=num_actions) #(π_θ) in J_GRPO Formula
old_policy = PolicyModel(num_actions=num_actions) #(π_θ_old) in J_GRPO Formula
old_policy.load_state_dict(policy.state_dict())

#%%Implement Policy Optimization using J_GRPO

def PolicyOptimization():

  #1: Sample len(G) outputs from the old policy (π_θ_old)
  with torch.no_grad():
    probs_old = old_policy(question_embedding)
    sampled_actions = torch.multinomial(probs_old.squeeze(), G, replacement=True)

  #2: Calculate New Probs from New policy (π_θ)
  probs_new = policy(question_embedding)

  #3: Calculate Main Ratio (π_θ)/(π_θ_old)
  ratios = probs_new[0, sampled_actions]/probs_old[0, sampled_actions]

  #4: Calculate Rewards and advantages A_i
  rewards = torch.tensor(
    [1.0 if idx == correct_answer_idx else -0.1 for idx in sampled_actions]
  )

  #A_i = (r_i - mean(r1, r2 ... rn) / std(r1, r2 ... rn))
  advantages = (rewards - rewards.mean())/(rewards.std() + 1e-8)

  #5: Generate Clipped Values per formula over ((π_θ)/(π_θ_old), 1-ε, 1+ε)
  clipped_ratios = torch.clamp(ratios, 1 - epsilon, 1 + epsilon)

  #6: Calculate Main Loss Values according to Minimization in formula min(.)
  loss_policy = -torch.min(ratios*advantages, clipped_ratios*advantages)

  #7: Calculate KL-Divergence per Formula D_KL(π_θ||π_ref) = π_ref(o_i|q)/π_θ(o_i|q) - log(π_ref(o_i|q)/π_θ(o_i|q)) - 1
  ratio_kl = probs_old.detach() / probs_new

  kl_penalty = (ratio_kl - torch.log(ratio_kl)-1)

  #8: Implement Main Loss Function
  total_loss = (loss_policy + beta * kl_penalty).mean()

  #9: Iterative Policy Update
  optimizer = optim.Adam(policy.parameters(), lr = learning_rate)
  optimizer.zero_grad()
  total_loss.backward()
  optimizer.step()

  return total_loss, loss_policy, kl_penalty


#%%Train Loop

print("Starting training...")
for epoch in range(100):
  loss, policy_loss, kl = PolicyOptimization()
  if (epoch + 1) % 10 == 0:
    print(f"Epoch: {epoch + 1} - Total Loss: {loss.item():.4f}")

#%% Test the trained policy
with torch.no_grad():
  probs_final = policy(question_embedding)
  predicted_answer_idx = torch.argmax(probs_final).item()
  probabilities = probs_final[0].numpy()

print("\nFinal Results:")
print(f"\nQuestion: '{question}'")
print(f"Predicted answer: '{possible_answers[predicted_answer_idx]}'")
print("\nProbabilities for each answer:")

for answer, prob in zip(possible_answers, probabilities):
  print(f"{answer}: {prob:.4f}")
