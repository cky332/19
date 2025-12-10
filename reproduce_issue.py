
import torch
from diffusers import DPMSolverMultistepInverseScheduler, DPMSolverMultistepScheduler

scheduler = DPMSolverMultistepScheduler()
inverse_scheduler = DPMSolverMultistepInverseScheduler.from_config(scheduler.config)

num_inference_steps = 1
inverse_scheduler.set_timesteps(num_inference_steps)
print(f"Timesteps for {num_inference_steps} steps: {inverse_scheduler.timesteps}")

t = inverse_scheduler.timesteps[-1]
next_timestep = t + inverse_scheduler.config.num_train_timesteps // inverse_scheduler.num_inference_steps
print(f"Calculated next_timestep: {next_timestep}")
print(f"num_train_timesteps: {inverse_scheduler.config.num_train_timesteps}")

try:
    val = inverse_scheduler.lambda_t[next_timestep]
    print(f"lambda_t[{next_timestep}] = {val}")
except IndexError as e:
    print(f"Error accessing lambda_t[{next_timestep}]: {e}")

num_inference_steps = 10
inverse_scheduler.set_timesteps(num_inference_steps)
print(f"Timesteps for {num_inference_steps} steps: {inverse_scheduler.timesteps}")
t = inverse_scheduler.timesteps[-1]
next_timestep = t + inverse_scheduler.config.num_train_timesteps // inverse_scheduler.num_inference_steps
print(f"Calculated next_timestep (last step): {next_timestep}")
