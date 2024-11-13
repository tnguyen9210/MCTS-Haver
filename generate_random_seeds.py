
import numpy as np

np.random.seed(0)

seed_seq = np.random.SeedSequence(12345)

state = seed_seq.generate_state(1)  # generate 1 state

print(state)
# rng = np.random.Generator(np.random.PCG64(state))

num_trials = 1000*3
random_seeds = seed_seq.generate_state(num_trials)
print(random_seeds)

np.savetxt('random_seeds.txt', random_seeds.astype(int))
