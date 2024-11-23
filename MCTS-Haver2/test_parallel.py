import time

start_time = time.time()  # Record start time

# Code for parallel execution
from multiprocessing import Pool

def work(x):
    return x * x

with Pool(4) as p:
    results = p.map(work, range(100))

end_time = time.time()  # Record end time

parallel_runtime = end_time - start_time
print(f"Parallel Runtime: {parallel_runtime:.2f} seconds")
