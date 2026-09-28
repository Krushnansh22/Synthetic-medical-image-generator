import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

np.random.seed(42)

steps =30
resolution = 512

N_requests = 200
true_lambda = 0.003               # 11 requests an hour
per_image_time = 1.09376              # seconds per image
possible_images = [50, 100, 250, 500, 1000]

# Arrival times
inter_arrivals = np.random.exponential(1 / true_lambda, N_requests)
arrival_times = np.cumsum(inter_arrivals)

start_service_times = []
end_service_times = []
wait_times = []
num_images_list = []
service_times = []

server_free_time = 0.0

for t in arrival_times:
    num_images = np.random.choice(possible_images, p=[0.05,0.2,0.35,0.2,0.2])
    service_time = num_images * per_image_time

    start_service = max(t, server_free_time)
    end_service = start_service + service_time

    num_images_list.append(num_images)
    service_times.append(service_time)
    start_service_times.append(start_service)
    end_service_times.append(end_service)
    wait_times.append(start_service - t)

    server_free_time = end_service

# Build request table
df_requests = pd.DataFrame({
    "request_id": np.arange(1, N_requests + 1),
    "arrival_time_s": arrival_times,
    "num_images": num_images_list,
    "start_service_time_s": start_service_times,
    "end_service_time_s": end_service_times,
    "service_time_s": service_times,
    "wait_time_s": wait_times
})

print("\n=== Synthetic Request Log with Variable Request Size ===\n")
print(df_requests.round(2))
df_requests.to_csv(f'{resolution}-{steps}-High.csv', index=False)

measured_inter_arrivals = np.diff(df_requests["arrival_time_s"])
lambda_hat = 1.0 / measured_inter_arrivals.mean()

# Performance metrics
# 1. Average service time per request
E_S = df_requests["service_time_s"].mean()

# 2. Service rate (requests per second)
mu_hat = 1.0 / E_S

# 3. Utilization
rho = lambda_hat / mu_hat

# 4. Average waiting time and response time from the log
Wq_emp = df_requests["wait_time_s"].mean()  # average waiting time in queue
W_emp = (df_requests["end_service_time_s"] - 
         df_requests["arrival_time_s"]).mean()  # total time in system

print("\nEstimated performance metrics from log")
print(f"Average service time E[S]   = {E_S:.2f} s")
print(f"Estimated service rate μ̂    = {mu_hat:.5f} req/s "
      f"({mu_hat*3600:.2f} req/hour)")
print(f"Estimated arrival rate λ̂    = {lambda_hat:.5f} req/s "
      f"({lambda_hat*3600:.2f} req/hour)")
print(f"Utilization ρ = λ̂/μ̂        = {rho:.2f}")
print(f"Avg waiting time Wq (emp)   = {Wq_emp:.2f} s "
      f"({Wq_emp/60:.2f} min)")
print(f"Avg response time W (emp)   = {W_emp:.2f} s "
      f"({W_emp/60:.2f} min)")

T_total = df_requests["end_service_time_s"].max()
dt = 1
time = np.arange(0, T_total, dt)

queue_length = np.zeros(len(time))
server_busy = False
remaining_service_time = 0.0

arrival_idx = 0
arrival_trace = df_requests["arrival_time_s"].values
service_trace = df_requests["service_time_s"].values

service_queue = []

for i in range(1, len(time)):
    while arrival_idx < len(arrival_trace) and arrival_trace[arrival_idx] < time[i]:
        service_queue.append(service_trace[arrival_idx])
        arrival_idx += 1

    queue_length[i] = len(service_queue)

    if server_busy:
        remaining_service_time -= dt
        if remaining_service_time <= 0:
            server_busy = False
            
    if not server_busy and len(service_queue) > 0:
        server_busy = True
        remaining_service_time = service_queue.pop(0)

    queue_length[i] = len(service_queue)

plt.figure(figsize=(15, 4))
plt.plot(time, queue_length)
plt.xlabel("Time (seconds)")
plt.ylabel("Queue length")
plt.title(
    f"Queue evolution with variable request size "
    f"(λ̂={lambda_hat:.3f} req/s) on High Load"
)
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{resolution}-{steps}.png', dpi=200)
plt.show()

print("\nQueue simulation completed.")