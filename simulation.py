import numpy as np
import networkx as nx
import json
from utils import SIR_net_adaptive, initNET_rnd
import sys
import multiprocessing as mp
import time

# import parameters of the simulation
with open('parameters.txt') as f:
    raw_par = f.read()
par = json.loads(raw_par)

nsim = int(sys.argv[1]) # number of simulations



# output file
# time pol r attak_rate clustering_mean clustering ave/std stat/dyn

output = open('SIR_simulation.csv', 'a')
columns = ['time\t', 'I\t', 'pol\t', 'r\t', 'ar\t', 'cc\t', 'kind\t', 'net_type']
output.writelines(columns)
output.write('\n')

# random generator
rng = np.random.default_rng(2022)

def simulation_step(par, rng, r, pol):
    seed = rng.integers(1, 1000)
    initial_novax = rng.choice(np.arange(par['N']), par['n_novax'])
    initial_infecteds = rng.choice(np.arange(par['N']), par['n_infecteds'])
    phys_net = nx.barabasi_albert_graph(par['N'], int(par['ave_degree']/2))
    info_net_stat = phys_net.copy()
    initNET_rnd(info_net_stat, initial_novax=initial_novax)
    info_net_dyn = info_net_stat.copy()

    time_stat, _, I_stat, _, I_tot_stat = SIR_net_adaptive(
        phys_net, info_net_stat,
        beta=par['beta'],
        mu=par['mu'],
        r=r,
        pro=par['pro'],
        pol=pol,
        p_sym=par['p_sym'],
        initial_infecteds=initial_infecteds,
        rewiring=False,
        rng=np.random.default_rng(seed),
        message=False)

    time_dyn, _, I_dyn, _, I_tot_dyn = SIR_net_adaptive(
        phys_net, info_net_dyn,
        beta=par['beta'],
        mu=par['mu'],
        r=r,
        pro=par['pro'],
        pol=pol,
        p_sym=par['p_sym'],
        initial_infecteds=initial_infecteds,
        rewiring=True,
        rng=np.random.default_rng(seed),
        message=False)
    cc_stat = nx.algorithms.cluster.average_clustering(info_net_stat)
    cc_dyn = nx.algorithms.cluster.average_clustering(info_net_dyn)

    return(
        [len(time_stat), len(time_dyn)],
        [I_stat, I_dyn],
        [I_tot_stat, I_tot_dyn],
        [cc_stat, cc_dyn])
    

def simulate_params(r, pol, par, rng, out_file):
    pool = mp.Pool(mp.cpu_count())
    results = [pool.apply_async(simulation_step, args=(par, rng, r, pol)) for _ in range(nsim)]
    answers = [res.get(timeout=240) for res in results]
    pool.close()

    answers = np.array(answers, dtype=object) # [nsim, 4, 2]
    static_data = answers[:, :, 0] # [nsim, 4]
    dynamic_data = answers[:, :, 1]# [nsim, 4]
    min_time_stat = min(static_data[:, 0])
    min_time_dyn = min(dynamic_data[:, 0])

    mean_I_stat = np.mean([static_data[i, 1][:min_time_stat] for i in range(nsim)], axis=0)
    std_I_stat = np.std([static_data[i, 1][:min_time_stat] for i in range(nsim)], axis=0)
    mean_I_dyn = np.mean([dynamic_data[i, 1][:min_time_dyn] for i in range(nsim)], axis=0)
    std_I_dyn = np.std([dynamic_data[i, 1][:min_time_dyn] for i in range(nsim)], axis=0)

    mean_I_tot_stat = np.mean(static_data[:, 2])
    std_I_tot_stat = np.std(static_data[:, 2])
    mean_I_tot_dyn = np.mean(dynamic_data[:, 2])
    std_I_tot_dyn = np.std(dynamic_data[:, 2])

    mean_cc_stat = np.mean(static_data[:, 3])
    std_cc_stat = np.std(static_data[:, 3])
    mean_cc_dyn = np.mean(dynamic_data[:, 3])
    std_cc_dyn = np.std(dynamic_data[:, 3])

    times_stat = np.arange(min_time_stat)
    times_dyn = np.arange(min_time_dyn)

    for t, i_mean, i_std in zip(times_stat, mean_I_stat, std_I_stat):
        out_file.write(f'{t}\t{i_mean}\t{pol}\t{r}\t{int(mean_I_tot_stat)}\t{round(mean_cc_stat, 3)}\tmean\tstatic')
        out_file.write('\n')
        out_file.write(f'{t}\t{i_std}\t{pol}\t{r}\t{int(std_I_tot_stat)}\t{round(std_cc_stat, 3)}\tstd\tstatic')
        out_file.write('\n')

    for t, i_mean, i_std in zip(times_dyn, mean_I_dyn, std_I_dyn):
        out_file.write(f'{t}\t{i_mean}\t{pol}\t{r}\t{int(mean_I_tot_dyn)}\t{round(mean_cc_dyn, 3)}\tmean\tdynamic')
        out_file.write('\n')
        out_file.write(f'{t}\t{i_std}\t{pol}\t{r}\t{int(std_I_tot_dyn)}\t{round(std_cc_dyn, 3)}\tstd\tdynamic')
        out_file.write('\n')
    print(f'completed: r={r}, pol={pol}')

start = time.time()
simulate_params(0.2, 0.4, par, rng, output)
stop = time.time()
output.close()
print('\nSimulation completed')
print('simulation time:', round((stop - start)/60, 1), 'min')