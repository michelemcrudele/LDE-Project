import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from utils import SIR_net_adaptive
import utils
import multiprocessing as mp
import time

N = 1000 # number of nodes
ave_degree = 15
p = ave_degree / N  # edge probability
initial_infecteds = np.random.choice(np.arange(N), size=100, replace=False)
mu = 0.1       # recovery rate
beta = 0.15    # infection rate
pro = 0.05       # rate of classical media influence on people
p_sym = 0.    # prob for an infected person to have severe symptoms
r = 0.5         # rate of vaccination for PV
polarization = [0.1, 0.3, 0.5, 0.7, 0.9]

def simulate_polarization():
    ar = []
    for pol in polarization:
        G_phys = nx.barabasi_albert_graph(N, int(ave_degree/2))
        G_info = G_phys.copy()
        utils.initNET_SI(G_info, 300, np.arange(10))
        sim = SIR_net_adaptive(G_phys, G_info, beta=beta, mu=mu, r=r, pro=pro, pol=pol, p_sym=p_sym, initial_infecteds=initial_infecteds, message=False)
        ar.append(sim[4])
    return ar

nsim = 5
answers = []
pool = mp.Pool(mp.cpu_count())

start = time.time()
results = [pool.apply_async(simulate_polarization) for _ in range(nsim)]
answers = [res.get(timeout=180) for res in results]
pool.close()
stop = time.time()

answers = np.array(answers)
ar_mean = np.mean(answers, axis=0)
ar_sd = np.std(answers, axis=0)

plt.figure(figsize=(10,6))
plt.xlabel('polarization', size=15)
plt.ylabel('final attack rate', size=15)
plt.title(f'{nsim} simulations', size=12)
plt.boxplot(answers, patch_artist=True, positions=polarization)
plt.grid(axis='y', alpha=0.5)
plt.xlim(0,1)
plt.savefig(f'polarization_{nsim}sim_{r}r.png')
print('n simulations:', nsim)
print('time elapsed:', round((stop - start)/60, 2), 'minutes')