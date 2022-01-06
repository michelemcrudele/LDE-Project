import numpy as np   
import networkx as nx
import matplotlib.pyplot as plt
import random
import sys

# Simple SIR process
def SIR_net(G, beta, mu, initial_infecteds, seed=123):
    """G = network, beta = infection rate, mu = recovery rate,
    initial_infecteds = list of infected nodes at time t=0 """

    np.random.seed(seed)
    random.seed(seed)

    #INITIALIZATION
    inf_status = {}  # infectious status of a node: 0 = susceptible; 1 = infectious; 2 = recovered 
    for i in range(G.number_of_nodes()):
        inf_status[i] = 0
    for i in initial_infecteds:
        inf_status[i] =1  
    nx.set_node_attributes(G, inf_status, 'inf_status')
    nx.set_node_attributes(G, inf_status, 'new_inf_status')

    #SIMULATION
    # one single iteration
    N = G.number_of_nodes()
    time = []
    S = []
    I = []
    R = []
    time.append(0)
    S.append(N - len(initial_infecteds))
    I.append(len(initial_infecteds))
    R.append(0)
    t=0
    while True:
        t+=1
        time.append(t)
        # transmission and recovery
        for i in nx.nodes(G):
            if G.nodes[i]['inf_status'] == 1:
                if random.random() < mu:
                    G.nodes[i]['new_inf_status'] = 2
            elif G.nodes[i]['inf_status'] == 0:
                for j in nx.all_neighbors(G, i):
                    if G.nodes[j]['inf_status'] == 1:
                        if random.random() < beta:
                            G.nodes[i]['new_inf_status'] = 1
                            break  

        # update infectious status                    
        for i in nx.nodes(G):
            G.nodes[i]['inf_status'] = G.nodes[i]['new_inf_status']

        # compute total number of susceptible, infectious, recovered
        suscep = 0
        infect = 0
        recov = 0
        for i in nx.nodes(G):
            if G.nodes[i]['inf_status'] == 0:
                suscep += 1
            elif G.nodes[i]['inf_status'] == 1:
                infect += 1
            elif G.nodes[i]['inf_status'] == 2:
                recov += 1
        S.append(suscep)
        I.append(infect)
        R.append(recov)

        # end the simulation if no mode infectious
        print(f'simulation until time t={t+1}', end='\r')
        sys.stdout.flush()
        if infect == 0:
            break

    return np.array(time), np.array(S), np.array(I), np.array(R)


def SIR_net_adaptive_NeutralState(G, NET, beta, mu,gamma, r, pro, pol, p_sym, initial_infecteds, initial_no_vax, rewiring=True, seed=123):
    """
    G = physical network
    NET = information network,
    beta = infection rate,
    mu = recovery rate,
    r = rate of vaccination for PV,
    pro = rate of classical media influence on people,
    pol = propensity of opinion polarization,
    p_sym = prob for an infected person to have severe symptoms,
    initial_infecteds = list of infected nodes at time t=0,
    initial_no_vax = list of NV nodes at time t=0
    rewiring = whether the information network should be static or dynamic"""

    np.random.seed(seed)
    random.seed(seed)
    
    #INITIALIZATION
    inf_status = {}  # infectious status of a node: 0 = susceptible; 1 = infectious; 2 = recovered 
    for i in range(G.number_of_nodes()):
        inf_status[i] = 0
    for i in initial_infecteds:
        inf_status[i] = 1
    nx.set_node_attributes(G, inf_status, 'inf_status')
    nx.set_node_attributes(G, inf_status, 'new_inf_status')
    nx.set_node_attributes(G, inf_status, 'got_infected')
    
    aware_status = {}
    for i in range(NET.number_of_nodes()):
        aware_status[i] = 0
    for i in initial_no_vax:
        aware_status[i] = 1 
    nx.set_node_attributes(NET, aware_status, 'aware_status')
    nx.set_node_attributes(NET, aware_status, 'new_aware_status')
    
    #SIMULATION
    # one single iteration
    N = G.number_of_nodes()
    time = []
    S = []
    I = []
    R = []
    Ne = [] # neutral opinion
    NV = [] # No-Vax
    PV = [] # Pro-Vax
    
    time.append(0)
    
    # SIR
    S.append(N - len(initial_infecteds))
    I.append(len(initial_infecteds))
    R.append(0)
    
    #Information
    NV.append(len(initial_no_vax))
    Ne.append(N - len(initial_no_vax))
    PV.append(0)
    
    t = 0
    total_infected = 0
    while True:
        t += 1
        time.append(t)

        # EPIDEMICS IN THE PHYSICAL NETWORK
        for i in nx.nodes(G):
            # all possible transitions
            if (NET.nodes[i]['aware_status'] == 2) and (G.nodes[i]['inf_status'] == 0):        # provax that get vaccinated
                if random.random() < r:
                    G.nodes[i]['new_inf_status'] = 2
                    
            if G.nodes[i]['inf_status'] == 1:                                                  # infectious that recover
                if random.random() < mu:
                    G.nodes[i]['new_inf_status'] = 2
                    
            elif G.nodes[i]['inf_status'] == 0:                                                # susceptible
                for j in nx.all_neighbors(G, i):
                    if G.nodes[j]['inf_status'] == 1:
                        if random.random() < beta:                                             # here they get the disease
                            G.nodes[i]['new_inf_status'] = 1
                            G.nodes[i]['got_infected'] = 1
            
            if G.nodes[i]['inf_status'] == 2:
                if random.random() < gamma:
                    G.nodes[i]['new_inf_status'] = 0
            
                            
            # EPIDEMICS IN THE INFORMATION NETWORK
            if NET.nodes[i]['aware_status'] == 0:                                              # a neutral person can:
                if random.random() < pro:
                    NET.nodes[i]['new_aware_status'] = 2                                       # become a pro vax due to classical media
                
                else:                                                                          # or look around into social media
                    x = 0 # No-Vax
                    y = 0 # Pro-Vax
                    n = 0 # neutral
                    for j in nx.all_neighbors(NET, i):                                         
                        if NET.nodes[j]['aware_status'] == 1:
                            x += 1
                        elif NET.nodes[j]['aware_status'] == 2:
                            y += 1
                        else:
                            n += 1
                    p_nv = x / (x + y + n) # prob of becoming No-Vax
                    p_pv = y / (x + y + n) # prob of becoming Pro-Vax
                    random_val = random.random()
                    if random_val <= p_nv:
                        NET.nodes[i]['new_aware_status'] = 1                                   # becomes a novax via neighbours
                                    
                    elif random_val > p_nv and random_val <= p_pv:                             # becomes a provax via neighbours
                        NET.nodes[i]['new_aware_status'] = 2
            
            elif G.nodes[i]['inf_status'] == 1:                                                # an ifected person, if symptomatic, can become provax
                if random.random() < p_sym:
                    NET.nodes[i]['new_aware_status'] = 2
        
        # REWIRING OF THE INFORMATION NETWORK 
        if rewiring:                           
            # the lists of nodes which are useful for rewiring
            neutral = []
            novax   = []
            provax  = []
        
            for u in nx.nodes(NET):
                if NET.nodes[u]['aware_status'] == 0:
                    neutral.append(u)
                elif NET.nodes[u]['aware_status'] == 1  :
                    novax.append(u)
                else:
                    provax.append(u)  

            # rewiring
            for i in nx.edges(NET):      
                i = list(i)  
                if (NET.nodes[i[0]]['aware_status'] + NET.nodes[i[1]]['aware_status'] == 3):      # two nodes with different opinions 
                    if random.random() < pol:    
                        NET.remove_edge(i[0], i[1])
                        if NET.nodes[i[0]]['aware_status'] == 1:
                            NET.add_edge(i[0], np.random.choice(novax)) 
                            NET.add_edge(i[1], np.random.choice(provax))
                        else:
                            NET.add_edge(i[1], np.random.choice(novax)) 
                            NET.add_edge(i[0], np.random.choice(provax)) 

        # UPDATE NETWORKS     
        for i in nx.nodes(G):
            G.nodes[i]['inf_status'] = G.nodes[i]['new_inf_status']
        for i in nx.nodes(NET):
            NET.nodes[i]['aware_status'] = NET.nodes[i]['new_aware_status']


        # COMPUTE THE TOTAL NUMBER OF SUSCEPTIBLE, INFECTED AND RECOVERED PEOPLE
        suscep = 0
        infect = 0
        recov = 0
        for i in nx.nodes(G):
            if G.nodes[i]['inf_status'] == 0:
                suscep += 1
            elif G.nodes[i]['inf_status'] == 1:
                infect += 1
        recov = len(nx.nodes(G)) - suscep - infect
        S.append(suscep)
        I.append(infect)
        R.append(recov)

        # end simulation if no more infectious are present
        print(f'simulation until time t={t+1}', end='\r')
        sys.stdout.flush()
        if infect == 0:
            for _, data in G.nodes(data=True):
                total_infected += data['got_infected']
            break

    return np.array(time), np.array(S), np.array(I), np.array(R), total_infected

def initNET_SI(G, I0, I_seed):
    """Initialize the information network with a SI model.
    G = network,
    I0 = number of desired initial infecteds,
    I_seed = list of infecteds to start the initialization with"""

    aware_status = {}
    for i in range(G.number_of_nodes()):
        aware_status[i] = 0
    for i in I_seed:
        aware_status[i] = 1
    nx.set_node_attributes(G, aware_status, 'aware_status')
    nx.set_node_attributes(G, aware_status, 'new_aware_status')
    actual_infected = len(I_seed)
    # SI dynamics
    while actual_infected < I0:
        for i in G.nodes():
            if G.nodes[i]['aware_status'] == 0:                                                
                for j in nx.all_neighbors(G, i):
                    if G.nodes[j]['aware_status'] == 1:
                        if random.random() < 0.1:                                             
                            if G.nodes[i]['new_aware_status'] != 1 and actual_infected < I0:
                                G.nodes[i]['new_aware_status'] = 1
                                actual_infected += 1
                            
        for i in nx.nodes(G):
                G.nodes[i]['aware_status'] = G.nodes[i]['new_aware_status']

def initNET_rnd(G, initial_novax):
    """Initialize the information network randomly from the initial_novax list"""
    aware_status = {}
    for i in range(G.number_of_nodes()):
        aware_status[i] = 0
    for i in initial_novax:
        aware_status[i] = 1
    nx.set_node_attributes(G, aware_status, 'aware_status')
    nx.set_node_attributes(G, aware_status, 'new_aware_status')


# wrt the former case, now we use just 2 states in the information network: PV and NV. The interaction between the two
# follows a voter model, plus there is the effet of classical media acting on the NV population (a small effect that 
# should account for the fact that, as time goes by, social and political pressure erode the NV population)

def SIR_net_adaptive(G, NET, beta, mu,gamma, r, pro, pol, p_sym, initial_infecteds, rewiring=True, seed=123, message=True):
    """
    G = physical network
    NET = information network,
    beta = infection rate,
    mu = recovery rate,
    r = rate of vaccination for PV,
    pro = rate of classical media influence on people,
    pol = propensity of opinion polarization,
    p_sym = prob for an infected person to have severe symptoms,
    initial_infecteds = list of infected nodes at time t=0,
    rewiring = whether the information network should be static or dynamic"""

    np.random.seed(seed)
    random.seed(seed)
    
    #INITIALIZATION
    inf_status = {}
    got_infected = {}
    for i in range(G.number_of_nodes()):
        inf_status[i] = 'S'
        got_infected[i] = 0
    for i in initial_infecteds:
        inf_status[i] = 'I'
        got_infected[i] = 1
    nx.set_node_attributes(G, inf_status, 'inf_status')
    nx.set_node_attributes(G, inf_status, 'new_inf_status')
    nx.set_node_attributes(G, got_infected, 'got_infected')
    
    #SIMULATION
    # one single iteration
    N = G.number_of_nodes()
    time = []
    S = []
    I = []
    R = []
    V = []
    NV = [] # No-Vax
    PV = [] # Pro-Vax
    
    time.append(0)
    
    # SIR
    S.append(N - len(initial_infecteds))
    I.append(len(initial_infecteds))
    R.append(0)
    V.append(0)
    
    # Information network initialized outside the function
    # 0 = pro vax, 1 = no vax
    nv_init = 0
    pv_init = 0
    for i in NET.nodes():
        if NET.nodes[i]['aware_status'] == 0:
            nv_init += 1
        else:
            pv_init += 1
    NV.append(nv_init)
    PV.append(pv_init)
    
    t = 0
    total_infected = 0
    while True:
        t += 1
        time.append(t)

        # REWIRING OF THE INFORMATION NETWORK 
        if rewiring:                           
            # the lists of nodes which are useful for rewiring
            novax   = []
            provax  = []
        
            for u in nx.nodes(NET):
                if NET.nodes[u]['aware_status'] == 0:
                    provax.append(u)
                else:
                    novax.append(u)  

            # rewiring
            for i in nx.edges(NET):      
                i = list(i)  
                if (NET.nodes[i[0]]['aware_status'] + NET.nodes[i[1]]['aware_status'] == 1):      # two nodes with different opinions 
                    if random.random() < pol:    
                        NET.remove_edge(i[0], i[1])
                        if NET.nodes[i[0]]['aware_status'] == 1:
                            NET.add_edge(i[0], np.random.choice(novax)) 
                            NET.add_edge(i[1], np.random.choice(provax))
                        else:
                            NET.add_edge(i[1], np.random.choice(novax)) 
                            NET.add_edge(i[0], np.random.choice(provax)) 

        # EPIDEMICS IN THE PHYSICAL NETWORK
        for i in nx.nodes(G):
            # all possible transitions
            if (NET.nodes[i]['aware_status'] == 0) and (G.nodes[i]['inf_status'] == 'S'):        # provax that get vaccinated
                if random.random() < r:
                    G.nodes[i]['new_inf_status'] = 'V'
                    
            if G.nodes[i]['inf_status'] == 'I':                                                  # infectious that recover
                if random.random() < mu:
                    G.nodes[i]['new_inf_status'] = 'R'
                    
            elif G.nodes[i]['inf_status'] == 'S':                                                # susceptible
                for j in nx.all_neighbors(G, i):
                    if G.nodes[j]['inf_status'] == 'I':
                        if random.random() < beta:                                             # here they get the disease
                            G.nodes[i]['new_inf_status'] = 'I'
                            G.nodes[i]['got_infected'] = 1
                            
                            
            if G.nodes[i]['inf_status'] == "R":
                if random.random() < gamma:
                    G.nodes[i]['new_inf_status'] = "S"
                            
            # EPIDEMICS IN THE INFORMATION NETWORK
            if NET.nodes[i]['aware_status'] == 1:                                              # a no vax person can:
                if random.random() < pro:
                    NET.nodes[i]['new_aware_status'] = 0                                       # become a pro vax due to classical media
                
                else:                                                                          # or look around into social media
                    x = 0 # No-Vax neighbours
                    y = 0 # Pro-Vax neighbours
                    for j in nx.all_neighbors(NET, i):                                         
                        if NET.nodes[j]['aware_status'] == 0:
                            y += 1
                        else:
                            x += 1
                    p_pv = y / (x + y) # prob of becoming Pro-Vax
                    if random.random() <= p_pv:
                        NET.nodes[i]['new_aware_status'] = 0                                   # become a pro vax via neighbours

                if G.nodes[i]['inf_status'] == 1:                                              # an ifected person, if symptomatic, can become provax
                    if random.random() < p_sym:
                        NET.nodes[i]['new_aware_status'] = 0
            
            else:                                                                              # a pro vax person can:
                if G.nodes[i]['inf_status'] != 'V':                                            # if not already vaccinated    
                    x = 0 # No-Vax neighbours
                    y = 0 # Pro-Vax neighbours
                    for j in nx.all_neighbors(NET, i):                                         
                            if NET.nodes[j]['aware_status'] == 0:
                               y += 1
                            else:
                               x += 1
                    p_nv = x / (x + y) # prob of becoming No-Vax
                    if random.random() <= p_nv:
                           NET.nodes[i]['new_aware_status'] = 1                                # become a no vax via neighbours
            

        # UPDATE NETWORKS     
        for i in nx.nodes(G):
            G.nodes[i]['inf_status'] = G.nodes[i]['new_inf_status']
        for i in nx.nodes(NET):
            NET.nodes[i]['aware_status'] = NET.nodes[i]['new_aware_status']


        # COMPUTE THE TOTAL NUMBER OF SUSCEPTIBLE, INFECTED AND RECOVERED PEOPLE
        suscep = 0
        infect = 0
        recov = 0
        for i in nx.nodes(G):
            if G.nodes[i]['inf_status'] == 'S':
                suscep += 1
            elif G.nodes[i]['inf_status'] == 'I':
                infect += 1
            elif G.nodes[i]['inf_status'] == 'R':
                recov += 1
        vaccin = len(nx.nodes(G)) - suscep - infect - recov
        S.append(suscep)
        I.append(infect)
        R.append(recov)
        V.append(vaccin)

        # end simulation if no more infectious are present
        if message:
            print(f'simulation until time t={t+1}', end='\r')
            sys.stdout.flush() 
        if infect == 0:
            for _, data in G.nodes(data=True):
                total_infected += data['got_infected']
            break

    return np.array(time), np.array(S), np.array(I), np.array(R), total_infected

def plot_info_network(G):
    nv_list = []
    pv_list = []
    for i in G.nodes():
        if G.nodes[i]['aware_status'] == 0:
            pv_list.append(i)
        else:
            nv_list.append(i)

    pos = nx.spring_layout(G)
    plt.figure(figsize=(8,8))
    nx.draw_networkx_nodes(G, pos=pos, nodelist=pv_list,
        node_color='blue', label='Pro-Vax', node_size=20)
    nx.draw_networkx_nodes(G, pos=pos, nodelist=nv_list,
        node_color='red', label='No-Vax', node_size=20)
    nx.draw_networkx_edges(G, pos=pos, width=0.1)
    plt.legend(scatterpoints=1, fontsize=12)