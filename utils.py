import numpy as np   
import networkx as nx
import random
import sys

# Simple SIR process
def SIR_net(G, beta, mu, initial_infecteds):
    """G = network, beta = infection rate, mu = recovery rate,
    initial_infecteds = list of infected nodes at time t=0 """

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


def SIR_net_adaptive(G, NET, beta, mu, r, h,pro ,initial_infecteds, initial_no_vax):


    N = len(nx.nodes(G))
    #INITIALISATION
    inf_status={}  #infectious status of a node: 0= susceptible; 1= infectious; 2= recovered 
    for i in range(G.number_of_nodes()):
        inf_status[i]= 0
    for i in initial_infecteds:
        inf_status[i]=1
    nx.set_node_attributes(G,inf_status, 'inf_status')
    nx.set_node_attributes(G,inf_status, 'new_inf_status')
    
    aware_status={}
    for i in range(NET.number_of_nodes()):
        aware_status[i]=0
    for i in initial_no_vax:
        aware_status[i]=1 
    nx.set_node_attributes(NET,aware_status, 'aware_status')
    nx.set_node_attributes(NET,aware_status, 'new_aware_status')
    
    
    #SIMULATION
    #one single iteration
    time=[]
    S=[]
    I=[]
    R=[]
    Ne=[]
    NV=[]
    PV=[]
    
    time.append(0)
    
    #SIR
    S.append(N-len(initial_infecteds))
    I.append(len(initial_infecteds))
    R.append(0)
    
    #Information
    NV.append(len(initial_no_vax))
    Ne.append(N-len(initial_no_vax))
    PV.append(0)
    
    t=0
    while True:
        
        t+=1
        time.append(t)
        
        #transmission and recovery
        for i in nx.nodes(G):
            
            # all possible transitions
            if (NET.nodes[i]['aware_status']==2) and (G.nodes[i]['inf_status']==0):         # provax that get vaccine
                rand = random.uniform(0,1)
                if rand<r:
                    G.nodes[i]['new_inf_status']=2
                    
                    
            if G.nodes[i]['inf_status']==1:                                             # infectuous that become recovered
                if random.uniform(0,1)<mu:
                    G.nodes[i]['new_inf_status']=2
                    
            elif G.nodes[i]['inf_status']==0:                                              # susceptible 
                
                if (NET.nodes[i]['aware_status']==2) or (NET.nodes[i]['aware_status']==0): #that can be
                    gamma = 1                                                              #normal persons
                else: 
                    gamma = h                                                              # or no vax
                for j in nx.all_neighbors(G, i):
                    if G.nodes[j]['inf_status']==1:
                        if random.uniform(0,1)<beta/gamma:                                 # here they get the disease
                            G.nodes[i]['new_inf_status']=1
                            
                            
            if NET.nodes[i]['aware_status']==0:                                          # a neutral person can:
                if random.uniform(0,1)<pro:
                    NET.nodes[i]['new_aware_status']=2                                   # becomes a pro vax from TV
                    continue
                
                for j in nx.all_neighbors(NET, i):                                         # or looking around
                    x = 0
                    y = 0
                    non = 0
                    if NET.nodes[j]['aware_status']==1:
                        x = x+1
                    if NET.nodes[j]['aware_status']==2:
                        y = y+1
                    non = len(list(nx.all_neighbors(NET, i)))-x-y
                        
                random2 = random.uniform(0,1)
                random3 = random.uniform(0,1)
                    
                if random2 < random3 and random2 < x/(non+x+y):
                    NET.nodes[i]['new_aware_status']=1                                    # becomes a novax via neighbours
                                    
                elif random3 < random2 and random3 < y/(non+x+y):                           # becomes a provax via neighbours
                    NET.nodes[i]['new_aware_status']=2
                            
        # the lists of nodes which are useful for rewiring
        neutral = []
        novax   = []
        provax  = []
        
        for u in nx.nodes(NET):
            if NET.nodes[u]['aware_status']==0  :
                neutral.append(u)
            elif NET.nodes[u]['aware_status']==1  :
                novax.append(u)
            else: provax.append(u)  

        # rewiring
        v = nx.edges(NET)  
        for i in v:      
            i = list(i)  
            if (NET.nodes[i[0]]['aware_status'] + NET.nodes[i[1]]['aware_status'] ==3): 
                if random.uniform(0,1)<0.5:    
                    NET.remove_edge(i[0],i[1])   
                    if random.uniform(0,1)<0.3:   
                        if NET.nodes[i[0]]['aware_status']==2:
                            NET.add_edge(i[1],np.random.choice(novax)) 
                            NET.add_edge(i[0],np.random.choice(provax))
                        else:
                            NET.add_edge(i[0],np.random.choice(novax)) 
                            NET.add_edge(i[1],np.random.choice(provax))

        
        #update infectious status                    
        for i in nx.nodes(G):
            G.nodes[i]['inf_status']=G.nodes[i]['new_inf_status']
        for i in nx.nodes(NET):
            NET.nodes[i]['aware_status']=NET.nodes[i]['new_aware_status']


        #compute total number of susceptible, infectious, recovered
        suscep=0
        infect=0
        recov=0
        for i in nx.nodes(G):
            if G.nodes[i]['inf_status']==0:
                suscep+=1
            elif G.nodes[i]['inf_status']==1:
                infect+= 1
        recov = len(nx.nodes(G)) - suscep - infect
        S.append(suscep)
        I.append(infect)
        R.append(recov)

        #end simulation if no mode infectious    
        if infect==0:
            break

    return np.array(time), np.array(S), np.array(I), np.array(R)


def SIR_net_adaptive2(G, NET, beta, mu, pro ,initial_infecteds, initial_no_vax, r=1, h=1):


    N = len(nx.nodes(G))
    #INITIALISATION
    inf_status={}  #infectious status of a node: 0= susceptible; 1= infectious; 2= recovered 
    for i in range(G.number_of_nodes()):
        inf_status[i]= 0
    for i in initial_infecteds:
        inf_status[i]=1
    nx.set_node_attributes(G,inf_status, 'inf_status')
    nx.set_node_attributes(G,inf_status, 'new_inf_status')
    
    aware_status={}
    for i in range(NET.number_of_nodes()):
        aware_status[i]=0
    for i in initial_no_vax:
        aware_status[i]=1 
    nx.set_node_attributes(NET,aware_status, 'aware_status')
    nx.set_node_attributes(NET,aware_status, 'new_aware_status')
    
    
    #SIMULATION
    #one single iteration
    time=[]
    S=[]
    I=[]
    R=[]
    Ne=[]
    NV=[]
    PV=[]
    
    time.append(0)
    
    #SIR
    S.append(N-len(initial_infecteds))
    I.append(len(initial_infecteds))
    R.append(0)
    
    #Information
    NV.append(len(initial_no_vax))
    Ne.append(N-len(initial_no_vax))
    PV.append(0)
    
    t=0
    while True:
        
        t+=1
        time.append(t)
        
        #transmission and recovery
        for i in nx.nodes(G):
            
            # all possible transitions
            if (NET.nodes[i]['aware_status']==2) and (G.nodes[i]['inf_status']==0):         # provax that get vaccine
                rand = random.uniform(0,1)
                if rand<r:
                    G.nodes[i]['new_inf_status']=2
                    
                    
            if G.nodes[i]['inf_status']==1:                                             # infectuous that become recovered
                if random.uniform(0,1)<mu:
                    G.nodes[i]['new_inf_status']=2
                    
            elif G.nodes[i]['inf_status']==0:                                              # susceptible 
                
                if (NET.nodes[i]['aware_status']==2) or (NET.nodes[i]['aware_status']==0): #that can be
                    gamma = 1                                                              #normal persons
                else: 
                    gamma = h                                                              # or no vax
                for j in nx.all_neighbors(G, i):
                    if G.nodes[j]['inf_status']==1:
                        if random.uniform(0,1)<beta/gamma:                                 # here they get the disease
                            G.nodes[i]['new_inf_status']=1
                            
                            
            if NET.nodes[i]['aware_status']==0:                                          # a neutral person can:
                if random.uniform(0,1)<pro:
                    NET.nodes[i]['new_aware_status']=2                                   # becomes a pro vax from TV
                    continue
                
                for j in nx.all_neighbors(NET, i):                                         # or looking around
                    x = 0
                    y = 0
                    non = 0
                    if NET.nodes[j]['aware_status']==1:
                        x = x+1
                    if NET.nodes[j]['aware_status']==2:
                        y = y+1
                    non = len(list(nx.all_neighbors(NET, i)))-x-y
                        
                random2 = random.uniform(0,1)
                random3 = random.uniform(0,1)
                    
                if random2 < random3 and random2 < x/(non+x+y):
                    NET.nodes[i]['new_aware_status']=1                                    # becomes a novax via neighbours
                                    
                elif random3 < random2 and random3 < y/(non+x+y):                           # becomes a provax via neighbours
                    NET.nodes[i]['new_aware_status']=2
                            
        # the lists of nodes which are useful for rewiring
        neutral = []
        novax   = []
        provax  = []
        
        for u in nx.nodes(NET):
            if NET.nodes[u]['aware_status']==0  :
                neutral.append(u)
            elif NET.nodes[u]['aware_status']==1  :
                novax.append(u)
            else: provax.append(u)  

        # rewiring
        v = nx.edges(NET)  
        for i in v:      
            i = list(i)  
            if (NET.nodes[i[0]]['aware_status'] + NET.nodes[i[1]]['aware_status'] ==3): 
                if random.uniform(0,1)<0.5:    
                    NET.remove_edge(i[0],i[1])   
                    if random.uniform(0,1)<0.3:   
                        if NET.nodes[i[0]]['aware_status']==2:
                            NET.add_edge(i[1],np.random.choice(novax)) 
                            NET.add_edge(i[0],np.random.choice(provax))
                        else:
                            NET.add_edge(i[0],np.random.choice(novax)) 
                            NET.add_edge(i[1],np.random.choice(provax))

        
        #update infectious status                    
        for i in nx.nodes(G):
            G.nodes[i]['inf_status']=G.nodes[i]['new_inf_status']
        for i in nx.nodes(NET):
            NET.nodes[i]['aware_status']=NET.nodes[i]['new_aware_status']


        #compute total number of susceptible, infectious, recovered
        suscep=0
        infect=0
        
        for i in nx.nodes(G):
            if G.nodes[i]['inf_status']==0:
                suscep+=1
            elif G.nodes[i]['inf_status']==1:
                infect+= 1
        recov = len(nx.nodes(G)) - suscep - infect
        S.append(suscep)
        I.append(infect)
        R.append(recov)

        #end simulation if no mode infectious    
        if infect==0:
            break

    return [np.max(np.array(I)/len(nx.nodes(G))), np.max(np.array(R)/len(nx.nodes(G)))]