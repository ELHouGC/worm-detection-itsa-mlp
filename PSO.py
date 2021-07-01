from random import uniform 
import numpy as np 
import MLP as mlp
from functions import load_data

def pso_algo(N,w, c1, c2, h, o, dmin, dmax, epochs,x_train,y_train):
    MSE,ACC,FAR,DR = [],[],[],[]
    acc,far,dr =0,0,0 
    i = len(x_train[0])
    #initialisation
    dim = (i+o+1)*h+o
    swarm = [[uniform(dmin,dmax) for _ in range(dim)] for __ in range(N)]
    velocity = [[0.00 for _ in range(dim)] for __ in range(N)]
    fitness =[]
    p_best_fitness=[]
    p_best_position=[]
    global_best_index = 0
    #calculate fitness of every particle
    for s in range(N):
        network = mlp.Mlp(np.array(swarm[s]),i,h,o,x_train,y_train)
        cur_fitness = network.mse()
        fitness.append(cur_fitness)
        p_best_fitness.append(cur_fitness) # first fitness is always the best
        p_best_position.append(swarm[s])
    for f in range(N):
        if fitness[f]<fitness[global_best_index]:
            global_best_index = f
    e = 1
    MSE.append(fitness[global_best_index])
    network = mlp.Mlp(np.array(swarm[global_best_index]),i,h,o,x_train, y_train)
    m = network.mse()
    acc,far,dr = network.performance_mesur()
    ACC.append(acc)
    FAR.append(far)
    DR.append(dr)
    print('epoch', e , 'best mse:', MSE[-1] ,'with acc of:', acc)
    while e < epochs: # and fitness[global_best_index]>0.01:
        #for every particle in swarm...
        for p in range(len(swarm)):
            #...move particle
            for d in range(dim):
                weight = w*velocity[p][d]
                cognitive = c1 * uniform(0.00, 1.00)*(p_best_position[p][d] - swarm[p][d])
                social = c2 * uniform(0.00, 1.00)*(swarm[global_best_index][d] - swarm[p][d])
                velocity[p][d] = weight + cognitive + social
                swarm[p][d] = swarm[p][d] +  velocity[p][d]
                if swarm[p][d] > dmax : ##absorbing boundaries
                    swarm[p][d] = dmax 
                elif swarm[p][d] < dmin :
                    swarm[p][d]  = dmin
            network = mlp.Mlp(np.array(swarm[p]),i,h,o,x_train,y_train)
            fitness[p] = network.mse()
            if fitness[p]<p_best_fitness[p]:
                p_best_fitness[p] = fitness[p]
                p_best_position[p]= swarm[p][:]
            if fitness[p]<fitness[global_best_index]:
                global_best_index = p
        e += 1
        MSE.append(fitness[global_best_index])
        network = mlp.Mlp(np.array(swarm[global_best_index]),i,h,o,x_train, y_train)
        m = network.mse()
        acc,far,dr = network.performance_mesur()
        ACC.append(acc)
        FAR.append(far)
        DR.append(dr)
        print('epoch', e , 'best mse:', MSE[-1] ,'with acc of:', acc)
    return MSE, ACC , FAR , DR , swarm[global_best_index]