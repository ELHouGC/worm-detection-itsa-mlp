import numpy as np
import MLP as mlp 
from numpy import random
from random import uniform
from functions import load_data

def itsa_algo(N,ST,h,o,dmin,dmax,epochs,x_train,y_train):
    MSE,ACC,FAR,DR = [],[],[],[]
    acc,far,dr =0,0,0
    i=len(x_train[0])
#initialisation
    #range of random number of seeds
    minSeed = round(N * 0.1)
    maxSeed = round(N * 0.75)
    #dimantion of the problem 
    dim = (i+o+1)*h+o
    #acceleration coefficient
    c = 2 - (dim**2 * 0.0001)
    #delta range
    delta_min = -0.1*(dmax - dmin)
    delta_max = 0.1*(dmax - dmin)
#Generate the initial population randomly according to the dimension of the problem 
#and calculating fitness of each one 
    population = []
    fitness = []
    for s in range(N):
        pos = [uniform(dmin,dmax) for _ in range(dim)]
        population.append(pos)
        network = mlp.Mlp(np.array(pos),i,h,o,x_train, y_train)
        cur_fitness = network.mse()
        fitness.append(cur_fitness) 
#Select the tree with the best fitness value (BestTree).
    best_fitness = fitness[0]
    index_best = 0 
    for f in range(len(fitness)):
        if fitness[f] < best_fitness :
            best_fitness = fitness[f]
            index_best = f
    f = None
#saving the best fitness record for later use 
    MSE.append(best_fitness)
#Searching with Seeds
    x = 1
    network = mlp.Mlp(np.array(population[index_best]),i,h,o,x_train, y_train)
    m = network.mse()
    acc,far,dr = network.performance_mesur()
    ACC.append(acc)
    FAR.append(far)
    DR.append(dr)
    print('epoch', x , 'best mse:', MSE[-1] ,'with acc of:', acc)
    while x < epochs: #and best_fitness > 0.01:# termination critiria 
        for j in range(len(population)):
            parent_tree = population[j]
            #Set number of seeds at random 
            ns = random.randint(minSeed,maxSeed)
            seed_population = []
            seed_fitness = []
            for y in range(ns):
                #choosing random tree to approch to it
                index_rand_tree = random.randint(0,len(population))
                while index_rand_tree == j : 
                    index_rand_tree = random.randint(0,len(population))
                #creating a seed with its fitness
                seed = [0 for _ in range(dim)]
                for d in range(dim):
                    rand = uniform(0,1)
                    delta = 0 
                    if(rand < ST):
                        delta = (population[index_best][d]+ population[index_rand_tree][d])*(rand-0.5)*c
                    else:
                        delta = (population[j][d]+ population[index_rand_tree][d])*(rand-0.5)*c
                    if delta > delta_max:
                        delta = delta_max
                    else:
                        if delta < delta_min:
                            delta = delta_min
                    seed[d]= population[j][d]+delta
                    if seed[d] > dmax:
                        seed[d]= dmax
                    else:
                        if seed[d] < dmin:
                            seed[d]= dmin
                seed_population.append(seed)
                network = mlp.Mlp(np.array(seed),i,h,o,x_train,y_train)
                cur_fitness = network.mse()
                seed_fitness.append(cur_fitness)
            #Select the seed with the best fitness value among all seeds 
            best_seed_fitness = seed_fitness[0]
            best_seed_index = 0 
            for s in range(len(seed_fitness)):
                if seed_fitness[s] < best_seed_fitness:
                    best_seed_fitness = seed_fitness[s]
                    best_seed_index = s
            #IF the seed is in a better position than the parent tree, then update the position 
            #of parent tree by position of the seed.
            if best_seed_fitness <fitness[j]:
                population[j] = seed_population[best_seed_index]
                fitness[j] = best_seed_fitness
        #Select the tree with the best fitness value (BestTree).
        for f in range(len(fitness)):
            if fitness[f] < best_fitness :
                best_fitness = fitness[f]
                index_best = f 
        MSE.append(best_fitness)
        x += 1
        network = mlp.Mlp(np.array(population[index_best]),i,h,o,x_train, y_train)
        m = network.mse()
        acc,far,dr = network.performance_mesur()
        ACC.append(acc)
        FAR.append(far)
        DR.append(dr)
        print('epoch', x , 'best mse:', MSE[-1] ,'with acc of:', acc)
    return MSE, ACC , FAR , DR , population[index_best]
        
        