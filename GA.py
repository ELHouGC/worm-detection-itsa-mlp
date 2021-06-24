import numpy as np 
import MLP as mlp
from random import uniform
from numpy import random
from functions import load_data,sort__

def ga_algo(N,cros_rate,mu_rate,el_prop,to_prop,h,o,dmin,dmax,epochs,x_train, y_train):
    MSE,ACC,FAR,DR = [],[],[],[]
    acc,far,dr =0,0,0 
    i = len(x_train[0])
    #initialisation
    dim = (i+o+1)*h+o
    el_num = round(N*el_prop)
    to_num=round(N*to_prop)
    population = [[uniform(dmin,dmax) for _ in range(dim)] for __ in range(N)]
    fitness = []
    for s in range(N):
        network = mlp.Mlp(np.array(population[s]),i,h,o,x_train,y_train)
        cur_fitness = network.mse()
        fitness.append(cur_fitness) 
    fitness,population = sort__(fitness,population)
    MSE.append(fitness[0])
    network = mlp.Mlp(np.array(population[0]),i,h,o,x_train, y_train)
    m = network.mse()
    acc,far,dr = network.performance_mesur()
    ACC.append(acc)
    FAR.append(far)
    DR.append(dr)
    x = 1
    print('epoch', x , 'best mse:', MSE[-1] ,'with acc of:', acc)
    while x < epochs: #and fitness[0] > 0.01:# termination critiria  
    #selection_phase
        mating_pool = []
        ##elite selection 
        elites = []
     
        for e in range(el_num):
            elites.append(population[e])
        del population[:len(elites)]
        ##tounement selection
        tournement,tournement_fitness=[],[]
            # grab percent% random individuals
        
        for _ in range(to_num):
            r_idx = random.randint(0, len(population)-1)
            tournement.append(population.pop(r_idx))
            tournement_fitness.append(fitness.pop(r_idx))
        tournement_fitness,tournement= sort__(tournement_fitness,tournement)
        winner = tournement[0]
        mating_pool.extend(elites)
        mating_pool.append(winner)
    #evolving_phase
        new_population = []
        new_population.extend(elites)
        elites.clear
        #cross_over_step
        while len(new_population) < N : 
            couple =[]
            rand = random.randint(0,len(mating_pool)-1)
            if uniform(0,1)>=cros_rate:#perform crossover based on cross over rate 
                cross = []
                cross = mating_pool[rand][:]
                if uniform(0.00, 1.00) <= mu_rate:
                    for gene in range(round(dim*mu_rate)):#mutate random genes
                        ind = random.randint(0,dim-1)
                        cross[ind]= uniform(dmin,dmax)
                new_population.append(cross)
            else:
                if len(couple)==1 and couple[0]==mating_pool[rand]: #Parent A cannot be Parent B
                    z = 11
                else:
                    couple.append(mating_pool[rand]) 
                    if len(couple) == 2: 
                        pivot_1 = random.randint(1, len(couple[0])-1)
                        pivot_2 = random.randint(pivot_1, len(couple[0])-1)
                        new_individu_1=[]
                        new_individu_2=[]
                        for p in range(dim-1):
                            if p < pivot_1:
                                new_individu_1.append(couple[0][p])
                                new_individu_2.append(couple[1][p])
                            elif p < pivot_b:
                                new_individu_1.append(couple[1][p])
                                new_individu_2.append(couple[0][p])
                            else:
                                new_individu_1.append(couple[0][p])
                                new_individu_2.append(couple[1][p])
                        if uniform(0.00, 1.00) <= mu_rate:
                            for gene in range(round(dim*mu_rate)):#mutate random genes
                                ind = random.randint(0,dim-1)
                                new_individu_1[ind]= uniform(dmin,dmax)
                                ind = random.randint(0,dim-1)
                                new_individu_2[ind] = uniform(dmin,dmax)
                        new_population.append(new_individu_1)
                        if len(new_population) < N:
                            new_population.append(new_individu_2)
                        couple.clear()
        population.clear
        population.extend(new_population)
        fitness.clear()
        for s in range(N):
            network = mlp.Mlp(np.array(population[s]),i,h,o,x_train,y_train)
            cur_fitness = network.mse()
            fitness.append(cur_fitness) 
        fitness,population = sort__(fitness,population)
        MSE.append(fitness[0])
        network = mlp.Mlp(np.array(population[0]),i,h,o,x_train, y_train)
        m = network.mse()
        acc,far,dr = network.performance_mesur()
        ACC.append(acc)
        x += 1
        print('epoch', x , 'best mse:', MSE[-1] ,'with acc of:', acc)
    return MSE, ACC, FAR, DR, population[0]
        