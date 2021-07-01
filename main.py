import sys 
from TSA_algorithm import itsa_algo
from PSO import pso_algo
from GA import ga_algo
import MLP
import numpy as np
import functions as fnc

#if __name__ == '__main__' :
#    if len(sys.argv)<3:
 #       fnc.print_usage()
  #  elif sys.argv[1] == 'auto':
def main(train_filepath,test_filepath):
    #data loading 
    print('loading data')
    #filepath = sys.argv[2]
    x_train,y_train,x_test,y_test = fnc.load_data(train_filepath,test_filepath)
    ## training results vars
    ITSA_MSE_train,ITSA_ACC_train, ITSA_far_train,ITSA_dr_train= [],[],[],[]

        
    GA_MSE_train,GA_ACC_train ,GA_far_train,GA_dr_train= [],[],[],[]

        
    PSO_MSE_train,PSO_ACC_train ,PSO_far_train,PSO_dr_train = [],[],[],[]
        
    N,ST,h,o,dmin,dmax,epochs = fnc.get_parameters('itsa')
    print('working with', len(x_train), 'samples with', len(x_train[0]) ,'input neurons and' ,h, 'hidden neurons and', o , ' outputs' )
    print('neural network will be trained with ITSA , GA, PSO algorithms ')
        
    #calling for auto values for itsa algorithms
    N,ST,h,o,dmin,dmax,epochs = fnc.get_parameters('itsa')
        
    print('geting results from the ITSA training :')
    ITSA_MSE_train,ITSA_ACC_train,ITSA_far_train,ITSA_dr_train,best_itsa = itsa_algo(N,ST,h,o,dmin,dmax,epochs,x_train,y_train)
        
    #calling for auto values for GA algorithms
    N,cros_rate,mu_rate,el_prop,to_prop,h,o,dmin,dmax,epochs = fnc.get_parameters('ga')

    print('geting results from the GA training :')
    GA_MSE_train,GA_ACC_train,GA_far_train,GA_dr_train,best_ga = ga_algo(N,cros_rate,mu_rate,el_prop,to_prop,h,o,dmin,dmax,epochs,x_train,y_train)
        
    #calling for auto values for PSA algorithms
    N,w,c1,c2,h,o,dmin,dmax,epochs = fnc.get_parameters('pso')
        
    print('geting results from the PSO training :')
    PSO_MSE_train,PSO_ACC_train,PSO_far_train,PSO_dr_train ,best_pso= pso_algo(N,w,c1,c2,h,o,dmin,dmax,epochs,x_train,y_train)
    FAR =[]
    FAR.append(sum(ITSA_far_train)/len(ITSA_far_train))
    FAR.append(sum(GA_far_train)/len(GA_far_train))
    FAR.append(sum(PSO_far_train)/len(PSO_far_train))
    DR = []
    DR.append(sum(ITSA_dr_train)/len(ITSA_dr_train))
    DR.append(sum(GA_dr_train)/len(GA_dr_train))
    DR.append(sum(PSO_dr_train)/len(PSO_dr_train))
    ACC = []
    ACC.append(sum(ITSA_ACC_train)/len(ITSA_ACC_train))
    ACC.append(sum(GA_ACC_train)/len(GA_ACC_train))
    ACC.append(sum(PSO_ACC_train)/len(PSO_ACC_train))
    ##testing 
    ## test with ITSA results 
    network = MLP.Mlp(np.array(best_itsa),len(x_test[0]),h,o,x_test,y_test)
    itsa_mse_test = network.mse()
    itsa_acc_test, itsa_far_test, itsa_dr_test = network.performance_mesur() 
   
    ## test with GA results 
    network = MLP.Mlp(np.array(best_ga),len(x_test[0]),h,o,x_test,y_test)
    ga_mse_test = network.mse()
    ga_acc_test, ga_far_test, ga_dr_test = network.performance_mesur() 
    
    ## test with PSO results 
    network = MLP.Mlp(np.array(best_pso),len(x_test[0]),h,o,x_test,y_test)
    pso_mse_test = network.mse()
    pso_acc_test, pso_far_test, pso_dr_test = network.performance_mesur()
    
    FAR_TEST=[]
    FAR_TEST.append(itsa_far_test)
    FAR_TEST.append(ga_far_test)
    FAR_TEST.append(pso_far_test)
    
    DR_TEST=[]
    DR_TEST.append(itsa_dr_test)
    DR_TEST.append(ga_dr_test)
    DR_TEST.append(pso_dr_test)
    
    ACC_TEST=[]
    ACC_TEST.append(itsa_acc_test)
    ACC_TEST.append(ga_acc_test)
    ACC_TEST.append(pso_acc_test)
    
    MSE_TEST=[]
    MSE_TEST.extend(itsa_mse_test)
    MSE_TEST.extend(ga_mse_test)
    MSE_TEST.extend(pso_mse_test)
    print('------------------------------------------------------------------------------------------------')
    print('                                   train (average of epochs                             |              test                               ')
    print('------------------------------------------------------------------------------------------------')
    print('     |        MSE       |       ACC       |        FAR       |        DR      |        MSE      |        ACC       |        FAR       |        DR |')
    print('ITSA | ',(sum(ITSA_MSE_train)/len(ITSA_MSE_train)),'|',ACC[0],'|',FAR[0],'|',DR[0],'|', (itsa_mse_test),'|', (itsa_acc_test),'|', (itsa_far_test),'|', (itsa_dr_test) ,'|')
    print('------------------------------------------------------------------------------------------------')
    print('PSO | ',(sum(PSO_MSE_train)/len(PSO_MSE_train)),'|',ACC[2],'|',FAR[2],'|',DR[2],'|', (pso_mse_test),'|', (pso_acc_test),'|', (pso_far_test),'|', (pso_dr_test) ,'|')
    print('------------------------------------------------------------------------------------------------')
    print('GA | ',(sum(GA_MSE_train)/len(GA_MSE_train)),'|',ACC[1],'|',FAR[1],'|',DR[1],'|', (ga_mse_test),'|', (ga_acc_test),'|', (ga_far_test),'|', (ga_dr_test) ,'|')
    fnc.plot_results(epochs,ITSA_MSE_train, GA_MSE_train, PSO_MSE_train ,ITSA_ACC_train, GA_ACC_train, PSO_ACC_train,FAR,DR,ACC,FAR_TEST,DR_TEST,ACC_TEST)
            
      