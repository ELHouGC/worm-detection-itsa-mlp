import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
from os import path
import numpy as np

    
def load_data(train_path_file,test_path_file):
    names = ["duration","protocol_type","service","flag","src_bytes",
        "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
        "logged_in","num_compromised","root_shell","su_attempted","num_root",
        "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
        "is_host_login","is_guest_login","count","srv_count","serror_rate",
        "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
        "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
        "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
        "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","garbage"]
    train_df = pd.read_csv(train_path_file, names=names, header=0)
    test_df = pd.read_csv(test_path_file, names=names, header=0)
# Removal of redundant features
    train_df.drop('garbage', axis=1, inplace=True)
    train_df.drop('num_outbound_cmds', axis=1, inplace=True)
    train_df.drop('is_host_login', axis=1, inplace=True)
    train_df.drop_duplicates(subset=None, keep='first', inplace=True)
    
    test_df.drop('garbage', axis=1, inplace=True)
    test_df.drop('num_outbound_cmds', axis=1, inplace=True)
    test_df.drop('is_host_login', axis=1, inplace=True)
    test_df.drop_duplicates(subset=None, keep='first', inplace=True)
    #change Multi-class to binary-class
  
    train_df['label'] = train_df['label'].replace(['back', 'buffer_overflow', 'ftp_write', 'guess_passwd', 'imap', 
        'ipsweep', 'land', 'loadmodule', 'multihop', 'neptune', 'nmap', 'perl', 'phf', 'pod', 'portsweep', 
        'rootkit', 'satan', 'smurf', 'spy', 'teardrop', 'warezclient', 'warezmaster'], 'attack')
    test_df['label'] = test_df['label'].replace(['neptune', 'guess_passwd' , 'mscan' , 'warezmaster','apache2' ,'satan',
        'processtable','smurf','back','snmpguess','saint','mailbomb','snmpgetattack','portsweep','ipsweep','httptunnel','nmap',
        'pod','buffer_overflow','multihop','named','ps','sendmail','rootkit','xterm','teardrop','xlock','land','xsnoop','ftp_write',
        'perl','phf','worm','udpstorm','sqlattack','loadmodule','imap'], 'attack')
      #encoding categorical data
    train_df['protocol_type'] = train_df['protocol_type'].astype('category')
    train_df['service'] = train_df['service'].astype('category')
    train_df['flag'] = train_df['flag'].astype('category')
    train_df['label'] = train_df['label'].astype('category')
    cat_columns = train_df.select_dtypes(['category']).columns
    train_df[cat_columns] = train_df[cat_columns].apply(lambda x: x.cat.codes)   

    test_df['protocol_type'] = test_df['protocol_type'].astype('category')
    test_df['service'] = test_df['service'].astype('category')
    test_df['flag'] = test_df['flag'].astype('category')
    test_df['label'] = test_df['label'].astype('category')
    cat_columns = test_df.select_dtypes(['category']).columns
    test_df[cat_columns] = test_df[cat_columns].apply(lambda x: x.cat.codes)   

    #scalling data to range of [0,1]
    scaler = preprocessing.MinMaxScaler()
 
    y_train = train_df['label']
    train_df.drop('label', axis=1, inplace=True)
    y_test = test_df['label']
    test_df.drop('label', axis=1, inplace=True)
    scaler.fit(train_df)
    x_train = scaler.fit_transform(train_df) 
    x_test = scaler.transform(test_df)
    return x_train , np.array(y_train) , x_test , np.array(y_test)

def sort__(fitness_list,population_list):
    yx = zip(fitness_list,population_list)
    yx = list(yx)
    yx.sort()
    sorted_population = [population_list for fitness_list, population_list in yx]
    yx.clear
    return sorted(fitness_list),sorted_population

def plot_results(epochs,MSE_itsa, MSE_ga, MSE_pso ,ACC_itsa, ACC_ga, ACC_pso,FAR,DR,ACC,FAR_TEST,DR_TEST,ACC_TEST):
    plt.clf()
    plt.rcParams.update({'font.size': 22})
    epochs_list = [i+1 for i in range(epochs)]
    
    f = plt.figure(figsize=(30,30))
    MSE_plt = plt.subplot2grid((2,2), (0,0))
    ACC_plt = plt.subplot2grid((2,2), (0,1))
    bar_plt = plt.subplot2grid((2,2), (1,0))
    bar_plt2 = plt.subplot2grid((2,2), (1,1))


    MSE_plt.plot(epochs_list,MSE_itsa,'--b',label ='ITSA')
    MSE_plt.plot(epochs_list,MSE_ga,'--g',label = 'GA')
    MSE_plt.plot(epochs_list,MSE_pso,'r',label = 'PSO')
    MSE_plt.set(xlabel='Epochs', ylabel='mse')
    MSE_plt.set_title('MSE results')
    MSE_plt.legend(loc='upper right',frameon=False)
    
    ACC_plt.plot(epochs_list,ACC_itsa,'--b',label ='ITSA')
    ACC_plt.plot(epochs_list,ACC_ga,'--g',label = 'GA')
    ACC_plt.plot(epochs_list,ACC_pso,'r',label = 'PSO')
    ACC_plt.set(xlabel='Epochs', ylabel='accuracy')
    ACC_plt.set_title('ACC results')
    ACC_plt.legend(loc='upper right',frameon=False)
    
    barWidth = 0.25
    br1 = np.arange(len(ACC))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    bar_plt.bar(br1, FAR, color ='r', width = barWidth,edgecolor ='grey', label ='FAR')
    bar_plt.bar(br2, DR, color ='g', width = barWidth,edgecolor ='grey', label ='DR')
    bar_plt.bar(br3, ACC, color ='b', width = barWidth, edgecolor ='grey', label ='ACC')
    bar_plt.set(xlabel='Metrics', ylabel='%')
    bar_plt.set_xticks([r + barWidth for r in range(len(ACC))]) # values
    bar_plt.set_xticklabels(['ITSA', 'GA', 'PSO']) 
    bar_plt.set_title('training metrics')
    bar_plt.legend(loc='upper right',frameon=False,)
    
    bar1 = np.arange(len(ACC_TEST))
    bar2 = [x + barWidth for x in br1]
    bar3 = [x + barWidth for x in br2]
    bar_plt2.bar(bar1, FAR_TEST, color ='r', width = barWidth,edgecolor ='grey', label ='FAR')
    bar_plt2.bar(bar2, DR_TEST, color ='g', width = barWidth,edgecolor ='grey', label ='DR')
    bar_plt2.bar(bar3, ACC_TEST, color ='b', width = barWidth, edgecolor ='grey', label ='ACC')
    bar_plt2.set(xlabel='Metrics', ylabel='%')
    bar_plt2.set_xticks([r + barWidth for r in range(len(ACC_TEST))]) # values
    bar_plt2.set_xticklabels(['ITSA', 'GA', 'PSO']) 
    bar_plt2.set_title('testing metrics')
    bar_plt2.legend(loc='upper right',frameon=False,)
    
    plt.show()
    

def get_parameters(algo):
    N = 50
    h = 79
    o = 1 
    dmin = -1.0 
    dmax = 1.0 
    epochs = 100
    if algo == 'ga':
        cros_rate = 0.8
        mu_rate = 0.08
        el_prop = 0.5
        to_prop = 0.2
        return N,cros_rate,mu_rate,el_prop,to_prop,h,o,dmin,dmax,epochs
    elif algo == 'pso':
        w = 0.65
        c1 = 1.49445
        c2 = 1.49445
        return N,w,c1,c2,h,o,dmin,dmax,epochs
    elif algo == 'itsa':
        ST = 0.715
        return N,ST,h,o,dmin,dmax,epochs
    else:
        print('wrong algo name')
        sys.exit(1)

def print_usage():
    print('usage :')
    print('for auto parameters:(use for comparisant between algos)')
    print('        file name' ,'auto', '[dataset path]')
    print('for specific algorithm test:')
    print('        file name' ,'[algorithm name]', '[algorithm parameters]' , '[neural network parameters]','[dataset path]',\
          '[number of individu in population]','[number of epochs]')
    print('[algorithm name] : ga , pso , itsa')
    print('        [ga parameters] : cross_rate , mutation_rate , elites_proportion , tournement_proportion [range(0,1)] ' )
    print('        [pso parameters]: weight [ragne(0.4,0.9)], conitive_acceleration_factor , social_acceleration_factor [range(0,2.0)]')
    print('        [itsa parameters]: Search_Tendency_factor [range(0,1)]')
    print('[neural network parameters] : number of hidden neurons , min_weigths_value , max_weiths_value')

            
            
            

        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    