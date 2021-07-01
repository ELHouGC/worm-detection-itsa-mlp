import numpy as np

class Mlp:
    def __init__(self,learnable,i,h,o,input_data, expected_data):
        self.weightsInputHidden = np.array([])
        self.weightsHiddenOut = np.array([])
        self.biasHidden = np.array([])
        self.biasOut = np.array([])
        self._learnable_to_mat(learnable,i,h,o)
        self.input_data = input_data
        self.expected = expected_data
        self.msse=0
        self.predections = self._feed_forward()
        
    def _learnable_to_mat(self,learnable,inputNum,hiddenNum,outNum):
        index=0
        self.weightsInputHidden = learnable[index:inputNum*hiddenNum]
        index = inputNum*hiddenNum
        self.weightsInputHidden = self.weightsInputHidden.reshape(inputNum,hiddenNum)
        self.biasHidden = learnable[index:index+hiddenNum ]
        index = index + hiddenNum
        self.weightsHiddenOut = learnable[index: index+hiddenNum*outNum]
        self.weightsHiddenOut = self.weightsHiddenOut.reshape(hiddenNum,outNum)
        index=index+hiddenNum*outNum
        self.biasOut = learnable[index:]
        

    def _feed_forward(self):
        predections = []
        for ipt in self.input_data:
            #calculate the outputs for the hidden layer (dot prod + biases + activation )
            hidden = np.dot(ipt,self.weightsInputHidden)
            hidden = hidden + self.biasHidden
            hidden = self._sigmoid(hidden)
            out = np.dot(hidden,self.weightsHiddenOut)
            out = out + self.biasOut
            p = self._sigmoid(out)
            predections.append(p)
        return np.array(predections)

    
    def _sigmoid(self,z):
        return 1 / (1 + np.exp(-z))

    
    #def _sse(self):
    #    pred=np.array(self.predections)
     #   expec=np.array(self.expected)
     #  return np.square(pred - expec)

    def _sse(self):
        sse = []
        for _ in range(len(self.predections)):   
            sse.append((self.expected[_]-self.predections[_])**2)
        return sse
    
    def mse(self):
        sse = self._sse()
        sum_sse= 0
        for _ in range(len(sse)):
            sum_sse += sse[_]
        self.msse = sum_sse/len(sse) 
        return self.msse
   

    #def mse(self):
     #   self.predections = self._feed_forward()
      #  sse = self._sse()
       # self.msse = np.sum(sse)/len(sse)    
        #return self.msse 
    
    def performance_mesur(self):
        classifications = []
        tp,tn,fp,fn,=0,0,0,0
        for i in range(len(self.predections)):
            if self.predections[i] >= 0.5:
                classifications.append(1)
            else:
                classifications.append(0)
        for i in range(len(classifications)):
            if not classifications[i] and not self.expected[i]:
                tp += 1
            elif classifications[i] and self.expected[i]:
                tn += 1
            elif not classifications[i] and self.expected[i]:
                fp += 1
            else :
                fn += 1
        ACC = (tp+tn)/len(classifications)
        if fp+tn == 0:
            FAR = 0
        else:
            FAR = fp/(fp+tn)
        if tp+fn == 0:
            DR =0
        else:
            DR = tp/(tp+fn)
        return  ACC,FAR,DR 
    
    
  
    
