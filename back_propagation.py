import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

#διαβασμα αρχειου
data = pd.read_csv("chd.csv")

p=input ("if you want a graphical-statistical description of the dataset enter 1, else simply press enter\n")
if p=="1":

    #παρουσίαση των δεδομένων του dataset 
    print("Description : \n\n", data.describe())
    data.hist(figsize=(20, 15), color='green')
    plt.show()
    print('ελεγχος για null τιμες\n')
    print(data.isnull().sum())
    print ("\n\n\n\n\n τα δειγματα εκπαιδευσης σε γραμμες και γνωρισματα-στηλες ειναι", data.shape,"\n")

data=np.array(data)

#κανονικοποιηση τιμων στο ευρος (0,1)
scaler=MinMaxScaler()
sdata=scaler.fit_transform(data)
print(scaler)

#σπαω τις 9 στηλες του dataset σε 8 για input->x και την 9η σε output->y 
X=np.delete(sdata,8,1)
y=sdata[:,[8]]

    
    
class Neural_Network(object):
    
    def __init__(self):
        
        # parameters
        self.inputlayer = 8  #αριθμος νευρωνων εισοδου
        self.outputlayer = 1 #αριθμος νευρωνων εξοδου
        self.hiddenlayer = 9 #αριθμος νευρωνων κρυφου επιπεδου
       
       
        #ρυθμος μαθησης
        self.learning_rate =0.01
        self.ormi_m=1
        
        # αρχικοποιηση βαρων τυχαια
        self.W1 = np.random.randn(self.inputlayer, self.hiddenlayer)  #*self.learning_rate
        self.W2 = np.random.randn(self.hiddenlayer, self.outputlayer)  #*self.learning_rate


     # συναρτηση ενεργοποιησης για τους νευρωνες εξοδου

    def sigmoid(self,s):

        return 1 / (1 + np.exp(-s))
    #παραγωγος σιγμοειδους
    def dsigmoid(self,s):

        return s * (1 - s)



    
    #ReLU  συναρτηση ενεργοποιησης για νευρωνες κρυφου επιπεδου
    def ReLU(self,s):
       
        return s*(s>0)
    
    def dReLU(self,s):
        
        return (s>0)*np.ones(s.shape)
        
        
      # forward διαδικασια
    def forward(self, X):
     
        self.fw = np.dot(X, self.W1)  
        # activation function
        self.fw2 = self.ReLU(self.fw) *self.learning_rate
        self.fw3 = np.dot(self.fw2, self.W2) *self.learning_rate
        # final activation functiοn
        out= self.sigmoid(self.fw3)    
        
        return out

    
    # πισω διαδοση σφαλματος 
    def backward(self, X, y, out):
        
        # λαθος στην εξοδο
        self.out_error = y - out 
        
        #δ
        self.out_delta = self.out_error * self.dsigmoid(out)  *self.learning_rate #*self.ormi_m 
        
         # fw2 error: how much our hidden layer weights contributed toutoutput error
        self.fw2_error = self.out_delta.dot(self.W2.T) *self.dsigmoid(out)
        
        # παραγωγος της σιγμοειδους στο  fw2 error
        self.fw2_delta = self.fw2_error * self.dsigmoid(self.fw2)  
        
        #προσαρμογή των πρώτων βαρων (input -> hidden layer) 
        self.W1 += X.T.dot(self.fw2_delta)
       
        #προςσαρμογη των βαρων (hidden layer-->output)
        self.W2 += self.fw2.T.dot(self.out_delta)   
   
    #συναρτηση εκπαιδευσης του δικτου
    def train(self, X, y,nout):
        out = self.forward(X)
        nout=self.backward(X, y, out)
        return nout 
    
    #συναρτηση υπολογισμου κοστους με L2 κανονικοποίηση
    def forward_with_regularization(self,X,r):
          
           m=y.shape[1]
           self.W1 = np.random.randn(self.inputlayer, self.hiddenlayer)  
           self.W2 = np.random.randn(self.hiddenlayer, self.outputlayer) 
           
           self.W1=self.W1*r/(2*m)
           self.W2=self.W2*r/(2*m)
           self.fw = np.dot(X, self.W1)  
           self.fw2 = self.ReLU(self.fw) *self.learning_rate
           self.fw3 = np.dot(self.fw2, self.W2) *self.learning_rate
           
           costwL2= self.sigmoid(self.fw3) 
           
           costwL2 = (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)*(r/(2*m))))
          
           return costwL2
      

    #back propagation με L2 κανονικοποίηση
    def backward_propagation_with_regularization(self,y,costwL2,r):
        m=X.shape[1]
        
        self.out_delta = self.out_error * self.dsigmoid(costwL2)  *self.learning_rate *(r/2*m)
        
        #υπολογισμος του κατα ποσο τα βαρη των κρυφων νευρώνων συνισέφεραν στο error
         
        self.fw2_error = self.out_delta.dot(self.W2.T) *self.dsigmoid(costwL2)
        
        # παραγωγος της σιγμοειδους στο  fw2 error
        self.fw2_delta = self.fw2_error * self.dsigmoid(self.fw2)  
        
        
        clf = Ridge(r)
        
        
        
        return clf
       

NN = Neural_Network()

#εναρξη εκπαιδευσης ΝΝ
p=input ("\n\n\nready to train the model?\nselect 1 and press enter for yes\n")
if p=="1":
    print("\n\nStarting training without L2 regularization\n\n\n")
    time.sleep(3)

    
for i in range(500):  # εκπαιδευω το νευρωνικο για  500 εποχες
        
        #10 fold cross validation 
        kf = KFold(n_splits=10)
        kf.get_n_splits(X)
        #το dataset σε σπαει σε x_train,x_test,y_train,y_test
        for train, test in kf.split(X):
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]
      
        print ("epoch ",i,"\n")
        print("\n\n>>>>>>>>TRAINING SET<<<<<<<<\n\n")
        print("known Output : \n",y_train,"\n")
        print("Predicted Output  : \n" ,NN.forward(X_train),"\n")
        rmse = np.sqrt(mean_squared_error(y_train, NN.forward(X_train)))
        print("loss function :\n\nrmse\n",rmse,"\n")
        rrse=np.sqrt(np.sum(np.square(y_train-NN.forward(X_train) )) / np.sum(np.square(y_train - np.mean(y_train))))
        print ("rrse  \n" ,rrse*100," % \n")
        
        print("\n>>>>>>>>TESTING SET<<<<<<<<<<<\n\n")
        print("known Output : \n",y_test,"\n")
        print("Predicted Output  : \n" ,NN.forward(X_test),"\n")
        rmse1 = np.sqrt(mean_squared_error(y_test, NN.forward(X_test)))
        print("loss function  :\n\nrmse\n",rmse1,"\n")
        rrse1=np.sqrt(np.sum(np.square(y_test-NN.forward(X_test) )) / np.sum(np.square(y_test - np.mean(y_test))))
       
        print ("rrse  \n" ,rrse1*100," % \n")
        
        print("\nabsolute difference between training and testing RMSerror is\n",abs(rmse-rmse1),"\n\n")
        
        NN.train(X_train, y_train,rmse)
        NN.train(X_test,y_test,rmse1)
        
        
        
        
        print("-----------------------------------------")
        
L2= input("do you want to compute L2 regularization to the weights?\nenter 1 if answer is Yes\nelse press enter and the program will end\n")
if L2=="1":    
        r=float(input("choose value for r :\n"))
        for i in range(500):
        
            costtrain= NN.forward_with_regularization(X_train,r)
            costtest= NN.forward_with_regularization(X_test,r)
            if i==499:
                rrse2=np.sqrt(np.sum(np.square(y_test- NN.forward_with_regularization(X_train,r) )) / np.sum(np.square(y_train - np.mean(y_train))))
                rrse3=np.sqrt(np.sum(np.square(y_test- NN.forward_with_regularization(X_test,r) )) / np.sum(np.square(y_test - np.mean(y_test))))
                print("cost with L2 regularization for training set is : \n", NN.forward_with_regularization(X_train,r))
                print ("RRSE with L2 regularization for training set is :\n",rrse2*100,"%")
                print("cost with L2 regularization for testing set is : \n", NN.forward_with_regularization(X_test,r))
                print ("RRSE with L2 regularization for testing set is :\n",rrse3*100,"%")

            NN.backward_propagation_with_regularization(y_train,costtrain,r)
            NN.backward_propagation_with_regularization(y_test,costtest,r)
