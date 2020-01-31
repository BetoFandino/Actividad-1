import numpy as np
import random
import math
import time

class Perceptron:
    def __init__(self, inputs, outputs, learning_rate, epochs):
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        learning_rate = learning_rate
       
        self.epochs=epochs
    def activation(self,x):
        return 1/(1 + np.exp(-x))

    def Fit(self):
        """
        Este método tiene como objetivo simular el proceso de un perceptrón, dentro de sus etapas encontramos:
        1. Generar pesos aleatorios en el rango [-1,1] para la matriz de entradas
        2. Realiza suma poderada de entradas con pesos
        3. Se aplica la función de activación para obtener una salida "y", y = 0 si la suma < 0 sino y = 1
        4. Se compara la salida "y" con la salida esperada si son iguales se procede con el siguiente vector de entrada,
        de lo contrario se generan nuevos pesos aleatorios
        :return:
        """
        # se generan pesos y bias aleatorios en el rango [-1,1]
        learning_rate=0.1     
        a=np.arange(999)
        b= np.random.choice(a)
        c=np.random.choice(a)
        self.bias =(b-c)/1000
        j=np.array(np.random.uniform(-1, 1, 1))
        weight = np.array([j,j])
        w = np.array(np.random.uniform(-1, 1, self.inputs.shape))
        #weights=np.reshape(weights,(np.size(weights),1))
        print('2',weight)
        #print('entrada:',self.inputs)
        for i in range(self.epochs):
                
                for input, w1, output in zip(self.inputs, w, self.outputs):
                    #if u==1:
                        #weight=np.reshape(weight,(np.size(weight),1))
                    # Realiza la suma ponderada de entradas con pesos
                    #print(input)
                    #print(weight)
                    dy=np.dot(input,weight) + self.bias
                    print(dy)
                    #------------------------------
                    #y_generate=self.activation(dy) 
                    #Y=y_generate-output
                   # dw=np.dot(input,Y)
                   # db=Y
                    #---------------------------------
                    #dj=np.dot(((1/(1 + np.exp(-dy))-(output))), dy)
                    #db=((1/(1 + np.exp(-dy))-(output)))
                    #weight=weight-self.learning_rate*dj
                   # self.bias=self.bias-self.learning_rate*db
                    #---------------------------------------
                    dZ=dy-output
                    #print(dZ)
                    Xt=np.reshape(input,(np.size(input),1))
                   # print(Xt)
                    #print(dZ)
                    dw1=Xt*dZ
                    #print(dw1)
                    weight=weight-learning_rate*dw1
                    print('revisar',weight)
                    self.bias=self.bias-learning_rate*dZ
                    print(self.bias)
                    #--------------------------------------
                    y_out=(1/(1 + np.exp(-(np.dot(input,weight)+ self.bias))))
                    #a=(-1)*np.dot(np.transpose(inputs),np.log(y_out))
                    #ones=np.full(np.shape(y),1)
                   # b=1-np.dot(np.transpose(inputs),(np.log(1-y_out)))
                    #perdida= np.add(a,b)
                    #------------------------------------------
                    #print('Resultados:')
                    #y_out=self.activation(input@weights+ self.bias)
                    print('entrada:',input)
                    print('salida:',y_out)
                    #print('revisar',weight)
                    #print(self.biast)
                    #print('perdida',perdida)
                    if i==5000:
                        if y_out<0.5:
                              print("false")
                        if y_out>0.5:
                              print("True")
                    u==0
                print('Epoca:',i)
                
    #def cost_func(self, y,y_out):
        #a=(-1)*np.dot(np.transpose(y),np.log(y_out))
        #ones=np.full(np.shape(y),1)
       # b=1-np.dot(np.transpose(y),(np.log(1-y_out)))
       # perdida= np.add(a,b)
       # return np.average(np.add(a,b))
if __name__ == '__main__':
    u=1 
    inputs = [
        [0,0],
        [0,1],
        [1,0],
        [1,1]
    ]

    outputs = [0,0,0,1]

    perceptron = Perceptron(inputs, outputs, learning_rate=0.111, epochs=5001)
    
    perceptron.Fit()
