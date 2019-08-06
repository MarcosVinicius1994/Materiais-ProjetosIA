#Eufrasio junio 17.1.8131
#Marcos Vinicius Timoteo 16.2.8388
#Periodo 6

#Trabalho modificado para fins da execução dos requisitos do trabalho 2 de Inteligencia Artificial
import random
import sklearn.model_selection as sl
import sklearn.metrics as mt
import matplotlib.pyplot as m
import matplotlib.pyplot as plot

class Perceptron:

    # Inicializacao do objeto Perceptron
    def __init__(self, learn_rate=0.01, epoch_number=1000, bias=-1):
        self.sample = self.setSamples()
        self.exit = self.setExits(self.sample)
        self.learn_rate = learn_rate
        self.epoch_number = epoch_number
        self.bias = bias
        self.number_sample = len(self.sample)
        self.col_sample = len(self.sample[0])
        self.weight = []
        self.tranning= []
        self.test = []
        self.tranningExit = []
        self.testExit = []
        #gera os dados de treino e teste
        self.tranningTest()
        self.number_traning = len(self.tranning)

    def tranningTest(self):
        self.tranning, self.test, self.tranningExit, self.testExit = sl.train_test_split(self.sample,self.exit,train_size = 0.6)

    # Funcao de Treinamento do Perceptron (Metodo Gradiente Descendente)
    def trannig(self):
        for tranning in self.tranning:
            tranning.insert(0, self.bias)

        # Inicializa os pesos w aleatoriamente
        for i in range(self.col_sample):
           self.weight.append(random.random())

        # Insere peso da entrada de polarizacao(bias)
        self.weight.insert(0, self.bias)

        epoch_count = 0
        errorL = []
        #Metodo do Gradiente Descendente para ajuste dos pesos do Perceptron
        while True:
            erro = False

            for i in range(self.number_traning):
                u = 0
                for j in range(self.col_sample + 1):
                    u = u + self.weight[j] * self.tranning[i][j]
                y = self.sign(u)
                if y != self.exit[i]:
                    for j in range(self.col_sample + 1):
                        self.weight[j] = self.weight[j] + self.learn_rate * (self.tranningExit[i] - y) * self.tranning[i][j]
                    erro = True
            print('Epoca: \n',epoch_count)
            errorL.append(erro)
            epoch_count = epoch_count + 1
            if epoch_count == 20:
                self.makegraph(epoch_count,errorL)
                break
            # Se parada porepocas ou erro
            if erro == False:
                print(('\nEpocas:\n',epoch_count))
                print('------------------------\n')
                break

    def sort(self, sample):
        sample.insert(0, self.bias)
        u = 0
        for i in range(self.col_sample + 1):
            u = u + self.weight[i] * sample[i]

        y = self.sign(u)

        if  y == -1:
            print(('Sample: ', sample))
            print('Classification: P1')
        else:
            print(('Sample: ', sample))
            print('Classification: P2')

# Funcao de Ativacao
    def sign(self, u):
        return 1 if u >= 0 else -1



    def setSamples(self):
        f = open('spam.txt','r')
        sample = []
        for line in f:
            lista = []
            aux = line.split(",")
            for num in aux:
                lista.append(float(num))
            sample.append(lista)
        f.close()
        return sample

    def setExits(self,sample):
        exit = []
        for line in sample:
            exit.append(line[-1])
        return exit


    def makegraph(self, epoch_count, err):
            plot.plot(range(epoch_count), err)
            plot.title('ErrosEpocas')
            plot.xlabel('Epocas')
            plot.ylabel('Erros')
            #plot.ylim(0, max(err))
            fig = plot.gcf()
            plot.show()
            #fig.savefig('Erro.png', format='png')

    def test(self):
        pre = []
        for elem in self.test:
            elem.insert(0, self.bias)
            u = 0
            for i in range(self.col + 1):
                u = u + self.weight[i] * elem[i]
            y = self.sign(u)
            pre.append(y)
        matrizC = mt.confusion_matrix(self.testExit, pre)
        specificity = matrizC[0][0]/(matrizC[0][0] + matrizC[0][1])
        sensitivity = matrizC[1][1]/(matrizC[1][0] + matrizC[1][1])
        print("     Matriz ")
        print(matrizC)
        print("     Especificidade")
        print(specificity)
        print("      Sensibilidade")
        print(sensitivity)

# Inicializa o Perceptron

network = Perceptron(learn_rate=0.01, epoch_number=1000, bias=-1)
# Chamada ao treinamento
network.trannig()
network.test()
