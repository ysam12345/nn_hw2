import math
import random

class Neural():
    def __init__(self, index, X, learning_rate, is_output_layer, D = None, next_layer = None):
        """
        Construct a new 'Neural' object.

        :param index: 這個神經元在本層List中的index
        :param X: 神經元輸入資料
        :param D: 神經元期望輸出
        :param learning_rate: 學習率
        :param is_output_layer: 是否為輸出層
        :param next_layer: 下一層神經網路的List
        :return: returns nothing
        """
        self.dimension = len(X)
        self.weight = []
        for i in range(self.dimension):
            self.weight.append(random.uniform(-1,1))
        self.index = index
        self.X = X
        self.D = D
        self.learning_rate = learning_rate
        self.is_output_layer = is_output_layer
        self.next_layer = None
        if next_layer != None:
            self.next_layer = next_layer
        self.bias = 0
        self.epoch = 0
        self.v = 0
        self.Y = 0

    def calculate_Y(self):
        self.v = 0
        for i in range(len(self.weight)):
            self.v += self.X[i] * self.weight[i]
        self.v -= self.bias
        self.Y = self.sigmoid(self.v)
        return self.sigmoid(self.v)

    def sigmoid(self, x):
        if x < 0:
            return 1 - 1 / (1 + math.exp(x))
        return 1 / (1 + math.exp(-x))

    def backpropagation(self):
        self.calculate_Y()
        self.delta = None
        # 計算區域梯度函數
        if(self.is_output_layer):
            # 輸出層神經元(有期望輸出)
            self.e = self.D - self.Y
            print("輸出層神經元 D:{}  Y:{}  e:{}".format(self.D, self.Y, self.e))
            self.delta = (self.e) * self.Y # self.Y = self.sigmoid(self.v)
        else:
            # 隱藏層神經元
            delta_next_layer_multiply_weight = 0
            for next_layer_neural in self.next_layer:
                delta_next_layer_multiply_weight += next_layer_neural.delta * next_layer_neural.weight[self.index]
            self.delta = self.Y * delta_next_layer_multiply_weight
        # 調整鍵結值
        for i in range(len(self.weight)):
            self.weight[i] += self.learning_rate * self.delta * self.X[i]
                    

