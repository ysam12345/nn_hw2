import math
import random
import numpy as np
from neural import Neural

class MLP():
    def __init__(self, X, D, learning_rate, category_num, hidden_layer_num = 1, hidden_layer_neural_num = 20):
        self.dimension = len(X[0])
        self.data_num = len(D)
        self.X = X
        self.D = D
        self.learning_rate = learning_rate
        self.hidden_layer_num = hidden_layer_num
        self.hidden_layer_neural_num = hidden_layer_neural_num
        self.category_num = category_num
        self.category_list = np.arange(0, 1, 1/(self.category_num - 1) ).tolist()
        self.category_list.append(1.0)
        self.epoch = 0
        self.hidden_layers = []
        self.output_layer = []

        # build layers
        self.build_hidden_layer()
        self.build_sigle_neural_output_layer()

    def build_hidden_layer(self):
        next_layer = []
        hidden_layer = []
        previous_layer_output = self.X[0]
        for i in range(self.hidden_layer_num):
            current_layer_output = []
            for j in range(self.hidden_layer_neural_num):
                hidden_layer.append(Neural(j, previous_layer_output, self.learning_rate, False, None, next_layer))
                current_layer_output.append(hidden_layer[j].calculate_Y())
            previous_layer_output = current_layer_output
            self.hidden_layers.append(hidden_layer)
            hidden_layer = next_layer
            next_layer = []
        for neural in self.hidden_layers[self.hidden_layer_num - 1]:
            neural.next_layer = self.output_layer

    '''
    def build_output_layer(self):
        previous_layer_output = []
        for neural in self.hidden_layers[self.hidden_layer_num - 1]:
            previous_layer_output.append(neural.Y)
        for i in range(self.category_num):
            self.output_layer.append(Neural(i, previous_layer_output, self.learning_rate, True, self.D[0], None))
    '''

    def build_sigle_neural_output_layer(self):
        previous_layer_output = []
        for neural in self.hidden_layers[self.hidden_layer_num - 1]:
            previous_layer_output.append(neural.Y)
        self.output_layer.append(Neural(0, previous_layer_output, self.learning_rate, True, self.D[0], None))

    def output_to_category(self, output):
        return min(self.category_list, key=lambda x:abs(x-output))

    def train(self):
        correct_num = 0
        result = {
            "weight": None,
            "acc": None
        }
        for i in range(self.data_num):
            # forward propagation
            previous_layer_output = self.X[i]
            print("train: {}: {}".format(i, self.X[i]))
            for hidden_layer in self.hidden_layers:
                current_layer_output = []
                for neural in hidden_layer:
                    neural.X = previous_layer_output
                    current_layer_output.append(neural.calculate_Y())
                previous_layer_output = current_layer_output
            forward_propagation_output = []
            for neural in self.output_layer:
                neural.X = previous_layer_output
                neural.D = self.D[i]
                forward_propagation_output.append(neural.calculate_Y())

            # back propagation
            for neural in self.output_layer:
                neural.backpropagation()
            for j in range(len(self.hidden_layers)-1, -1,-1):
                for neural in self.hidden_layers[j]:
                    neural.backpropagation()

            '''
            # calculate answer again
            previous_layer_output = self.X[i]
            for hidden_layer in self.hidden_layers:
                current_layer_output = []
                for neural in hidden_layer:
                    neural.X = previous_layer_output
                    current_layer_output.append(neural.calculate_Y())
                previous_layer_output = current_layer_output
            forward_propagation_output = []
            for neural in self.output_layer:
                neural.X = previous_layer_output
                forward_propagation_output.append(neural.calculate_Y())
            '''

            output = self.output_to_category(forward_propagation_output[0])
            print("output: {}  expect: {}".format(output, self.D[i]))
            if (output == self.D[i]):
                correct_num += 1
            else:
                print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

        print("############")
        print("acc: {}".format(100 * correct_num/self.data_num))
        print("############")
        
        hidden_layer_weight = []
        for hidden_layer in self.hidden_layers:
            weight = []
            for neural in hidden_layer:
                weight.append(neural.weight)
            hidden_layer_weight.append(weight)
        output_layer_weight = []
        for neural in self.output_layer:
            output_layer_weight.append(neural.weight)
        result["weight"] = {
            "hidden_layer_weight": hidden_layer_weight,
            "output_layer_weight": output_layer_weight
        }
        result["acc"] = 100 * correct_num/self.data_num

        return result
        

    def test(self, X, y):
        data_num = len(y)
        correct_num = 0
        result = {
            "acc": None,
            "output_list": []
        }
        for i in range(data_num):
            # forward propagation
            previous_layer_output = X[i]
            print("test: {}: {}".format(i, X[i]))
            for hidden_layer in self.hidden_layers:
                current_layer_output = []
                for neural in hidden_layer:
                    neural.X = previous_layer_output
                    current_layer_output.append(neural.calculate_Y())
                previous_layer_output = current_layer_output
            forward_propagation_output = []
            for neural in self.output_layer:
                neural.X = previous_layer_output
                neural.D = y[i]
                forward_propagation_output.append(neural.calculate_Y())
            predict = self.output_to_category(forward_propagation_output[0])
            output = {
                "input": X[i],
                "expect": y[i],
                "predict": predict
            }
            result["output_list"].append(output)
            if predict == y[i]:
                correct_num += 1

        print("############")
        print("test acc: {}".format(100 * correct_num/data_num))
        print("############")
        result["acc"] = 100 * correct_num/data_num

        return result