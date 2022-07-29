import random
import math

#
# Shorthand:
#   "pd_" as a variable prefix means "partial derivative"
#   "d_" as a variable prefix means "derivative"
#   "_wrt_" is shorthand for "with respect to"
#   "w_ho" and "w_ih" are the index of weights from hidden to output layer neurons and input to hidden layer neurons respectively
#
# Comment references:
#
# [1] Wikipedia article on Backpropagation
#   http://en.wikipedia.org/wiki/Backpropagation#Finding_the_derivative_of_the_error
# [2] Neural Networks for Machine Learning course on Coursera by Geoffrey Hinton
#   https://class.coursera.org/neuralnets-2012-001/lecture/39
# [3] The Back Propagation Algorithm
#   https://www4.rgu.ac.uk/files/chapter3%20-%20bp.pdf

class NeuralNetwork:
    LEARNING_RATE = 0.5

    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights = None, hidden_layer_bias = None, output_layer_weights = None, output_layer_bias = None):
        self.num_inputs = num_inputs

        self.hidden_layer = NeuronLayer(num_hidden, hidden_layer_bias)
        self.output_layer = NeuronLayer(num_outputs, output_layer_bias)

        self.init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights)
        self.init_weights_from_hidden_layer_neurons_to_output_layer_neurons(output_layer_weights)

    def init_weights_from_inputs_to_hidden_layer_neurons(self, hidden_layer_weights):
        weight_num = 0
        print('初始隱藏層權重')
        for h in range(len(self.hidden_layer.neurons)):
            for i in range(self.num_inputs):
                if not hidden_layer_weights:
                    self.hidden_layer.neurons[h].weights.append(random.random())
                else:
                    self.hidden_layer.neurons[h].weights.append(hidden_layer_weights[weight_num])
                x=hidden_layer_weights[weight_num]
                print('權重',i,':',x,end=" ")
                weight_num += 1

    def init_weights_from_hidden_layer_neurons_to_output_layer_neurons(self, output_layer_weights):
        weight_num = 0
        print('\n初始輸出層權重')
        for o in range(len(self.output_layer.neurons)):
            for h in range(len(self.hidden_layer.neurons)):
                if not output_layer_weights:
                    self.output_layer.neurons[o].weights.append(random.random())
                else:
                    self.output_layer.neurons[o].weights.append(output_layer_weights[weight_num])
                x=output_layer_weights[weight_num]
                print('權重:',h,':',x,end=' ')
                weight_num += 1

    def inspect(self):
        print('------')
        print('* Inputs: {}'.format(self.num_inputs))
        print('------')
        print('Hidden Layer')
        self.hidden_layer.inspect()
        print('------')
        print('* Output Layer')
        self.output_layer.inspect()
        print('------')

    def feed_forward(self, inputs):
        hidden_layer_outputs = self.hidden_layer.feed_forward(inputs)
        return self.output_layer.feed_forward(hidden_layer_outputs)

    # Uses online learning, ie updating the weights after each training case
    def train(self, training_inputs, training_outputs):
        self.feed_forward(training_inputs)

        # 1. Output neuron deltas
        pd_errors_wrt_output_neuron_total_net_input = [0] * len(self.output_layer.neurons)
        for o in range(len(self.output_layer.neurons)):

            # ∂E/∂zⱼ
            pd_errors_wrt_output_neuron_total_net_input[o] = self.output_layer.neurons[o].calculate_pd_error_wrt_total_net_input(training_outputs[o])

        # 2. Hidden neuron deltas
        pd_errors_wrt_hidden_neuron_total_net_input = [0] * len(self.hidden_layer.neurons)
        for h in range(len(self.hidden_layer.neurons)):

            # We need to calculate the derivative of the error with respect to the output of each hidden layer neuron
            # dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
            d_error_wrt_hidden_neuron_output = 0
            for o in range(len(self.output_layer.neurons)):
                d_error_wrt_hidden_neuron_output += pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].weights[h]

            # ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
            pd_errors_wrt_hidden_neuron_total_net_input[h] = d_error_wrt_hidden_neuron_output * self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_input()
        
        print('\n輸出層權重更新')    
        # 3. Update output neuron weights
        for o in range(len(self.output_layer.neurons)):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):

                # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
                pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].calculate_pd_total_net_input_wrt_weight(w_ho)

                # Δw = α * ∂Eⱼ/∂wᵢ
                self.output_layer.neurons[o].weights[w_ho] -= self.LEARNING_RATE * pd_error_wrt_weight
                outputweight=self.output_layer.neurons[o].weights[w_ho]-self.LEARNING_RATE * pd_error_wrt_weight
                print('權重' ,':',round(outputweight,3),end=" ")
        
        print('\n隱藏層權重更新') 
        # 4. Update hidden neuron weights
        for h in range(len(self.hidden_layer.neurons)):
            for w_ih in range(len(self.hidden_layer.neurons[h].weights)):

                # ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
                pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input[h] * self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_weight(w_ih)

                # Δw = α * ∂Eⱼ/∂wᵢ
                self.hidden_layer.neurons[h].weights[w_ih] -= self.LEARNING_RATE * pd_error_wrt_weight
                hiddenweight=self.hidden_layer.neurons[h].weights[w_ih]- self.LEARNING_RATE * pd_error_wrt_weight
                print('權重',':',round(hiddenweight,3),end=" ")
        print('\n\n結果:',end=" ")

    def calculate_total_error(self, training_sets):
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.feed_forward(training_inputs)
            for o in range(len(training_outputs)):
                total_error += self.output_layer.neurons[o].calculate_error(training_outputs[o])
        return total_error

class NeuronLayer:
    def __init__(self, num_neurons, bias):

        # Every neuron in a layer shares the same bias
        self.bias = bias if bias else random.random()

        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(self.bias))

    def inspect(self):
        print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                print('  Weight:', self.neurons[n].weights[w])
            print('  Bias:', self.bias)

    def feed_forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs

    def get_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs

class Neuron:
    def __init__(self, bias):
        self.bias = bias
        self.weights = []

    def calculate_output(self, inputs):
        self.inputs = inputs
        self.output = self.squash(self.calculate_total_net_input())
        return self.output

    def calculate_total_net_input(self):
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return total + self.bias

    # Apply the logistic function to squash the output of the neuron
    # The result is sometimes referred to as 'net' [2] or 'net' [1]
    #激活函數
    def squash(self, total_net_input):
          #print('所有的:',total_net_input)
          #a,X,c,X,e,X,g,X,X = total_net_input
          return 1 / (1 + math.exp(-total_net_input))

    # Determine how much the neuron's total input has to change to move closer to the expected output
    #
    # Now that we have the partial derivative of the error with respect to the output (∂E/∂yⱼ) and
    # the derivative of the output with respect to the total net input (dyⱼ/dzⱼ) we can calculate
    # the partial derivative of the error with respect to the total net input.
    # This value is also known as the delta (δ) [1]
    # δ = ∂E/∂zⱼ = ∂E/∂yⱼ * dyⱼ/dzⱼ
    #
    def calculate_pd_error_wrt_total_net_input(self, target_output):
        return self.calculate_pd_error_wrt_output(target_output) * self.calculate_pd_total_net_input_wrt_input();

    # The error for each neuron is calculated by the Mean Square Error method:
    def calculate_error(self, target_output):
        return 0.5 * (target_output - self.output) ** 2

    # The partial derivate of the error with respect to actual output then is calculated by:
    # = 2 * 0.5 * (target output - actual output) ^ (2 - 1) * -1
    # = -(target output - actual output)
    #
    # The Wikipedia article on backpropagation [1] simplifies to the following, but most other learning material does not [2]
    # = actual output - target output
    #
    # Alternative, you can use (target - output), but then need to add it during backpropagation [3]
    #
    # Note that the actual output of the output neuron is often written as yⱼ and target output as tⱼ so:
    # = ∂E/∂yⱼ = -(tⱼ - yⱼ)
    def calculate_pd_error_wrt_output(self, target_output):
        print(target_output - self.output)
        return -(target_output - self.output)

    # The total net input into the neuron is squashed using logistic function to calculate the neuron's output:
    # yⱼ = φ = 1 / (1 + e^(-zⱼ))
    # Note that where ⱼ represents the output of the neurons in whatever layer we're looking at and ᵢ represents the layer below it
    #
    # The derivative (not partial derivative since there is only one variable) of the output then is:
    # dyⱼ/dzⱼ = yⱼ * (1 - yⱼ)
    def calculate_pd_total_net_input_wrt_input(self):
        return self.output * (1 - self.output)

    # The total net input is the weighted sum of all the inputs to the neuron and their respective weights:
    # = zⱼ = netⱼ = x₁w₁ + x₂w₂ ...
    #
    # The partial derivative of the total net input with respective to a given weight (with everything else held constant) then is:
    # = ∂zⱼ/∂wᵢ = some constant + 1 * xᵢw₁^(1-0) + some constant ... = xᵢ
    def calculate_pd_total_net_input_wrt_weight(self, index):
        return self.inputs[index]

import numpy as np
# Blog post example:
x1_1=np.arange(1312)
x1_1=np.loadtxt( r'D:\Desktop\test\test001-1.txt', delimiter='\n',max_rows =100)
x1_1=np.around(x1_1,0)

x1_2=np.arange(1312)
x1_2=np.loadtxt( r'D:\Desktop\test\test001-2.txt', delimiter='\n',max_rows =100)
x1_2=np.around(x1_2,0)

x1_3=np.arange(1312)
x1_3=np.loadtxt( r'D:\Desktop\test\test001-3.txt', delimiter='\n',max_rows =100)
x1_3=np.around(x1_3,0)

x1_4=np.arange(1312)
x1_4=np.loadtxt( r'D:\Desktop\test\test001-4.txt', delimiter='\n',max_rows =100)
x1_4=np.around(x1_4,0)

x1_5=np.arange(1312)
x1_5=np.loadtxt( r'D:\Desktop\test\test001-5.txt', delimiter='\n',max_rows =100)
x1_5=np.around(x1_5,0)

x1_6=np.arange(1312)
x1_6=np.loadtxt( r'D:\Desktop\test\test001-6.txt', delimiter='\n',max_rows =100)
x1_6=np.around(x1_6,0)

x1_7=np.arange(1312)
x1_7=np.loadtxt( r'D:\Desktop\test\test001-7.txt', delimiter='\n',max_rows =100)
x1_7=np.around(x1_7,0)

x2_1=np.arange(1312)
x2_1=np.loadtxt( r'D:\Desktop\test\test002-1.txt', delimiter='\n',max_rows =100)
x2_1=np.around(x2_1,0)

x2_2=np.arange(1312)
x2_2=np.loadtxt( r'D:\Desktop\test\test002-2.txt', delimiter='\n',max_rows =100)
x2_2=np.around(x2_2,0)

x2_3=np.arange(1312)
x2_3=np.loadtxt( r'D:\Desktop\test\test002-3.txt', delimiter='\n',max_rows =100)
x2_3=np.around(x2_3,0)

x2_4=np.arange(1312)
x2_4=np.loadtxt( r'D:\Desktop\test\test002-4.txt', delimiter='\n',max_rows =100)
x2_4=np.around(x2_4,0)

x2_5=np.arange(1312)
x2_5=np.loadtxt( r'D:\Desktop\test\test002-5.txt', delimiter='\n',max_rows =100)
x2_5=np.around(x2_5,0)

x2_6=np.arange(1312)
x2_6=np.loadtxt( r'D:\Desktop\test\test002-6.txt', delimiter='\n',max_rows =100)
x2_6=np.around(x2_6,0)

x2_7=np.arange(1312)
x2_7=np.loadtxt( r'D:\Desktop\test\test002-7.txt', delimiter='\n',max_rows =100)
x2_7=np.around(x2_7,0)

x3_1=np.arange(1312)
x3_1=np.loadtxt( r'D:\Desktop\test\test003-1.txt', delimiter='\n',max_rows =100)
x3_1=np.around(x3_1,0)

x3_2=np.arange(1312)
x3_2=np.loadtxt( r'D:\Desktop\test\test003-2.txt', delimiter='\n',max_rows =100)
x3_2=np.around(x3_2,0)

x3_3=np.arange(1312)
x3_3=np.loadtxt( r'D:\Desktop\test\test003-3.txt', delimiter='\n',max_rows =100)
x3_3=np.around(x3_3,0)

x3_4=np.arange(1312)
x3_4=np.loadtxt( r'D:\Desktop\test\test003-4.txt', delimiter='\n',max_rows =100)
x3_4=np.around(x3_4,0)

x3_5=np.arange(1312)
x3_5=np.loadtxt( r'D:\Desktop\test\test003-5.txt', delimiter='\n',max_rows =100)
x3_5=np.around(x3_5,0)

x3_6=np.arange(1312)
x3_6=np.loadtxt( r'D:\Desktop\test\test003-6.txt', delimiter='\n',max_rows =100)
x3_6=np.around(x3_6,0)

x3_7=np.arange(1312)
x3_7=np.loadtxt( r'D:\Desktop\test\test003-7.txt', delimiter='\n',max_rows =100)
x3_7=np.around(x3_7,0)

x4_1=np.arange(1312)
x4_1=np.loadtxt( r'D:\Desktop\test\test004-1.txt', delimiter='\n',max_rows =100)
x4_1=np.around(x4_1,0)

x4_2=np.arange(1312)
x4_2=np.loadtxt( r'D:\Desktop\test\test004-2.txt', delimiter='\n',max_rows =100)
x4_2=np.around(x4_2,0)

x4_3=np.arange(1312)
x4_3=np.loadtxt( r'D:\Desktop\test\test004-3.txt', delimiter='\n',max_rows =100)
x4_3=np.around(x4_3,0)

x4_4=np.arange(1312)
x4_4=np.loadtxt( r'D:\Desktop\test\test004-4.txt', delimiter='\n',max_rows =100)
x4_4=np.around(x4_4,0)

x4_5=np.arange(1312)
x4_5=np.loadtxt( r'D:\Desktop\test\test004-5.txt', delimiter='\n',max_rows =100)
x4_5=np.around(x4_5,0)

x4_6=np.arange(1312)
x4_6=np.loadtxt( r'D:\Desktop\test\test004-6.txt', delimiter='\n',max_rows =100)
x4_6=np.around(x4_6,0)

x4_7=np.arange(1312)
x4_7=np.loadtxt( r'D:\Desktop\test\test004-7.txt', delimiter='\n',max_rows =100)
x4_7=np.around(x4_7,0)

x5_1=np.arange(1312)
x5_1=np.loadtxt( r'D:\Desktop\test\test005-1.txt', delimiter='\n',max_rows =100)
x5_1=np.around(x5_1,0)

x5_2=np.arange(1312)
x5_2=np.loadtxt( r'D:\Desktop\test\test005-2.txt', delimiter='\n',max_rows =100)
x5_2=np.around(x5_2,0)

x5_3=np.arange(1312)
x5_3=np.loadtxt( r'D:\Desktop\test\test005-3.txt', delimiter='\n',max_rows =100)
x5_3=np.around(x5_3,0)

x5_4=np.arange(1312)
x5_4=np.loadtxt( r'D:\Desktop\test\test005-4.txt', delimiter='\n',max_rows =100)
x5_4=np.around(x5_4,0)

x5_5=np.arange(1312)
x5_5=np.loadtxt( r'D:\Desktop\test\test005-5.txt', delimiter='\n',max_rows =100)
x5_5=np.around(x5_5,0)

x5_6=np.arange(1312)
x5_6=np.loadtxt( r'D:\Desktop\test\test005-6.txt', delimiter='\n',max_rows =100)
x5_6=np.around(x5_6,0)

x5_7=np.arange(1312)
x5_7=np.loadtxt( r'D:\Desktop\test\test005-7.txt', delimiter='\n',max_rows =100)
x5_7=np.around(x5_7,0)

x6_1=np.arange(1312)
x6_1=np.loadtxt( r'D:\Desktop\test\test006-1.txt', delimiter='\n',max_rows =100)
x6_1=np.around(x6_1,0)

x6_2=np.arange(1312)
x6_2=np.loadtxt( r'D:\Desktop\test\test006-2.txt', delimiter='\n',max_rows =100)
x6_2=np.around(x6_2,0)

x6_3=np.arange(1312)
x6_3=np.loadtxt( r'D:\Desktop\test\test006-3.txt', delimiter='\n',max_rows =100)
x6_3=np.around(x6_3,0)

x6_4=np.arange(1312)
x6_4=np.loadtxt( r'D:\Desktop\test\test006-4.txt', delimiter='\n',max_rows =100)
x6_4=np.around(x6_4,0)

x6_5=np.arange(1312)
x6_5=np.loadtxt( r'D:\Desktop\test\test006-5.txt', delimiter='\n',max_rows =100)
x6_5=np.around(x6_5,0)

x6_6=np.arange(1312)
x6_6=np.loadtxt( r'D:\Desktop\test\test006-6.txt', delimiter='\n',max_rows =100)
x6_6=np.around(x6_6,0)

x6_7=np.arange(1312)
x6_7=np.loadtxt( r'D:\Desktop\test\test006-7.txt', delimiter='\n',max_rows =100)
x6_7=np.around(x6_7,0)

x7_1=np.arange(1312)
x7_1=np.loadtxt( r'D:\Desktop\test\test007-1.txt', delimiter='\n',max_rows =100)
x7_1=np.around(x7_1,0)


x7_2=np.arange(1312)
x7_2=np.loadtxt( r'D:\Desktop\test\test007-2.txt', delimiter='\n',max_rows =100)
x7_2=np.around(x7_2,0)


x7_3=np.arange(1312)
x7_3=np.loadtxt( r'D:\Desktop\test\test007-3.txt', delimiter='\n',max_rows =100)
x7_3=np.around(x7_3,0)

x7_4=np.arange(1312)
x7_4=np.loadtxt( r'D:\Desktop\test\test007-4.txt', delimiter='\n',max_rows =100)
x7_4=np.around(x7_4,0)

x7_5=np.arange(1312)
x7_5=np.loadtxt( r'D:\Desktop\test\test007-5.txt', delimiter='\n',max_rows =100)
x7_5=np.around(x7_5,0)

x7_6=np.arange(1312)
x7_6=np.loadtxt( r'D:\Desktop\test\test007-6.txt', delimiter='\n',max_rows =100)
x7_6=np.around(x7_6,0)

x7_7=np.arange(1312)
x7_7=np.loadtxt( r'D:\Desktop\test\test007-7.txt', delimiter='\n',max_rows =100)
x7_7=np.around(x7_7,0)

x8_1=np.arange(1312)
x8_1=np.loadtxt( r'D:\Desktop\test\test008-1.txt', delimiter='\n',max_rows =100)
x8_1=np.around(x8_1,0)

x8_2=np.arange(1312)
x8_2=np.loadtxt( r'D:\Desktop\test\test008-2.txt', delimiter='\n',max_rows =100)
x8_2=np.around(x8_2,0)

x8_3=np.arange(1312)
x8_3=np.loadtxt( r'D:\Desktop\test\test008-3.txt', delimiter='\n',max_rows =100)
x8_3=np.around(x8_3,0)

x8_4=np.arange(1312)
x8_4=np.loadtxt( r'D:\Desktop\test\test008-4.txt', delimiter='\n',max_rows =100)
x8_4=np.around(x8_4,0)

x8_5=np.arange(1312)
x8_5=np.loadtxt( r'D:\Desktop\test\test008-5.txt', delimiter='\n',max_rows =100)
x8_5=np.around(x8_5,0)

x8_6=np.arange(1312)
x8_6=np.loadtxt( r'D:\Desktop\test\test008-6.txt', delimiter='\n',max_rows =100)
x8_6=np.around(x8_6,0)

x8_7=np.arange(1312)
x8_7=np.loadtxt( r'D:\Desktop\test\test008-7.txt', delimiter='\n',max_rows =100)
x8_7=np.around(x8_7,0)

x9_1=np.arange(1312)
x9_1=np.loadtxt( r'D:\Desktop\test\test009-1.txt', delimiter='\n',max_rows =100)
x9_1=np.around(x9_1,0)

x9_2=np.arange(1312)
x9_2=np.loadtxt( r'D:\Desktop\test\test009-2.txt', delimiter='\n',max_rows =100)
x9_2=np.around(x9_2,0)

x9_3=np.arange(1312)
x9_3=np.loadtxt( r'D:\Desktop\test\test009-3.txt', delimiter='\n',max_rows =100)
x9_3=np.around(x9_3,0)

x9_4=np.arange(1312)
x9_4=np.loadtxt( r'D:\Desktop\test\test009-4.txt', delimiter='\n',max_rows =100)
x9_4=np.around(x9_4,0)

x9_5=np.arange(1312)
x9_5=np.loadtxt( r'D:\Desktop\test\test009-5.txt', delimiter='\n',max_rows =100)
x9_5=np.around(x9_5,0)

x9_6=np.arange(1312)
x9_6=np.loadtxt( r'D:\Desktop\test\test009-6.txt', delimiter='\n',max_rows =100)
x9_6=np.around(x9_6,0)

x9_7=np.arange(1312)
x9_7=np.loadtxt( r'D:\Desktop\test\test009-7.txt', delimiter='\n',max_rows =100)
x9_7=np.around(x9_7,0)

x10_1=np.arange(1312)
x10_1=np.loadtxt( r'D:\Desktop\test\test010-1.txt', delimiter='\n',max_rows =100)
x10_1=np.around(x10_1,0)


xAll=[]

for i in range(100):
    xAll.append(x1_1[i])
for i in range(100):
    xAll.append(x1_2[i])
for i in range(100):
    xAll.append(x1_3[i])
for i in range(100):
    xAll.append(x1_4[i])
for i in range(100):
    xAll.append(x1_5[i])
for i in range(100):
    xAll.append(x1_6[i])

for i in range(100):
    xAll.append(x1_7[i])
for i in range(100):
    xAll.append(x2_1[i])
for i in range(100):
    xAll.append(x2_2[i])
for i in range(100):
    xAll.append(x2_3[i])

for i in range(100):
    xAll.append(x2_4[i])
for i in range(100):
    xAll.append(x2_5[i])
for i in range(100):
    xAll.append(x2_6[i])
for i in range(100):
    xAll.append(x2_7[i])
for i in range(100):
    xAll.append(x3_1[i])
for i in range(100):
    xAll.append(x3_2[i])
for i in range(100):
    xAll.append(x3_3[i])
for i in range(100):
    xAll.append(x3_4[i])
for i in range(100):
    xAll.append(x3_5[i])

for i in range(100):
    xAll.append(x3_6[i])

for i in range(100):
    xAll.append(x3_7[i])
for i in range(100):
    xAll.append(x4_1[i])

for i in range(100):
    xAll.append(x4_2[i])

for i in range(100):
    xAll.append(x4_3[i])
for i in range(100):
    xAll.append(x4_4[i])
for i in range(100):
    xAll.append(x4_5[i])
for i in range(100):
    xAll.append(x4_6[i])
for i in range(100):
    xAll.append(x4_7[i])
for i in range(100):
    xAll.append(x5_1[i])
for i in range(100):
    xAll.append(x5_2[i])
for i in range(100):
    xAll.append(x5_3[i])
for i in range(100):
    xAll.append(x5_4[i])
for i in range(100):
    xAll.append(x5_5[i])
for i in range(100):
    xAll.append(x5_6[i])

for i in range(100):
    xAll.append(x5_7[i])
for i in range(100):
    xAll.append(x6_1[i])

for i in range(100):
    xAll.append(x6_2[i])
for i in range(100):
    xAll.append(x6_3[i])
for i in range(100):
    xAll.append(x6_4[i])
for i in range(100):
    xAll.append(x6_5[i])
for i in range(100):
    xAll.append(x6_6[i])
for i in range(100):
    xAll.append(x6_7[i])
for i in range(100):
    xAll.append(x7_1[i])
for i in range(100):
    xAll.append(x7_2[i])
for i in range(100):
    xAll.append(x7_3[i])
for i in range(100):
    xAll.append(x7_4[i])

for i in range(100):
    xAll.append(x7_5[i])

for i in range(100):
    xAll.append(x7_6[i])
for i in range(100):
    xAll.append(x7_7[i])
for i in range(100):
    xAll.append(x8_1[i])
for i in range(100):
    xAll.append(x8_2[i])
for i in range(100):
    xAll.append(x8_3[i])
for i in range(100):
    xAll.append(x8_4[i])
for i in range(100):
    xAll.append(x8_5[i])

for i in range(100):
    xAll.append(x8_6[i])
for i in range(100):
    xAll.append(x8_7[i])

for i in range(100):
    xAll.append(x9_1[i])
for i in range(100):
    xAll.append(x9_2[i])

for i in range(100):
    xAll.append(x9_3[i])

for i in range(100):
    xAll.append(x9_4[i])
for i in range(100):
    xAll.append(x9_5[i])

for i in range(100):
    xAll.append(x9_6[i])

for i in range(100):
    xAll.append(x9_7[i])

#print(len(xAll))
#light
l1=np.arange(10)
l1=np.loadtxt( r'D:\Desktop\test\1-light.txt', delimiter='\n',max_rows =100)

l1=np.around(l1,0)
#print('l1:',len(l1))
#ROI
R1= np.arange(10)
R1=np.loadtxt( r'D:\Desktop\test\2-ROI.txt', delimiter='\n',max_rows =100)
R1=np.around(R1,0)

#膚況
F1=np.arange(10)
F1=np.loadtxt( r'D:\Desktop\test\3-face.txt', delimiter='\n',max_rows =100)
F1=np.around(F1,0)

#距離
D1=np.arange(10)
D1=np.loadtxt( r'D:\Desktop\test\4-distance.txt', delimiter='\n',max_rows =100)
D1=np.around(D1,0)
print(D1)
x10_1=np.arange(1312)
x10_1=np.loadtxt( r'D:\Desktop\test\test010-1.txt', delimiter='\n',max_rows =100)
x10_1=np.around(x10_1,0)

x10_2= np.arange(1312)
x10_2= np.loadtxt( r'D:\Desktop\test\test010-2.txt', delimiter='\n',max_rows =100)
x10_2= np.around(x10_2,0)
#print(len(x10_2))

x10_3= np.arange(1312)
x10_3= np.loadtxt( r'D:\Desktop\test\test010-3.txt', delimiter='\n',max_rows =100)
x10_3= np.around(x10_3,0)
#print(len(x10_2))

x10_4= np.arange(1312)
x10_4= np.loadtxt( r'D:\Desktop\test\test010-4.txt', delimiter='\n',max_rows =100)
x10_4= np.around(x10_4,0)
#print(len(x10_2))

x10_5= np.arange(1312)
x10_5= np.loadtxt( r'D:\Desktop\test\test010-5.txt', delimiter='\n',max_rows =100)
x10_5= np.around(x10_5,0)
#print(len(x10_2))

x10_6= np.arange(1312)
x10_6= np.loadtxt( r'D:\Desktop\test\test010-6.txt', delimiter='\n',max_rows =100)
x10_6= np.around(x10_6,0)
#print(len(x10_2))

x10_7= np.arange(1312)
x10_7= np.loadtxt( r'D:\Desktop\test\test010-7.txt', delimiter='\n',max_rows =100)
x10_7= np.around(x10_7,0)
#print(len(x10_2))


###
p1=[*range(1, 16)]
#print(p1)
p2=[*range(1, 11)]
#print(p2)
d=[]
b=[]
for i in range(25):    
    d.append(round(random.random(),2))

#print(len(d))
for i in range(35):
    b.append(round(random.random(),2))
print(xAll[1]) 

# Blog post example:
nn = NeuralNetwork(5, 5, 7, hidden_layer_weights=d, hidden_layer_bias=0.35, output_layer_weights=b, output_layer_bias=0.5)
for i in range(10000):
  for j in range(1):
    nn.train([xAll[j],l1[j],R1[j],F1[j],D1[j]], [x10_1[j],x10_2[j],x10_3[j],x10_4[j],x10_5[j],x10_6[j],x10_7[j]])
    print('第',i,'次', round(nn.calculate_total_error([[[xAll[j],l1[j],R1[j],F1[j],D1[j]], [x10_1[j],x10_2[j],x10_3[j],x10_4[j],x10_5[j],x10_6[j],x10_7[j]]]]), 9))

# XOR example:
# training_sets = [
#     [[0, 0], [0]],
#     [[0, 1], [1]],
#     [[1, 0], [1]],
#     [[1, 1], [0]]
# ]
# nn = NeuralNetwork(len(training_sets[0][0]), 5, len(training_sets[0][1]))
# for i in range(10000):
#     training_inputs, training_outputs = random.choice(training_sets)
#     nn.train(training_inputs, training_outputs)
#     print(i, nn.calculate_total_error(training_sets))