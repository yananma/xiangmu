import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mping 
import math

from multilayer_perceptron import MultilayerPerceptron


data = pd.read_csv('../data/mnist-demo.csv')
numbers_to_display = 25
num_cells = math.ceil(math.sqrt(numbers_to_display))
plt.figure(figsize=(10,10))
for plot_index in range(numbers_to_display):
    digit = data[plot_index:plot_index+1].values
    digit_label = digit[0][0]
    digit_pixels = digit[0][1:]
    image_size = int(math.sqrt(digit_pixels.shape[0]))
    frame = digit_pixels.reshape((image_size,image_size))
    plt.subplot(num_cells,num_cells,plot_index+1)
    plt.imshow(frame,cmap='Greys')
    plt.title(digit_label)
plt.subplots_adjust(wspace=0.5,hspace=0.5)
plt.show()

train_data = data.sample(frac = 0.8)
test_data = data.drop(train_data.index)

train_data = train_data.values
test_data = test_data.values

num_training_examples = 5000

x_train = train_data[:num_training_examples,1:]
y_train = train_data[:num_training_examples,[0]]

x_test = test_data[:,1:]
y_test = test_data[:,[0]]


layers=[784,25,10]

normalize_data = True
max_iterations = 500
alpha = 0.1


multilayer_perceptron = MultilayerPerceptron(x_train,y_train,layers,normalize_data)
(thetas,costs) = multilayer_perceptron.train(max_iterations,alpha)
plt.plot(range(len(costs)),costs)
plt.xlabel('Grident steps')
plt.xlabel('costs')
plt.show()


y_train_predictions = multilayer_perceptron.predict(x_train)
y_test_predictions = multilayer_perceptron.predict(x_test)

train_p = np.sum(y_train_predictions == y_train)/y_train.shape[0] * 100
test_p = np.sum(y_test_predictions == y_test)/y_test.shape[0] * 100
print ('训练集准确率：',train_p)
print ('测试集准确率：',test_p)

numbers_to_display = 64

num_cells = math.ceil(math.sqrt(numbers_to_display))

plt.figure(figsize=(15, 15))

for plot_index in range(numbers_to_display):
    digit_label = y_test[plot_index, 0]
    digit_pixels = x_test[plot_index, :]
    
    predicted_label = y_test_predictions[plot_index][0]

    image_size = int(math.sqrt(digit_pixels.shape[0]))
    
    frame = digit_pixels.reshape((image_size, image_size))
    
    color_map = 'Greens' if predicted_label == digit_label else 'Reds'
    plt.subplot(num_cells, num_cells, plot_index + 1)
    plt.imshow(frame, cmap=color_map)
    plt.title(predicted_label)
    plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.show()




