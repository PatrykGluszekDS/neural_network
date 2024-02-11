from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import scipy.special as sc


class DataGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Generation Program")

        self.standard_font = ("Arial", 16, "bold")
        self.entry_font = ("Arial", 14)
        self.button_font = ("Arial", 12, "bold")
        self.entry_width = 7
        self.entry_pad_x = 15
        self.label_pad_y = 7
        self.button_pad_y = 10

        # Labels with entries
        self.num_of_modes_label = Label(text="Number of modes per class:", font=self.standard_font)
        self.num_of_modes_label.grid(column=0, row=2, columnspan=2, pady=self.label_pad_y)
        self.num_of_modes_entry = Entry(font=self.entry_font, width=self.entry_width)
        self.num_of_modes_entry.grid(column=2, row=2, columnspan=2, padx=self.entry_pad_x)

        self.num_of_samples_label = Label(text="Number of samples per mode:", font=self.standard_font)
        self.num_of_samples_label.grid(column=0, row=3, columnspan=2, pady=self.label_pad_y)
        self.num_of_samples_entry = Entry(font=self.entry_font, width=self.entry_width)
        self.num_of_samples_entry.grid(column=2, row=3, columnspan=2, padx=self.entry_pad_x)

        self.var_min_label = Label(text="Lower interval of variance", font=self.standard_font)
        self.var_min_label.grid(column=0, row=0, pady=self.label_pad_y)
        self.var_min_entry = Entry(font=self.entry_font, width=self.entry_width)
        self.var_min_entry.insert(0, "0.1")
        self.var_min_entry.grid(column=1, row=0, padx=self.entry_pad_x)

        self.var_max_label = Label(text="Upper interval of variance", font=self.standard_font)
        self.var_max_label.grid(column=2, row=0, pady=self.label_pad_y)
        self.var_max_entry = Entry(font=self.entry_font, width=self.entry_width)
        self.var_max_entry.insert(0, "0.5")
        self.var_max_entry.grid(column=3, row=0, padx=self.entry_pad_x)

        self.mean_min_label = Label(text="Lower interval of the mean", font=self.standard_font)
        self.mean_min_label.grid(column=0, row=1, pady=self.label_pad_y)
        self.mean_min_entry = Entry(font=self.entry_font, width=self.entry_width)
        self.mean_min_entry.insert(0, "-5.0")
        self.mean_min_entry.grid(column=1, row=1, padx=self.entry_pad_x)

        self.mean_max_label = Label(text="Upper interval of the mean", font=self.standard_font)
        self.mean_max_label.grid(column=2, row=1, pady=self.label_pad_y)
        self.mean_max_entry = Entry(font=self.entry_font, width=self.entry_width)
        self.mean_max_entry.insert(0, "5.0")
        self.mean_max_entry.grid(column=3, row=1, padx=self.entry_pad_x)

        # Button
        self.generate_button = Button(text="Generate plot", command=self.generate_random_data, font=self.button_font)
        self.generate_button.grid(column=0, row=5, pady=self.button_pad_y)

        # Plot
        self.canvas_frame = Frame()
        self.canvas_frame.grid(column=0, row=6, columnspan=4)

        self.figure, self.ax = plt.subplots(figsize=(4, 4))
        self.ax.grid(True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.canvas_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(column=0, row=6, columnspan=4)

        # Numbers of Input Neurons
        self.input_neurons_label = Label(text="Number of neurons in the input layer:", font=self.standard_font)
        self.input_neurons_label.grid(column=0, row=4)
        self.input_neurons_entry = Entry(font=self.entry_font, width=self.entry_width)
        self.input_neurons_entry.insert(0, "2")
        self.input_neurons_entry.grid(column=1, row=4)

        # Numbers of Hidden Neurons
        self.hidden_neurons_label = Label(text="Number of neurons in the hidden layer:", font=self.standard_font)
        self.hidden_neurons_label.grid(column=2, row=4)
        self.hidden_neurons_entry = Entry(font=self.entry_font, width=self.entry_width)
        self.hidden_neurons_entry.insert(0, "10")
        self.hidden_neurons_entry.grid(column=3, row=4)

        # Visualize button
        self.visualize_button = Button(text="Train network and plot", command=self.visualize, font=self.button_font)
        self.visualize_button.grid(column=2, row=5, pady=self.button_pad_y)

        self.generated_inputs = None
        self.generated_targets = None

        # Initial values
        self.initial_input_weights = None
        self.biases_input = None

        self.weights_hidden = None
        self.biases_hidden = None

        self.weights_output = None
        self.biases_output = None

        self.nmin = 0.0001
        self.nmax = 0.01
        self.Nmax = 1000
        self.learning_rate = 0.003

    def generate_random_data(self):
        num_modes = int(self.num_of_modes_entry.get())
        num_samples = int(self.num_of_samples_entry.get())

        mean_low = float(self.mean_min_entry.get())
        mean_high = float(self.mean_max_entry.get())
        variance_low = float(self.var_min_entry.get())
        variance_high = float(self.var_max_entry.get())

        data = []

        self.initial_input_weights = np.random.randn(2, int(self.input_neurons_entry.get())) * np.sqrt(1 / 2)
        self.biases_input = np.random.randn(1, int(self.input_neurons_entry.get()))

        self.weights_hidden = np.random.randn(int(self.input_neurons_entry.get()),
                                              int(self.hidden_neurons_entry.get())) * np.sqrt(
            1 / int(self.input_neurons_entry.get()))
        self.biases_hidden = np.random.randn(1, int(self.hidden_neurons_entry.get()))

        self.weights_output = np.random.randn(int(self.hidden_neurons_entry.get()), 2) * np.sqrt(
            1 / int(self.hidden_neurons_entry.get()))  # Fixed 2 output neurons
        self.biases_output = np.random.randn(1, 2)

        for class_label in [0, 1]:
            for _ in range(num_modes):
                mean = np.random.uniform(low=mean_low, high=mean_high, size=2)
                variance = np.random.uniform(variance_low, variance_high, size=2)
                samples = np.random.normal(mean, np.sqrt(variance), size=(num_samples, 2))
                labels = np.full((num_samples, 1), class_label)
                data.append(np.hstack((samples, labels)))

        concatenated_data = np.concatenate(data, axis=0)

        input_samples = concatenated_data[:, :2]
        target_labels = concatenated_data[:, 2]
        color_for_points = target_labels
        target_labels = np.array([[1, 0] if label == 0 else [0, 1] for label in target_labels])

        self.ax.clear()
        self.ax.scatter(input_samples[:, 0], input_samples[:, 1], c=color_for_points, cmap="coolwarm")
        self.ax.grid(True)
        self.canvas.draw()

        self.generated_inputs = input_samples
        self.generated_targets = target_labels

    # Activation function
    def logistic(self, x):
        return sc.expit(x)

    def logistic_derivative(self, x):
        return self.logistic(x) * (1 - self.logistic(x))

    def train_neural_network(self, initial_input_weights, weights_hidden, weights_output, input_samples,
                             target_labels, biases_output, biases_hidden, biases_input):

        for epoch in range(self.Nmax):
            shuffled_indices = np.random.permutation(len(input_samples))
            input_samples = np.array([input_samples[i] for i in shuffled_indices])
            target_labels = np.array([target_labels[i] for i in shuffled_indices])

            # learning_rate = self.nmin + (self.nmax - self.nmin) * (1 + np.cos((epoch / self.Nmax) * np.pi))
            learning_rate = self.learning_rate

            # Forward propagation
            a = np.dot(input_samples, initial_input_weights) + biases_input
            input_layer_output = self.logistic(a)

            b = np.dot(input_layer_output, weights_hidden) + biases_hidden
            hidden_layer_output = self.logistic(b)

            c = np.dot(hidden_layer_output, weights_output) + biases_output
            final_output = self.logistic(c)

            # Backward propagation

            output_error = (target_labels - final_output) * self.logistic_derivative(c)  # output layer
            hidden_error = np.dot(output_error, weights_output.T) * self.logistic_derivative(
                b)  # hidden layer
            input_error = np.dot(hidden_error, weights_hidden.T) * self.logistic_derivative(
                a)  # input layer

            # Output layer
            delta_output = learning_rate * np.dot(hidden_layer_output.T, output_error)
            weights_output += delta_output
            delta_biases_output = learning_rate * output_error
            biases_output += np.mean(delta_biases_output, axis=0, keepdims=True)

            # Hidden layer
            delta_hidden = learning_rate * np.dot(input_layer_output.T, hidden_error)
            weights_hidden += delta_hidden
            delta_biases_hidden = learning_rate * hidden_error
            biases_hidden += np.mean(delta_biases_hidden, axis=0, keepdims=True)

            # Input layer
            delta_input = learning_rate * np.dot(input_samples.T, input_error)
            initial_input_weights += delta_input
            delta_biases_input = learning_rate * input_error
            biases_input += np.mean(delta_biases_input, axis=0, keepdims=True)

        return initial_input_weights, weights_hidden, weights_output, biases_input, biases_hidden, biases_output

    def visualize(self):
        trained_weights = self.train_neural_network(self.initial_input_weights, self.weights_hidden,
                                                    self.weights_output, self.generated_inputs,
                                                    self.generated_targets, self.biases_output,
                                                    self.biases_hidden, self.biases_input)

        bias_input, bias_hidden, bias_output = trained_weights[3], trained_weights[4], trained_weights[5]

        x_min, x_max = self.generated_inputs[:, 0].min() - 1, self.generated_inputs[:, 0].max() + 1
        y_min, y_max = self.generated_inputs[:, 1].min() - 1, self.generated_inputs[:, 1].max() + 1

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        input_layer_pred = self.logistic(np.dot(grid_points, trained_weights[0]) + bias_input)
        hidden_layer_pred = self.logistic(np.dot(input_layer_pred, trained_weights[1]) + bias_hidden)
        final_output_pred = self.logistic(np.dot(hidden_layer_pred, trained_weights[2]) + bias_output)
        final_output_pred = np.array([0 if point[0] > point[1] else 1 for point in final_output_pred])
        final_output_pred = final_output_pred.reshape(xx.shape)
        predictions_grid = final_output_pred

        # Plot decision boundary
        self.ax.contourf(xx, yy, predictions_grid, levels=[0, 0.5, 1], colors=['blue', 'red'],
                    alpha=0.3)  # Decision boundary

        self.canvas.draw()


if __name__ == "__main__":
    root = Tk()
    root.minsize(600, 400)
    app = DataGeneratorApp(root)
    root.mainloop()
