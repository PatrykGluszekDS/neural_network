from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class DataGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Generation Program")

        self.standard_font = ("Arial", 18, "bold")
        self.entry_font = ("Arial", 16)
        self.button_font = ("Arial", 14, "bold")
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
        self.mean_min_entry.insert(0, "-1.0")
        self.mean_min_entry.grid(column=1, row=1, padx=self.entry_pad_x)

        self.mean_max_label = Label(text="Upper interval of the mean", font=self.standard_font)
        self.mean_max_label.grid(column=2, row=1, pady=self.label_pad_y)
        self.mean_max_entry = Entry(font=self.entry_font, width=self.entry_width)
        self.mean_max_entry.insert(0, "1.0")
        self.mean_max_entry.grid(column=3, row=1, padx=self.entry_pad_x)

        # Button
        self.generate_button = Button(text="Generate plot", command=self.generate_data, font=self.button_font)
        self.generate_button.grid(column=0, row=5, pady=self.button_pad_y)

        # Plot
        self.canvas_frame = Frame()
        self.canvas_frame.grid(column=0, row=6, columnspan=4)

        self.figure, self.ax = plt.subplots(figsize=(4, 4))
        self.ax.grid(True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.canvas_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(column=0, row=6, columnspan=4)

        # Dropdown for choosing activation function
        self.activation_label = Label(text="Choose Activation Function:", font=self.standard_font)
        self.activation_label.grid(column=0, row=4, columnspan=2)

        self.activation_var = StringVar(root)
        self.activation_var.set("heaviside")  # Default activation function
        self.activation_dropdown = OptionMenu(root, self.activation_var, "heaviside", "logistic", "sin", "tanh",
                                              "sign", "relu", "leaky_relu")
        self.activation_dropdown.grid(column=2, row=4, columnspan=2)
        self.activation_dropdown.config(font=self.entry_font)

        menu = self.activation_dropdown["menu"]
        menu.config(font=self.entry_font)

        # Neuron initialization
        self.neuron = Neuron()

    def generate_data(self):
        num_modes = int(self.num_of_modes_entry.get())
        num_samples = int(self.num_of_samples_entry.get())

        self.data = self.generate_random_data(num_modes, num_samples)

        # Train the neuron on generated data
        self.train_neuron(self.data)

        # Plot the decision boundary
        self.plot_decision_boundary()

    def generate_random_data(self, num_modes, num_samples):
        mean_low = float(self.mean_min_entry.get())
        mean_high = float(self.mean_max_entry.get())
        variance_low = float(self.var_min_entry.get())
        variance_high = float(self.var_max_entry.get())

        data = []

        for class_label in [0, 1]:
            for _ in range(num_modes):
                mean = np.random.uniform(low=mean_low, high=mean_high, size=2)
                variance = np.random.uniform(variance_low, variance_high, size=2)
                samples = np.random.normal(mean, np.sqrt(variance), size=(num_samples, 2))
                labels = np.full((num_samples, 1), class_label)
                data.append(np.hstack((samples, labels)))

        return np.concatenate(data, axis=0)

    def train_neuron(self, data):
        learning_rate = 0.01
        epochs = 1000
        convergence_threshold = 1e-8

        activation_function = self.activation_var.get()

        for epoch in range(epochs):
            for sample in data:
                features = sample[:2]
                expected_label = sample[2]
                if self.neuron.train(features, expected_label, learning_rate, activation_function,
                                     convergence_threshold):
                    print(f"Converged at epoch {epoch}. Stopping training.")
                    return

    def plot_decision_boundary(self):
        self.ax.clear()
        self.ax.scatter(self.data[:, 0], self.data[:, 1], c=self.data[:, 2], edgecolor='k')

        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        Z = self.neuron.predict(np.c_[xx.ravel(), yy.ravel()], self.activation_var.get())

        Z = Z.reshape(xx.shape)

        self.ax.contourf(xx, yy, Z, alpha=0.3)
        self.ax.grid(True)

        self.canvas.draw()


class Neuron:
    def __init__(self):
        self.weights = np.random.rand(2)
        self.bias = np.random.rand(1)

    def train(self, features, expected_label, learning_rate, activation_function, convergence_threshold=1e-5):
        previous_weights = self.weights.copy()
        previous_bias = self.bias.copy()

        predicted_label = self.predict(features, activation_function)
        error = expected_label - predicted_label

        activation_derivative = self.get_activation_derivative(np.dot(features, self.weights), activation_function)

        self.weights += learning_rate * error * activation_derivative * features
        self.bias += learning_rate * error * np.sum(activation_derivative)

        # Check convergence
        weight_change = np.linalg.norm(self.weights - previous_weights)
        bias_change = np.abs(self.bias - previous_bias)

        if weight_change < convergence_threshold and bias_change < convergence_threshold:
            print("Converged. Stopping training.")
            return True

        return False

    def predict(self, features, activation_function):
        linear_output = np.dot(features, self.weights) + self.bias
        return self.apply_activation(linear_output, activation_function)

    def apply_activation(self, x, activation_function):
        if activation_function == "heaviside":
            return np.heaviside(x, 0)
        elif activation_function == "logistic":
            return 1 / (1 + np.exp(-x))
        elif activation_function == "sin":
            return np.sin(x)
        elif activation_function == "tanh":
            return np.tanh(x)
        elif activation_function == "sign":
            return np.sign(x)
        elif activation_function == "relu":
            return np.maximum(0, x)
        elif activation_function == "leaky_relu":
            alpha = 0.01
            return np.where(x > 0, x, alpha * x)

    def get_activation_derivative(self, x, activation_function):
        if activation_function == "heaviside":
            return 1
        elif activation_function == "logistic":
            return (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))
        elif activation_function == "sin":
            return np.cos(x)
        elif activation_function == "tanh":
            return (1 - np.cosh(x))**2
        elif activation_function == "sign":
            return 1
        elif activation_function == "relu":
            return np.heaviside(x, 0)
        elif activation_function == "leaky_relu":
            alpha = 0.01
            return np.where(x > 0, 1, alpha)


if __name__ == "__main__":
    root = Tk()
    app = DataGeneratorApp(root)
    root.mainloop()
