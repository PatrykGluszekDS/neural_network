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
        self.entry_pad_x = 13
        self.label_pad_y = 6
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
        self.generate_button.grid(column=0, row=4, pady=self.button_pad_y)

        # Plot
        self.canvas_frame = Frame()
        self.canvas_frame.grid(column=0, row=5, columnspan=4)

        self.figure, self.ax = plt.subplots(figsize=(4, 4))
        self.ax.grid(True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.canvas_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(column=0, row=5, columnspan=4)

    def generate_data(self):
        num_modes = int(self.num_of_modes_entry.get())
        num_samples = int(self.num_of_samples_entry.get())

        data = self.generate_random_data(num_modes, num_samples)

        self.plot_data(data)

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

    def plot_data(self, data):
        self.ax.clear()
        self.ax.scatter(data[:, 0], data[:, 1], c=data[:, 2])
        self.ax.grid(True)
        self.canvas.draw()


if __name__ == "__main__":
    root = Tk()
    root.minsize(600, 400)
    app = DataGeneratorApp(root)
    root.mainloop()
    