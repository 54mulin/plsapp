import tkinter as tk
from tkinter import filedialog, messagebox, Listbox, MULTIPLE, Toplevel, ttk
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


class PLSApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PLS Regression GUI")
        self.data = None

        self.load_button = tk.Button(
            root, text="Load CSV", command=self.load_csv)
        self.load_button.pack(pady=20)

        self.field_frame = tk.Frame(root)
        self.field_frame.pack(pady=20)

        self.x_label = tk.Label(self.field_frame, text="X fields")
        self.x_label.grid(row=0, column=0)
        self.x_listbox = Listbox(self.field_frame, selectmode=MULTIPLE)
        self.x_listbox.grid(row=0, column=1)

        self.select_all_button = tk.Button(
            self.field_frame, text="Select All", command=self.select_all)
        self.select_all_button.grid(row=0, column=2)
        self.deselect_all_button = tk.Button(
            self.field_frame, text="Deselect All", command=self.deselect_all)
        self.deselect_all_button.grid(row=0, column=3)


        self.y_label = tk.Label(self.field_frame, text="y field")
        self.y_label.grid(row=1, column=0)
        self.y_combobox = ttk.Combobox(self.field_frame)
        self.y_combobox.grid(row=1, column=1)

        # self.wl_range_label = tk.Label(
        #     self.field_frame, text="WL range (start, stop, step)")
        # self.wl_range_label.grid(row=2, column=0)
        # self.wl_entry = tk.Entry(self.field_frame)
        # self.wl_entry.grid(row=2, column=1)

        self.fig1_x_label_label = tk.Label(
            self.field_frame, text="Fig 1 X Label")
        self.fig1_x_label_label.grid(row=2, column=0)
        self.fig1_x_label_entry = tk.Entry(self.field_frame)
        self.fig1_x_label_entry.grid(row=2, column=1)

        self.fig1_y_label_label = tk.Label(
            self.field_frame, text="Fig 1 Y Label")
        self.fig1_y_label_label.grid(row=3, column=0)
        self.fig1_y_label_entry = tk.Entry(self.field_frame)
        self.fig1_y_label_entry.grid(row=3, column=1)

        self.n_components_label = tk.Label(
            self.field_frame, text="n_components (not less than length of X)")
        self.n_components_label.grid(row=4, column=0)
        self.n_components_entry = tk.Entry(self.field_frame)
        self.n_components_entry.grid(row=4, column=1)

        self.windows_length_label = tk.Label(
            self.field_frame, text="windows length of savgol_filter (not less than length of X)")
        self.windows_length_label.grid(row=5, column=0)
        self.windows_length_entry = tk.Entry(self.field_frame)
        self.windows_length_entry.grid(row=5, column=1)

        self.results_text = tk.Text(root, height=10, width=80)
        self.results_text.pack(pady=20)

        self.plot_button = tk.Button(
            root, text="Plot & Train", command=self.plot_train)
        self.plot_button.pack(pady=20)

        self.save_button = tk.Button(
            root, text="Save Plots", command=self.save_plots)
        self.save_button.pack(pady=20)

    def load_csv(self):
        file_path = filedialog.askopenfilename(
            filetypes=[('CSV files', '*.csv')])
        if file_path:
            self.data = pd.read_csv(file_path)
            self.populate_fields()
            messagebox.showinfo("Info", "CSV loaded successfully!")

    def populate_fields(self):
        columns = self.data.columns.tolist()
        for col in columns:
            self.x_listbox.insert(tk.END, col)
        self.y_combobox['values'] = columns
        if columns:
            self.y_combobox.set(columns[0])

    def select_all(self):
        self.x_listbox.select_set(0, tk.END)

    def deselect_all(self):
        self.x_listbox.selection_clear(0, tk.END)

    def plot_train(self):
        if self.data is None:
            messagebox.showerror("Error", "Please load a CSV file first.")
            return

        x_fields = [self.x_listbox.get(idx)
                    for idx in self.x_listbox.curselection()]
        x_num = len(x_fields)
        y_field = self.y_combobox.get()
        wl_start, wl_stop, wl_step = 1100, 1100+len(x_fields), 1
        fig1_x_label = self.fig1_x_label_entry.get()
        fig1_y_label = self.fig1_y_label_entry.get()
        n_comp = int(self.n_components_entry.get())

        X = self.data[x_fields].values
        y = self.data[y_field].values
        wl = np.arange(wl_start, wl_stop, wl_step)

        if len(wl) != X.shape[1]:
            messagebox.showerror(
                "Error", "The length of WL range doesn't match the number of X fields.")
            return

        windows_length = int(self.windows_length_entry.get())
        X2 = savgol_filter(X, windows_length, polyorder=2, deriv=2)
        fig1 = plt.figure(figsize=(8, 4.5))
        plt.plot(wl, X2.T)
        plt.xlabel(fig1_x_label)
        plt.ylabel(fig1_y_label)
        plt.title('Savitzky-Golay Filtered Data')
        plt.tight_layout()

        mse = []
        component = np.arange(1, n_comp)
        for i in component:
            pls = PLSRegression(n_components=i)
            y_cv = cross_val_predict(pls, X2, y, cv=10)
            mse.append(mean_squared_error(y, y_cv))

        msemin = np.argmin(mse)
        fig2 = plt.figure(figsize=(8, 4.5))
        plt.plot(component, np.array(mse), '-v', color='blue', mfc='blue')
        plt.plot(component[msemin], np.array(mse)
                 [msemin], 'P', ms=10, mfc='red')
        plt.xlabel('Number of PLS components')
        plt.ylabel('MSE')
        plt.title('PLS')
        plt.xlim(left=-1)
        plt.tight_layout()

        pls_opt = PLSRegression(n_components=msemin+1)
        pls_opt.fit(X2, y)
        y_c = pls_opt.predict(X2)
        y_cv = cross_val_predict(pls_opt, X2, y, cv=10)
        score_c = r2_score(y, y_c)
        score_cv = r2_score(y, y_cv)
        mse_c = mean_squared_error(y, y_c)
        mse_cv = mean_squared_error(y, y_cv)

        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f'X fields: choose {x_num} columns\n')
        self.results_text.insert(tk.END, f'{x_fields}\n')
        self.results_text.insert(tk.END, f'R2 calib: {score_c:.3f}\n')
        self.results_text.insert(tk.END, f'R2 CV: {score_cv:.3f}\n')
        self.results_text.insert(tk.END, f'MSE calib: {mse_c:.3f}\n')
        self.results_text.insert(tk.END, f'MSE CV: {mse_cv:.3f}\n')

        z = np.polyfit(y, y_c, 1)
        fig3 = plt.figure(figsize=(9, 5))
        plt.scatter(y_c, y, c='red', edgecolors='k')
        plt.plot(np.polyval(z, y), y, c='blue', linewidth=1)
        plt.plot(y, y, color='green', linewidth=1)
        plt.title(f'$R^2$ (CV): {score_cv}')
        plt.xlabel('Predicted $^{\circ}$Brix')
        plt.ylabel('Measured $^{\circ}$Brix')
        plt.tight_layout()

        # Store all figures for saving
        self.figures = [fig1, fig2, fig3]

        # Show all plots
        plt.show()

    def save_plots(self):
        if not hasattr(self, 'figures') or not self.figures:
            messagebox.showerror("Error", "Please plot the data first.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                filetypes=[("PNG files", "*.png")])
        if file_path:
            for i, fig in enumerate(self.figures):
                filename = f'{file_path[:-4]}_{i+1}.png'
                fig.savefig(filename, bbox_inches='tight')

            messagebox.showinfo("Info", "Plots saved successfully!")


if __name__ == '__main__':
    root = tk.Tk()
    app = PLSApp(root)
    root.mainloop()
