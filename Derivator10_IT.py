# multi_analysis.py - Fit polinomiale con grado selezionabile
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from numpy.polynomial import Polynomial
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import ScalarFormatter
import os
import re
from datetime import datetime

class MultiAnalysisApp:
    def __init__(self, root=None):
        if root is None:
            self.root = tk.Tk()
        else:
            self.root = root
        self.root.title("Derivatore 1.0 - fit polinomiale")
        self.root.geometry("950x750")
        
        self.datasets = []          # lista di dict: {'name': str, 'x': array, 'y': array}
        self.current_dataset = None
        self.preview_poly = None     # polinomio dell'anteprima
        self.preview_curve = None    # linea del fit nell'anteprima
        
        self.create_widgets()
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        if root is None:
            self.root.mainloop()
    
    def create_widgets(self):
        # Frame superiore
        top_frame = ttk.Frame(self.root, padding="5")
        top_frame.pack(fill=tk.X)
        
        ttk.Button(top_frame, text="Carica file multipli", command=self.load_multiple_files).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(top_frame, text="Dataset attivo:").pack(side=tk.LEFT, padx=(20,5))
        self.dataset_var = tk.StringVar()
        self.dataset_menu = ttk.Combobox(top_frame, textvariable=self.dataset_var, state="readonly", width=40)
        self.dataset_menu.pack(side=tk.LEFT, padx=5)
        self.dataset_menu.bind("<<ComboboxSelected>>", self.on_dataset_selected)
        
        ttk.Button(top_frame, text="Cancella dataset attivo", command=self.delete_current_dataset).pack(side=tk.LEFT, padx=10)
        
        # Parametri analisi
        param_frame = ttk.LabelFrame(self.root, text="Parametri fit polinomiale", padding="5")
        param_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(param_frame, text="X min (per fit):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.xmin_entry = ttk.Entry(param_frame, width=10)
        self.xmin_entry.grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(param_frame, text="X max (per fit):").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        self.xmax_entry = ttk.Entry(param_frame, width=10)
        self.xmax_entry.grid(row=0, column=3, padx=5, pady=2)
        
        ttk.Label(param_frame, text="Grado polinomio:").grid(row=0, column=4, sticky=tk.W, padx=5, pady=2)
        self.degree_var = tk.IntVar(value=6)
        self.degree_spin = ttk.Spinbox(param_frame, from_=1, to=9, textvariable=self.degree_var, width=5)
        self.degree_spin.grid(row=0, column=5, padx=5, pady=2)
        
        self.batch_plot_button = ttk.Button(param_frame, text="Plotta fit per TUTTI", command=self.open_fit_window)
        self.batch_plot_button.grid(row=1, column=2, columnspan=2, padx=5, pady=2)
        
        self.preview_fit_button = ttk.Button(param_frame, text="Anteprima fit (primo file)", command=self.preview_fit)
        self.preview_fit_button.grid(row=1, column=4, columnspan=2, padx=5, pady=2)
        
        # Frame pendenza desiderata
        slope_frame = ttk.LabelFrame(self.root, text="Trova X per pendenza desiderata (derivata prima)", padding="5")
        slope_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(slope_frame, text="Pendenza desiderata:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.slope_var = tk.StringVar()
        self.slope_entry = ttk.Entry(slope_frame, textvariable=self.slope_var, width=10)
        self.slope_entry.grid(row=0, column=1, padx=5, pady=2)
        self.slope_result = ttk.Label(slope_frame, text="X = ---", relief=tk.SUNKEN, width=20)
        self.slope_result.grid(row=0, column=2, padx=5, pady=2, sticky=tk.W)
        
        self.batch_slope_button = ttk.Button(slope_frame, text="Trova X per TUTTI (salva CSV)", command=self.batch_find_slope)
        self.batch_slope_button.grid(row=0, column=3, padx=5, pady=2)
        
        self.slope_var.trace('w', lambda *args: self.find_slope())
        
        # Area grafico principale
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.fig, self.ax = plt.subplots(figsize=(6,4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.coord_label = ttk.Label(self.root, text="Mouse: x = ---, y = ---", relief=tk.SUNKEN, anchor=tk.W)
        self.coord_label.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=2)
        
        self.info_label = ttk.Label(self.root, text="Nessun dataset caricato", relief=tk.SUNKEN, anchor=tk.W)
        self.info_label.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=2)
    
    # Metodi mouse e zoom (invariati)
    def on_mouse_move(self, event):
        if event.inaxes == self.ax:
            self.coord_label.config(text=f"Mouse: x = {event.xdata:.6f}, y = {event.ydata:.6f}")
        else:
            self.coord_label.config(text="Mouse: x = ---, y = ---")
    
    def on_scroll(self, event):
        if event.inaxes != self.ax:
            return
        scale = 1.1
        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None:
            return
        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()
        if event.step > 0:
            xmin = xdata - (xdata - xmin) / scale
            xmax = xdata + (xmax - xdata) / scale
            ymin = ydata - (ydata - ymin) / scale
            ymax = ydata + (ymax - ydata) / scale
        else:
            xmin = xdata - (xdata - xmin) * scale
            xmax = xdata + (xmax - xdata) * scale
            ymin = ydata - (ydata - ymin) * scale
            ymax = ydata + (ymax - ydata) * scale
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.canvas.draw_idle()
    
    def delete_current_dataset(self):
        if self.current_dataset is not None and self.current_dataset < len(self.datasets):
            del self.datasets[self.current_dataset]
            self.update_dataset_menu()
            self.update_plot()
            self.info_label.config(text=f"Dataset cancellato. Rimangono {len(self.datasets)} dataset.")
        else:
            messagebox.showwarning("Attenzione", "Nessun dataset selezionato da cancellare.")
    
    def preview_fit(self):
        """Fit polinomiale di grado selezionato sul primo dataset."""
        if not self.datasets:
            messagebox.showwarning("Attenzione", "Nessun dataset caricato.")
            return
        try:
            xmin = float(self.xmin_entry.get())
            xmax = float(self.xmax_entry.get())
            degree = self.degree_var.get()
        except:
            messagebox.showerror("Errore", "Inserisci valori numerici per X min, X max e grado valido.")
            return
        if xmin >= xmax:
            messagebox.showerror("Errore", "X min deve essere minore di X max.")
            return
        if degree < 1 or degree > 9:
            messagebox.showerror("Errore", "Grado polinomio deve essere tra 1 e 9.")
            return
        
        ds = self.datasets[0]
        x_data = ds['x']
        y_data = ds['y']
        mask = (x_data >= xmin) & (x_data <= xmax)
        x_fit = x_data[mask]
        y_fit = y_data[mask]
        if len(x_fit) < degree + 1:
            messagebox.showerror("Errore", f"Sono necessari almeno {degree+1} punti per un polinomio di grado {degree}. (Disponibili: {len(x_fit)})")
            return
        
        # Fit polinomiale
        coefs = np.polyfit(x_fit, y_fit, degree)
        self.preview_poly = np.poly1d(coefs)
        
        # Rimuovi vecchia curva
        if self.preview_curve:
            self.preview_curve.remove()
        x_curve = np.linspace(xmin, xmax, 200)
        y_curve = self.preview_poly(x_curve)
        self.preview_curve, = self.ax.plot(x_curve, y_curve, 'r-', linewidth=2, label=f'Fit grado {degree}')
        self.canvas.draw_idle()
        messagebox.showinfo("Anteprima", f"Fit polinomiale di grado {degree} visualizzato.\nPolinomio:\n{self.preview_poly}")
    
    def find_slope(self):
        """Per il dataset corrente, trova X dove la derivata del polinomio eguaglia la pendenza desiderata."""
        if self.current_dataset is None or self.current_dataset >= len(self.datasets):
            self.slope_result.config(text="X = ---")
            return
        try:
            target_slope = float(self.slope_var.get())
        except:
            self.slope_result.config(text="X = ---")
            return
        
        ds = self.datasets[self.current_dataset]
        x_data = ds['x']
        y_data = ds['y']
        if len(x_data) == 0:
            self.slope_result.config(text="X = ---")
            return
        
        try:
            xmin = float(self.xmin_entry.get())
            xmax = float(self.xmax_entry.get())
            degree = self.degree_var.get()
        except:
            self.slope_result.config(text="X = ---")
            return
        
        mask = (x_data >= xmin) & (x_data <= xmax)
        x_fit = x_data[mask]
        y_fit = y_data[mask]
        if len(x_fit) < degree + 1:
            self.slope_result.config(text=f"X = --- (occorrono {degree+1} punti)")
            return
        
        coefs = np.polyfit(x_fit, y_fit, degree)
        poly = np.poly1d(coefs)
        deriv = poly.deriv()
        # Risolvi deriv(x) = target_slope -> deriv(x) - target_slope = 0
        diff_coefs = deriv.coefficients
        diff_coefs[-1] -= target_slope
        radici = np.roots(diff_coefs)
        radici_real = [r.real for r in radici if np.isreal(r) and xmin <= r.real <= xmax]
        if not radici_real:
            self.slope_result.config(text="X = --- (nessuna soluzione)")
            return
        radici_real.sort(reverse=True)
        best_x = radici_real[0]
        self.slope_result.config(text=f"X = {best_x:.6f}")
    
    # --- Funzioni di importazione (invariate) ---
    def clean_number(self, s):
        if isinstance(s, str):
            s = s.replace(',', '').strip()
            if s == '':
                return np.nan
            try:
                return float(s)
            except:
                return np.nan
        return np.nan
    
    def read_file_custom(self, filepath, sep, skip, has_header):
        sep_map = {"tab": "\t", "spazio": " ", "virgola": ",", "punto e virgola": ";"}
        separator = sep_map.get(sep, "\t")
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()[skip:]
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(separator)
                if len(parts) < 2:
                    continue
                cleaned = [self.clean_number(p) for p in parts]
                if any(np.isnan(cleaned)):
                    continue
                data.append(cleaned)
        if not data:
            return None, None
        data_arr = np.array(data, dtype=float)
        n_cols = data_arr.shape[1]
        if has_header:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()[skip:]
                if not lines:
                    headers = [f"Colonna {i+1}" for i in range(n_cols)]
                else:
                    first_line = lines[0].strip()
                    if first_line:
                        parts = first_line.split(separator)
                        if len(parts) >= 2:
                            headers = parts[:n_cols]
                        else:
                            headers = [f"Colonna {i+1}" for i in range(n_cols)]
                    else:
                        headers = [f"Colonna {i+1}" for i in range(n_cols)]
            data_arr = data_arr[1:] if len(data_arr) > 1 else data_arr
        else:
            headers = [f"Colonna {i+1}" for i in range(n_cols)]
        return data_arr, headers
    
    def show_import_dialog(self, file_path):
        import_dialog = tk.Toplevel(self.root)
        import_dialog.title("Opzioni di importazione")
        import_dialog.geometry("700x500")
        import_dialog.transient(self.root)
        import_dialog.grab_set()
        
        skip_rows = tk.IntVar(value=0)
        header = tk.BooleanVar(value=False)
        delimiter = tk.StringVar(value="tab")
        
        ttk.Label(import_dialog, text="Numero di righe da saltare all'inizio:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Spinbox(import_dialog, from_=0, to=100, textvariable=skip_rows, width=5).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Checkbutton(import_dialog, text="Utilizza la prima riga come intestazione", variable=header).grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        ttk.Label(import_dialog, text="Separatore:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        delim_combo = ttk.Combobox(import_dialog, textvariable=delimiter, values=["tab", "spazio", "virgola", "punto e virgola"], state="readonly")
        delim_combo.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        delim_combo.current(0)
        
        preview_frame = ttk.LabelFrame(import_dialog, text="Anteprima dati")
        preview_frame.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        import_dialog.grid_rowconfigure(3, weight=1)
        import_dialog.grid_columnconfigure(0, weight=1)
        
        tree_frame = ttk.Frame(preview_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        tree_scroll_y = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL)
        tree_scroll_x = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)
        tree = ttk.Treeview(tree_frame, yscrollcommand=tree_scroll_y.set, xscrollcommand=tree_scroll_x.set)
        tree_scroll_y.config(command=tree.yview)
        tree_scroll_x.config(command=tree.xview)
        tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        tree.pack(fill=tk.BOTH, expand=True)
        
        col_x = tk.StringVar()
        col_y = tk.StringVar()
        col_frame = ttk.Frame(import_dialog)
        col_frame.grid(row=4, column=0, columnspan=2, padx=5, pady=5)
        
        result = [None]
        
        def preview():
            sep = delimiter.get()
            skip = skip_rows.get()
            has_header = header.get()
            try:
                data_arr, headers = self.read_file_custom(file_path, sep, skip, has_header)
                if data_arr is None:
                    messagebox.showerror("Errore", "Il file non contiene dati validi con almeno due colonne.")
                    return
                for item in tree.get_children():
                    tree.delete(item)
                tree["columns"] = headers
                tree["show"] = "headings"
                for col in headers:
                    tree.heading(col, text=col)
                    tree.column(col, width=100, anchor=tk.CENTER)
                for row in data_arr[:10]:
                    tree.insert("", tk.END, values=[f"{val:.6f}" for val in row])
                import_dialog.preview_data = data_arr
                import_dialog.headers = headers
                for w in col_frame.winfo_children():
                    w.destroy()
                ttk.Label(col_frame, text="Seleziona colonna X:").grid(row=0, column=0, sticky=tk.W, padx=5)
                x_combo = ttk.Combobox(col_frame, textvariable=col_x, values=headers, state="readonly")
                x_combo.grid(row=0, column=1, padx=5)
                x_combo.current(0)
                ttk.Label(col_frame, text="Seleziona colonna Y:").grid(row=1, column=0, sticky=tk.W, padx=5)
                y_combo = ttk.Combobox(col_frame, textvariable=col_y, values=headers, state="readonly")
                y_combo.grid(row=1, column=1, padx=5)
                y_combo.current(1 if len(headers)>1 else 0)
                ttk.Label(import_dialog, text=f"Lette {len(data_arr)} righe valide", foreground="blue").grid(row=6, column=0, columnspan=2, padx=5, pady=5)
            except Exception as e:
                messagebox.showerror("Errore", f"Lettura fallita: {str(e)}")
        
        def confirm():
            if not hasattr(import_dialog, 'preview_data'):
                messagebox.showerror("Errore", "Premi 'Anteprima' prima di confermare.")
                return
            if not col_x.get() or not col_y.get():
                messagebox.showerror("Errore", "Seleziona le colonne X e Y.")
                return
            result[0] = (import_dialog.preview_data, import_dialog.headers, col_x.get(), col_y.get(),
                         skip_rows.get(), header.get(), delimiter.get())
            import_dialog.destroy()
        
        ttk.Button(import_dialog, text="Anteprima", command=preview).grid(row=5, column=0, padx=5, pady=10)
        ttk.Button(import_dialog, text="Conferma", command=confirm).grid(row=5, column=1, padx=5, pady=10)
        ttk.Button(import_dialog, text="Annulla", command=import_dialog.destroy).grid(row=5, column=2, padx=5, pady=10)
        
        self.root.wait_window(import_dialog)
        return result[0]
    
    def load_multiple_files(self):
        file_paths = filedialog.askopenfilenames(
            title="Seleziona più file",
            filetypes=[("File di dati", "*.txt *.csv"), ("Tutti i file", "*.*")]
        )
        if not file_paths:
            return
        result = self.show_import_dialog(file_paths[0])
        if result is None:
            return
        data_arr, headers, col_x, col_y, skip_rows, header_flag, sep = result
        for fpath in file_paths:
            try:
                data_arr, _ = self.read_file_custom(fpath, sep, skip_rows, header_flag)
                if data_arr is None:
                    continue
                col_x_idx = headers.index(col_x)
                col_y_idx = headers.index(col_y)
                x_data = data_arr[:, col_x_idx]
                y_data = data_arr[:, col_y_idx]
                name = os.path.basename(fpath)
                self.datasets.append({'name': name, 'x': x_data, 'y': y_data})
            except Exception as e:
                messagebox.showerror("Errore", f"Errore import {fpath}: {str(e)}")
        self.update_dataset_menu()
        self.update_plot()
        self.info_label.config(text=f"Caricati {len(file_paths)} file")
    
    def update_dataset_menu(self):
        names = [ds['name'] for ds in self.datasets]
        self.dataset_menu['values'] = names
        if names:
            self.dataset_menu.current(0)
            self.current_dataset = 0
        else:
            self.dataset_menu.set('')
            self.current_dataset = None
        self.find_slope()
    
    def on_dataset_selected(self, event):
        if self.dataset_menu.current() >= 0:
            self.current_dataset = self.dataset_menu.current()
            self.update_plot()
            self.find_slope()
    
    def update_plot(self):
        self.ax.clear()
        if self.datasets:
            colors = plt.cm.tab10(np.linspace(0, 1, len(self.datasets)))
            for i, ds in enumerate(self.datasets):
                color = colors[i % len(colors)]
                self.ax.scatter(ds['x'], ds['y'], s=5, label=ds['name'], color=color, alpha=0.7)
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.set_title("Dati caricati")
            self.ax.legend(loc='best', fontsize='small')
            self.ax.grid(True)
            self.ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            self.ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        self.canvas.draw()
        self.root.update_idletasks()
    
    def batch_find_slope(self):
        """Per ogni dataset, calcola il polinomio (grado selezionato) e trova X per pendenza target, salva CSV."""
        if not self.datasets:
            messagebox.showwarning("Attenzione", "Carica prima almeno un dataset.")
            return
        try:
            target_slope = float(self.slope_var.get())
        except:
            messagebox.showerror("Errore", "Inserisci un valore numerico per la pendenza desiderata.")
            return
        try:
            xmin = float(self.xmin_entry.get())
            xmax = float(self.xmax_entry.get())
            degree = self.degree_var.get()
        except:
            messagebox.showerror("Errore", "Inserisci X min, X max e grado validi.")
            return
        results = []
        for ds in self.datasets:
            x_data = ds['x']
            y_data = ds['y']
            mask = (x_data >= xmin) & (x_data <= xmax)
            x_fit = x_data[mask]
            y_fit = y_data[mask]
            if len(x_fit) < degree + 1:
                results.append((ds['name'], np.nan, f"punti insufficienti (richiesti {degree+1})"))
                continue
            try:
                coefs = np.polyfit(x_fit, y_fit, degree)
                poly = np.poly1d(coefs)
                deriv = poly.deriv()
                diff_coefs = deriv.coefficients
                diff_coefs[-1] -= target_slope
                radici = np.roots(diff_coefs)
                radici_real = [r.real for r in radici if np.isreal(r) and xmin <= r.real <= xmax]
                if not radici_real:
                    results.append((ds['name'], np.nan, "nessuna soluzione"))
                else:
                    radici_real.sort(reverse=True)
                    results.append((ds['name'], radici_real[0], "ok"))
            except Exception as e:
                results.append((ds['name'], np.nan, str(e)))
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if file_path:
            try:
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    f.write("File,X_trovato,Note\n")
                    for name, x_val, note in results:
                        if np.isnan(x_val):
                            f.write(f"{name},,{note}\n")
                        else:
                            f.write(f"{name},{x_val:.6f},{note}\n")
                messagebox.showinfo("Salvato", f"Risultati salvati in {file_path}")
            except Exception as e:
                messagebox.showerror("Errore", f"Impossibile salvare: {str(e)}")
    
    def open_fit_window(self):
        """Apre una nuova finestra con i fit polinomiali di tutti i dataset."""
        if not self.datasets:
            messagebox.showwarning("Attenzione", "Carica prima almeno un dataset.")
            return
        try:
            xmin = float(self.xmin_entry.get())
            xmax = float(self.xmax_entry.get())
            degree = self.degree_var.get()
        except:
            messagebox.showerror("Errore", "Inserisci valori numerici per X min, X max e grado.")
            return
        FitWindow(self.root, self.datasets, xmin, xmax, degree)


class FitWindow:
    def __init__(self, parent, datasets, xmin_fit, xmax_fit, degree):
        self.parent = parent
        self.datasets = datasets
        self.xmin_fit = xmin_fit
        self.xmax_fit = xmax_fit
        self.degree = degree
        self.fit_polys = []   # lista di (name, poly)
        
        self.window = tk.Toplevel(parent)
        self.window.title(f"Fit polinomiale grado {degree} - Tutti i dataset")
        self.window.geometry("900x750")
        
        # Frame per pendenza desiderata con intervallo di ricerca
        slope_frame = ttk.LabelFrame(self.window, text="Trova X per pendenza desiderata (derivata prima)", padding="5")
        slope_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(slope_frame, text="Pendenza desiderata:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.slope_var = tk.StringVar()
        self.slope_entry = ttk.Entry(slope_frame, textvariable=self.slope_var, width=10)
        self.slope_entry.grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(slope_frame, text="Cerca X in intervallo:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.search_xmin = ttk.Entry(slope_frame, width=10)
        self.search_xmin.grid(row=1, column=1, padx=5, pady=2)
        ttk.Label(slope_frame, text="—").grid(row=1, column=2, sticky=tk.W, padx=2)
        self.search_xmax = ttk.Entry(slope_frame, width=10)
        self.search_xmax.grid(row=1, column=3, padx=5, pady=2)
        
        self.slope_result = ttk.Label(slope_frame, text="(seleziona una curva)", relief=tk.SUNKEN, width=30)
        self.slope_result.grid(row=0, column=2, columnspan=2, padx=5, pady=2, sticky=tk.W)
        
        ttk.Label(slope_frame, text="Curva:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.curve_var = tk.StringVar()
        self.curve_menu = ttk.Combobox(slope_frame, textvariable=self.curve_var, state="readonly", width=40)
        self.curve_menu.grid(row=2, column=1, padx=5, pady=2)
        self.curve_menu.bind("<<ComboboxSelected>>", lambda e: self.update_slope_display())
        
        self.batch_slope_button = ttk.Button(slope_frame, text="Trova X per TUTTE le curve (salva CSV)", command=self.batch_find_slope)
        self.batch_slope_button.grid(row=2, column=2, padx=5, pady=2)
        
        self.slope_var.trace('w', lambda *args: self.update_slope_display())
        
        # Area grafico
        plot_frame = ttk.Frame(self.window)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.fig, self.ax = plt.subplots(figsize=(6,4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Coordinate mouse e zoom
        self.coord_label = ttk.Label(self.window, text="Mouse: x = ---, y = ---", relief=tk.SUNKEN, anchor=tk.W)
        self.coord_label.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=2)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        btn_frame = ttk.Frame(self.window)
        btn_frame.pack(pady=5)
        ttk.Button(btn_frame, text="Salva CSV (dati e fit)", command=self.save_curves_csv).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Chiudi", command=self.window.destroy).pack(side=tk.LEFT, padx=5)
        
        self.update_plot()
    
    def on_mouse_move(self, event):
        if event.inaxes == self.ax:
            self.coord_label.config(text=f"Mouse: x = {event.xdata:.6f}, y = {event.ydata:.6f}")
        else:
            self.coord_label.config(text="Mouse: x = ---, y = ---")
    
    def on_scroll(self, event):
        if event.inaxes != self.ax:
            return
        scale = 1.1
        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None:
            return
        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()
        if event.step > 0:
            xmin = xdata - (xdata - xmin) / scale
            xmax = xdata + (xmax - xdata) / scale
            ymin = ydata - (ydata - ymin) / scale
            ymax = ydata + (ymax - ydata) / scale
        else:
            xmin = xdata - (xdata - xmin) * scale
            xmax = xdata + (xmax - xdata) * scale
            ymin = ydata - (ydata - ymin) * scale
            ymax = ydata + (ymax - ydata) * scale
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.canvas.draw_idle()
    
    def compute_fits(self):
        """Calcola i polinomi di grado selezionato per tutti i dataset."""
        fits = []
        for ds in self.datasets:
            x_data = ds['x']
            y_data = ds['y']
            mask = (x_data >= self.xmin_fit) & (x_data <= self.xmax_fit)
            x_fit = x_data[mask]
            y_fit = y_data[mask]
            if len(x_fit) >= self.degree + 1:
                try:
                    coefs = np.polyfit(x_fit, y_fit, self.degree)
                    poly = np.poly1d(coefs)
                    fits.append((ds['name'], poly))
                except:
                    fits.append((ds['name'], None))
            else:
                fits.append((ds['name'], None))
        return fits
    
    def update_plot(self):
        self.ax.clear()
        self.fit_polys = self.compute_fits()
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.fit_polys)))
        curve_names = []
        for i, (name, poly) in enumerate(self.fit_polys):
            color = colors[i % len(colors)]
            # Mostra i dati originali
            ds = self.datasets[i]
            self.ax.scatter(ds['x'], ds['y'], s=5, color=color, alpha=0.5)
            if poly is not None:
                x_curve = np.linspace(self.xmin_fit, self.xmax_fit, 200)
                y_curve = poly(x_curve)
                self.ax.plot(x_curve, y_curve, '-', color=color, linewidth=2, label=name)
                curve_names.append(name)
            else:
                self.ax.plot([], [], color=color, label=name + " (fit fallito)")
                curve_names.append(name)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_title(f"Fit polinomiale grado {self.degree} per tutti i dataset")
        self.ax.legend(loc='best', fontsize='small')
        self.ax.grid(True)
        self.ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        self.ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        self.canvas.draw()
        self.curve_menu['values'] = curve_names
        if curve_names:
            self.curve_var.set(curve_names[0])
        self.update_slope_display()
    
    def get_search_interval(self):
        """Restituisce (xmin, xmax) per la ricerca, usando i campi utente o default."""
        try:
            xmin = float(self.search_xmin.get())
            xmax = float(self.search_xmax.get())
            if xmin < xmax:
                return xmin, xmax
        except:
            pass
        # Default: intervallo di fit
        return self.xmin_fit, self.xmax_fit
    
    def update_slope_display(self):
        """Aggiorna il risultato per la curva selezionata, pendenza e intervallo di ricerca."""
        if not self.fit_polys:
            self.slope_result.config(text="Nessuna curva")
            return
        try:
            target_slope = float(self.slope_var.get())
        except:
            self.slope_result.config(text="Inserisci pendenza")
            return
        selected = self.curve_var.get()
        poly = None
        for name, p in self.fit_polys:
            if name == selected:
                poly = p
                break
        if poly is None:
            self.slope_result.config(text="Fit non disponibile")
            return
        xmin_search, xmax_search = self.get_search_interval()
        deriv = poly.deriv()
        diff_coefs = deriv.coefficients
        diff_coefs[-1] -= target_slope
        radici = np.roots(diff_coefs)
        radici_real = [r.real for r in radici if np.isreal(r) and xmin_search <= r.real <= xmax_search]
        if not radici_real:
            self.slope_result.config(text=f"Nessuna soluzione in [{xmin_search:.2f}, {xmax_search:.2f}]")
        else:
            radici_real.sort(reverse=True)
            self.slope_result.config(text=f"X = {radici_real[0]:.6f}")
    
    def batch_find_slope(self):
        """Per tutte le curve, calcola X per la pendenza target nell'intervallo specificato, salva CSV."""
        if not self.fit_polys:
            messagebox.showwarning("Attenzione", "Nessun fit disponibile.")
            return
        try:
            target_slope = float(self.slope_var.get())
        except:
            messagebox.showerror("Errore", "Inserisci un valore numerico per la pendenza.")
            return
        xmin_search, xmax_search = self.get_search_interval()
        results = []
        for name, poly in self.fit_polys:
            if poly is None:
                results.append((name, np.nan, "fit fallito"))
                continue
            deriv = poly.deriv()
            diff_coefs = deriv.coefficients
            diff_coefs[-1] -= target_slope
            radici = np.roots(diff_coefs)
            radici_real = [r.real for r in radici if np.isreal(r) and xmin_search <= r.real <= xmax_search]
            if not radici_real:
                results.append((name, np.nan, f"nessuna soluzione in [{xmin_search:.2f},{xmax_search:.2f}]"))
            else:
                radici_real.sort(reverse=True)
                results.append((name, radici_real[0], "ok"))
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if file_path:
            try:
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    f.write("File,X_trovato,Note\n")
                    for name, x_val, note in results:
                        if np.isnan(x_val):
                            f.write(f"{name},,{note}\n")
                        else:
                            f.write(f"{name},{x_val:.6f},{note}\n")
                messagebox.showinfo("Salvato", f"Risultati salvati in {file_path}")
            except Exception as e:
                messagebox.showerror("Errore", f"Impossibile salvare: {str(e)}")
    
    def save_curves_csv(self):
        """Salva i dati originali e i fit polinomiali in un unico CSV."""
        if not self.datasets:
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not file_path:
            return
        try:
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                f.write("File, Tipo, X, Y\n")
                for i, ds in enumerate(self.datasets):
                    name = ds['name']
                    # Dati originali
                    for xi, yi in zip(ds['x'], ds['y']):
                        f.write(f"{name},originale,{xi},{yi}\n")
                    # Fit polinomiale (se disponibile)
                    if i < len(self.fit_polys) and self.fit_polys[i][1] is not None:
                        poly = self.fit_polys[i][1]
                        x_curve = np.linspace(self.xmin_fit, self.xmax_fit, 200)
                        y_curve = poly(x_curve)
                        for xi, yi in zip(x_curve, y_curve):
                            f.write(f"{name},fit,{xi},{yi}\n")
            messagebox.showinfo("Salvato", f"Dati salvati in {file_path}")
        except Exception as e:
            messagebox.showerror("Errore", f"Impossibile salvare: {str(e)}")

if __name__ == "__main__":
    app = MultiAnalysisApp() 
