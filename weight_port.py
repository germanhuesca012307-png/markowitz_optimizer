import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


class OptimizadorMarkowitz2D:
    def __init__(self, root):
        self.root = root
        self.root.title("Modelo de Portafolio 2D: E[Rp] = w^T μ  vs  σp = √(w^T Σ w)")
        self.root.geometry("1120x740")
        self.root.configure(bg="#f4f6f9")

        self.entries_tickers = []
        self.entries_pesos_manuales = []

        self._construir_interfaz()

    def _construir_interfaz(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#f4f6f9")
        style.configure("TLabelframe", background="#f4f6f9", font=("Segoe UI", 10, "bold"))
        style.configure("TLabelframe.Label", background="#f4f6f9", foreground="#1a252f")
        style.configure("TLabel", background="#f4f6f9", font=("Segoe UI", 9))
        style.configure("TButton", font=("Segoe UI", 9, "bold"), padding=6)
        style.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=[14, 6])

        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        frame_izq = ttk.Frame(main_pane, width=350)
        frame_der = ttk.Frame(main_pane)
        main_pane.add(frame_izq, weight=1)
        main_pane.add(frame_der, weight=3)

        # ==================== PANEL IZQUIERDO ====================
        # 1. Holdings
        lf_config = ttk.LabelFrame(frame_izq, text="1. Número de Holdings (Acciones)", padding=10)
        lf_config.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(lf_config, text="Cantidad de acciones:").grid(row=0, column=0, sticky="w", pady=4)
        self.var_num_holdings = tk.IntVar(value=2)
        spin_holdings = ttk.Spinbox(lf_config, from_=2, to=15, textvariable=self.var_num_holdings, width=7)
        spin_holdings.grid(row=0, column=1, sticky="w", padx=6, pady=4)

        btn_generar = ttk.Button(lf_config, text="Actualizar", command=self.generar_campos_tickers)
        btn_generar.grid(row=0, column=2, padx=4, pady=4)

        # 2. Fechas (1 Año Exacto)
        lf_fechas = ttk.LabelFrame(frame_izq, text="2. Periodo (1 Año para todas)", padding=10)
        lf_fechas.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(lf_fechas, text="Fecha Inicio (YYYY-MM-DD):").grid(row=0, column=0, sticky="w", pady=4)
        self.var_fecha_ini = tk.StringVar(value="2025-08-25")
        entry_ini = ttk.Entry(lf_fechas, textvariable=self.var_fecha_ini, width=13)
        entry_ini.grid(row=0, column=1, padx=5, pady=4)
        entry_ini.bind("<FocusOut>", self.actualizar_fecha_fin)
        entry_ini.bind("<Return>", self.actualizar_fecha_fin)

        ttk.Label(lf_fechas, text="Fecha Fin (YYYY-MM-DD):").grid(row=1, column=0, sticky="w", pady=4)
        self.var_fecha_fin = tk.StringVar(value="2026-08-25")
        entry_fin = ttk.Entry(lf_fechas, textvariable=self.var_fecha_fin, width=13)
        entry_fin.grid(row=1, column=1, padx=5, pady=4)

        self.var_forzar_anio = tk.BooleanVar(value=True)
        chk_anio = ttk.Checkbutton(
            lf_fechas,
            text="Fijar exactamente a 1 año (365 días)",
            variable=self.var_forzar_anio,
            command=self.actualizar_fecha_fin
        )
        chk_anio.grid(row=2, column=0, columnspan=2, sticky="w", pady=4)

        # 3. Siglas (Tickers)
        lf_tickers = ttk.LabelFrame(frame_izq, text="3. Siglas (Tickers) y Pesos w", padding=10)
        lf_tickers.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.frame_lista_tickers = ttk.Frame(lf_tickers)
        self.frame_lista_tickers.pack(fill=tk.BOTH, expand=True)

        self.generar_campos_tickers(defaults=["WMT", "TSLA"])

        # 4. Botones de Cálculo
        lf_acciones = ttk.LabelFrame(frame_izq, text="4. Ejecutar Optimización", padding=10)
        lf_acciones.pack(fill=tk.X)

        btn_yahoo = ttk.Button(
            lf_acciones,
            text="Descargar de Yahoo Finance y Calcular",
            command=lambda: self.ejecutar_modelo(fuente="yahoo")
        )
        btn_yahoo.pack(fill=tk.X, pady=3)

        btn_excel = ttk.Button(
            lf_acciones,
            text="Cargar desde tu Excel y Calcular",
            command=lambda: self.ejecutar_modelo(fuente="excel")
        )
        btn_excel.pack(fill=tk.X, pady=3)

        # ==================== PANEL DERECHO ====================
        self.notebook = ttk.Notebook(frame_der)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Pestaña 1: Escenarios Ideales (Texto)
        tab_texto = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(tab_texto, text="  1. Escenarios Ideales (Resultados)  ")

        self.txt_resultados = tk.Text(
            tab_texto, font=("Consolas", 10), bg="#ffffff", fg="#1e272e", wrap=tk.NONE, padx=12, pady=12
        )
        scroll_y = ttk.Scrollbar(tab_texto, orient=tk.VERTICAL, command=self.txt_resultados.yview)
        scroll_x = ttk.Scrollbar(tab_texto, orient=tk.HORIZONTAL, command=self.txt_resultados.xview)
        self.txt_resultados.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.txt_resultados.pack(fill=tk.BOTH, expand=True)

        # Pestaña 2: Gráfica 2D (Scatter Plot with Smooth Lines)
        tab_grafica = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(tab_grafica, text="  2. Gráfica 2D (Scatter Plot con Línea Suavizada)  ")

        self.fig, self.ax = plt.subplots(figsize=(7.5, 5.2), dpi=100)
        self.fig.patch.set_facecolor("#ffffff")
        self.canvas = FigureCanvasTkAgg(self.fig, master=tab_grafica)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def actualizar_fecha_fin(self, event=None):
        if self.var_forzar_anio.get():
            try:
                dt_ini = datetime.strptime(self.var_fecha_ini.get().strip(), "%Y-%m-%d")
                dt_fin = dt_ini + timedelta(days=365)
                self.var_fecha_fin.set(dt_fin.strftime("%Y-%m-%d"))
            except ValueError:
                pass

    def generar_campos_tickers(self, defaults=None):
        for widget in self.frame_lista_tickers.winfo_children():
            widget.destroy()
        self.entries_tickers.clear()
        self.entries_pesos_manuales.clear()

        n = self.var_num_holdings.get()
        peso_igual = round(1.0 / n, 4)

        ttk.Label(self.frame_lista_tickers, text="#", font=("Segoe UI", 9, "bold")).grid(row=0, column=0, padx=3, pady=2)
        ttk.Label(self.frame_lista_tickers, text="Siglas (Ticker)", font=("Segoe UI", 9, "bold")).grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(self.frame_lista_tickers, text="Peso w_i", font=("Segoe UI", 9, "bold")).grid(row=0, column=2, padx=5, pady=2)

        lista_def = defaults if defaults else ["WMT", "TSLA", "MSFT", "AMZN", "AAPL"]

        for i in range(n):
            ttk.Label(self.frame_lista_tickers, text=f"Acción {i+1}:").grid(row=i+1, column=0, sticky="e", pady=3)

            val_t = lista_def[i] if i < len(lista_def) else ""
            ent_t = ttk.Entry(self.frame_lista_tickers, width=14)
            ent_t.insert(0, val_t)
            ent_t.grid(row=i+1, column=1, padx=5, pady=3)
            self.entries_tickers.append(ent_t)

            ent_w = ttk.Entry(self.frame_lista_tickers, width=10)
            ent_w.insert(0, str(peso_igual))
            ent_w.grid(row=i+1, column=2, padx=5, pady=3)
            self.entries_pesos_manuales.append(ent_w)

    def obtener_datos(self, fuente, tickers):
        self.actualizar_fecha_fin()
        fecha_ini = self.var_fecha_ini.get().strip()
        fecha_fin = self.var_fecha_fin.get().strip()

        if fuente == "yahoo":
            if not YFINANCE_AVAILABLE:
                messagebox.showerror("Error", "Instala yfinance: pip install yfinance")
                return None
            datos = yf.download(tickers, start=fecha_ini, end=fecha_fin, auto_adjust=True, progress=False)
            if datos.empty:
                raise ValueError("No se encontraron datos para los tickers y fechas seleccionados.")
            precios = datos["Close"] if "Close" in datos else datos
            if isinstance(precios, pd.Series):
                precios = precios.to_frame(name=tickers[0])
            precios = precios[tickers].dropna().sort_index(ascending=True)
            return precios.pct_change().dropna()

        elif fuente == "excel":
            filepath = filedialog.askopenfilename(
                title="Selecciona tu archivo de Excel",
                filetypes=[("Archivos de Excel", "*.xlsx *.xls")]
            )
            if not filepath:
                return None
            df = pd.read_excel(filepath, sheet_name=0)
            mapa = {
                "MSFT": "Microsoft", "AMZN": "Amazon", "WMT": "Walmart",
                "AAPL": "Apple", "TSLA": "Tesla", "NASDQ": "NASDQ"
            }
            cols = []
            for t in tickers:
                if t in df.columns:
                    cols.append(t)
                elif t.upper() in mapa and mapa[t.upper()] in df.columns:
                    cols.append(mapa[t.upper()])
                else:
                    raise ValueError(f"No se encontró la columna '{t}' en el Excel.")

            precios = df[cols].dropna()
            precios.columns = tickers
            if "Fecha" in df.columns:
                fechas = pd.to_datetime(df["Fecha"], format="%d.%m.%Y", errors="coerce")
                if fechas.iloc[0] > fechas.iloc[-1]:
                    precios = precios.iloc[::-1].reset_index(drop=True)
            return precios.pct_change().dropna()

    # =========================================================================
    # FÓRMULAS MATRICIALES EXACTAS: E[R_p] = w^T μ   y   σ_p = sqrt(w^T Σ w)
    # =========================================================================
    @staticmethod
    def rendimiento_esperado(w, mu):
        """Calcula E[R_p] = w^T * mu"""
        w = np.asarray(w, dtype=float)
        return float(np.dot(w.T, mu))

    @staticmethod
    def varianza_portafolio(w, sigma_mat):
        """Calcula Var(R_p) = w^T * Sigma * w"""
        w = np.asarray(w, dtype=float)
        return float(np.dot(w.T, np.dot(sigma_mat, w)))

    @staticmethod
    def volatilidad_portafolio(w, sigma_mat):
        """Calcula sigma_p = sqrt(w^T * Sigma * w)"""
        var_p = OptimizadorMarkowitz2D.varianza_portafolio(w, sigma_mat)
        return float(np.sqrt(max(var_p, 0.0)))

    def ejecutar_modelo(self, fuente="yahoo"):
        try:
            tickers = [e.get().strip().upper() for e in self.entries_tickers if e.get().strip()]
            n = len(tickers)
            if n < 2:
                messagebox.showwarning("Atención", "Ingresa al menos 2 acciones.")
                return

            w_manual = np.array([float(e.get()) for e in self.entries_pesos_manuales[:n]], dtype=float)
            if not np.isclose(np.sum(w_manual), 1.0, atol=1e-2):
                w_manual = w_manual / np.sum(w_manual)

            rendimientos = self.obtener_datos(fuente, tickers)
            if rendimientos is None or rendimientos.empty:
                return

            # Vector μ (en tu Excel celda M2 es la suma anual =SUM(F2:F274), y también guardamos la media diaria)
            mu_diario = rendimientos.mean().values
            mu_anual = rendimientos.sum().values  # Vector μ anualizado como en el Excel

            # Matriz de Covarianza Poblacional Σ (ddof=0 como COVARIANCE.P y VAR.P en Excel)
            Sigma = rendimientos.cov(ddof=0).values
            corr_mat = rendimientos.corr().values

            var_indiv = np.diag(Sigma)
            sdv_indiv = np.sqrt(var_indiv)

            restriccion_suma = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
            limites = tuple((0.0, 1.0) for _ in range(n))
            w0 = np.ones(n) / n

            # 1. ESCENARIO IDEAL 1: MÍNIMA VOLATILIDAD -> min σ_p = sqrt(w^T Σ w)
            res_min_vol = minimize(
                lambda w: self.volatilidad_portafolio(w, Sigma),
                w0, method='SLSQP', bounds=limites, constraints=restriccion_suma,
                options={'ftol': 1e-12, 'maxiter': 1000}
            )
            w_min_vol = res_min_vol.x

            # 2. ESCENARIO IDEAL 2: MÁXIMO RENDIMIENTO -> max E[R_p] = w^T μ
            res_max_ret = minimize(
                lambda w: -self.rendimiento_esperado(w, mu_anual),
                w0, method='SLSQP', bounds=limites, constraints=restriccion_suma,
                options={'ftol': 1e-12, 'maxiter': 1000}
            )
            w_max_ret = res_max_ret.x

            # 3. ESCENARIOS EN 2D (11 puntos de 0.0 a 1.0 para la tabla y el Scatter Plot)
            escenarios_2d = self._calcular_escenarios_2d(n, mu_anual, Sigma, limites)

            # Mostrar reporte numérico y actualizar gráfica 2D
            self._mostrar_resultados(
                tickers, len(rendimientos), mu_diario, mu_anual, sdv_indiv, var_indiv,
                Sigma, corr_mat, w_manual, w_min_vol, w_max_ret, escenarios_2d
            )
            self._graficar_scatter_smooth_2d(
                tickers, mu_anual, Sigma, w_min_vol, w_max_ret, escenarios_2d
            )

        except Exception as e:
            messagebox.showerror("Error en el cálculo", str(e))

    def _calcular_escenarios_2d(self, n, mu, Sigma, limites):
        """
        Genera los 11 escenarios (pasos de 0.0 a 1.0) evaluando exactamente:
        E[R_p] = w^T μ   y   σ_p = sqrt(w^T Σ w)
        """
        escenarios = []
        pasos = np.linspace(0.0, 1.0, 11)

        if n == 2:
            # Exactamente la tabla de tu Excel (filas 13 a 23): w_1 va de 0.0 a 1.0 y w_2 = 1 - w_1
            for alpha in pasos:
                w = np.array([alpha, 1.0 - alpha])
                erp = self.rendimiento_esperado(w, mu)
                sig = self.volatilidad_portafolio(w, Sigma)
                var = self.varianza_portafolio(w, Sigma)
                escenarios.append((alpha, w, erp, sig, var))
        else:
            # Para N > 2 acciones: proyecta en 2D resolviendo la frontera de mínima varianza
            # para cada nivel de rendimiento objetivo E[R_p] entre min(μ) y max(μ)
            r_min_posible = float(np.min(mu))
            r_max_posible = float(np.max(mu))
            objetivos_r = np.linspace(r_min_posible, r_max_posible, 11)
            w_prev = np.ones(n) / n

            for idx, r_obj in enumerate(objetivos_r):
                cons = (
                    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                    {'type': 'eq', 'fun': lambda w, target=r_obj: self.rendimiento_esperado(w, mu) - target}
                )
                res = minimize(
                    lambda w: self.varianza_portafolio(w, Sigma),
                    w_prev, method='SLSQP', bounds=limites, constraints=cons,
                    options={'ftol': 1e-12, 'maxiter': 1000}
                )
                w_opt = res.x if res.success else w_prev
                w_prev = w_opt
                erp = self.rendimiento_esperado(w_opt, mu)
                sig = self.volatilidad_portafolio(w_opt, Sigma)
                var = self.varianza_portafolio(w_opt, Sigma)
                escenarios.append((pasos[idx], w_opt, erp, sig, var))

        return escenarios

    def _mostrar_resultados(self, tickers, n_dias, mu_d, mu_a, sdv_i, var_i,
                            Sigma, corr_mat, w_man, w_min, w_max, escenarios_2d):
        self.txt_resultados.delete("1.0", tk.END)
        sep = "=" * 86 + "\n"
        sub = "-" * 86 + "\n"

        txt = sep
        txt += f" MODELO MATRICIAL DE PORTAFOLIO ({len(tickers)} HOLDINGS | {n_dias} RENDIMIENTOS - 1 AÑO)\n"
        txt += " Fórmulas:  E[R_p] = w^T μ     |     σ_p = √(w^T Σ w)     |     Var(R_p) = w^T Σ w\n"
        txt += sep

        # Evaluar los dos escenarios ideales
        erp_min = self.rendimiento_esperado(w_min, mu_a)
        erp_min_d = self.rendimiento_esperado(w_min, mu_d)
        sig_min = self.volatilidad_portafolio(w_min, Sigma)
        var_min = self.varianza_portafolio(w_min, Sigma)

        erp_max = self.rendimiento_esperado(w_max, mu_a)
        erp_max_d = self.rendimiento_esperado(w_max, mu_d)
        sig_max = self.volatilidad_portafolio(w_max, Sigma)
        var_max = self.varianza_portafolio(w_max, Sigma)

        txt += "\n★ ESCENARIO IDEAL 1: MÍNIMA VOLATILIDAD  [ Minimiza σ_p = √(w^T Σ w) ]\n"
        txt += sub
        for i, t in enumerate(tickers):
            txt += f"   • Peso ideal w_{i+1} ({t:<6})       : {w_min[i]:.6f}  ({w_min[i]*100:6.2f}%)\n"
        txt += "   --------------------------------------------------------\n"
        txt += f"   • E[R_p] = w^T μ (Anual Acum.)  : {erp_min:.6f}  ({erp_min*100:.2f}%)\n"
        txt += f"   • E[R_p] = w^T μ (Media Diaria) : {erp_min_d:.6f}\n"
        txt += f"   • σ_p = √(w^T Σ w) (Volatilidad): {sig_min:.6f}  ({sig_min*100:.2f}%)\n"
        txt += f"   • σ_p² = w^T Σ w   (Varianza)   : {var_min:.6f}\n"

        txt += "\n★ ESCENARIO IDEAL 2: MÁXIMO RENDIMIENTO  [ Maximiza E[R_p] = w^T μ ]\n"
        txt += sub
        for i, t in enumerate(tickers):
            txt += f"   • Peso ideal w_{i+1} ({t:<6})       : {w_max[i]:.6f}  ({w_max[i]*100:6.2f}%)\n"
        txt += "   --------------------------------------------------------\n"
        txt += f"   • E[R_p] = w^T μ (Anual Acum.)  : {erp_max:.6f}  ({erp_max*100:.2f}%)\n"
        txt += f"   • E[R_p] = w^T μ (Media Diaria) : {erp_max_d:.6f}\n"
        txt += f"   • σ_p = √(w^T Σ w) (Volatilidad): {sig_max:.6f}  ({sig_max*100:.2f}%)\n"
        txt += f"   • σ_p² = w^T Σ w   (Varianza)   : {var_max:.6f}\n\n"

        # Tabla de escenarios (0.0 a 1.0)
        txt += sep
        txt += " TABLA DE ESCENARIOS 2D (COMPORTAMIENTO DE E[R_p] CONTRA σ_p Y VARIANZA)\n"
        txt += sep
        col_paso = f"Peso {tickers[0]}" if len(tickers) == 2 else "Escenario"
        txt += f"{col_paso:<12} | {'E[R_p] = w^T μ':>16} | {'σ_p = √(w^T Σ w)':>18} | {'Var = w^T Σ w':>16} | Pesos w\n"
        txt += sub
        for paso, w_vec, erp, sig, var in escenarios_2d:
            w_str = ", ".join([f"{tickers[i]}:{w_vec[i]:.2f}" for i in range(len(tickers))])
            txt += f"{paso:<12.1f} | {erp:>16.6f} | {sig:>18.6f} | {var:>16.6f} | [{w_str}]\n"

        # Estadísticos individuales y Matriz Σ
        txt += "\n" + sep
        txt += " VECTOR DE MEDIAS (μ) Y MATRIZ DE COVARIANZA POBLACIONAL (Σ)\n"
        txt += sep
        txt += f"{'Ticker':<10} | {'μ Diaria':>14} | {'μ Anual (SUM)':>15} | {'σ_i (sdv)':>14} | {'σ_i² (var.p)':>14}\n"
        txt += sub
        for i, t in enumerate(tickers):
            txt += f"{t:<10} | {mu_d[i]:>14.6f} | {mu_a[i]:>15.6f} | {sdv_i[i]:>14.6f} | {var_i[i]:>14.6f}\n"

        df_cov = pd.DataFrame(Sigma, index=tickers, columns=tickers)
        txt += f"\nMatriz de Covarianza Σ (COVARIANCE.P):\n{df_cov.to_string(float_format=lambda x: f'{x:.6f}')}\n"

        self.txt_resultados.insert(tk.END, txt)

    def _graficar_scatter_smooth_2d(self, tickers, mu_a, Sigma, w_min, w_max, escenarios_2d):
        """
        Grafica en 2D el Scatter Plot con Smooth Lines usando:
        Eje X: σ_p = sqrt(w^T Σ w)
        Eje Y: E[R_p] = w^T μ
        """
        self.ax.clear()

        # Coordenadas de los 11 escenarios (puntos del Scatter Plot)
        x_pts = np.array([item[3] for item in escenarios_2d])  # σ_p = sqrt(w^T Σ w)
        y_pts = np.array([item[2] for item in escenarios_2d])  # E[R_p] = w^T μ

        # Crear línea suavizada (Smooth Line de Excel) interpolando paramétricamente sobre los puntos
        t_param = np.linspace(0.0, 1.0, len(x_pts))
        t_suave = np.linspace(0.0, 1.0, 300)

        try:
            if len(x_pts) >= 4:
                spline_x = make_interp_spline(t_param, x_pts, k=3)
                spline_y = make_interp_spline(t_param, y_pts, k=3)
                x_smooth = spline_x(t_suave)
                y_smooth = spline_y(t_suave)
            else:
                x_smooth, y_smooth = x_pts, y_pts
        except Exception:
            x_smooth, y_smooth = x_pts, y_pts

        # 1. Trazar la línea suavizada y los puntos de los escenarios (estilo Excel Scatter with Smooth Lines)
        self.ax.plot(
            x_smooth, y_smooth,
            color="#2b579a", linewidth=2.4, zorder=2,
            label="Curva de Escenarios (Smooth Line)"
        )
        self.ax.scatter(
            x_pts, y_pts,
            color="#2b579a", s=45, zorder=3, edgecolors="white", linewidth=0.8,
            label="Escenarios (0.0 a 1.0)"
        )

        # 2. Marcar el punto exacto de Mínima Volatilidad y el de Máximo Rendimiento
        erp_min = self.rendimiento_esperado(w_min, mu_a)
        sig_min = self.volatilidad_portafolio(w_min, Sigma)

        erp_max = self.rendimiento_esperado(w_max, mu_a)
        sig_max = self.volatilidad_portafolio(w_max, Sigma)

        self.ax.scatter(
            sig_min, erp_min,
            color="#e74c3c", s=140, zorder=5, edgecolors="black", linewidth=1.2,
            label=f"Mínima Volatilidad: σp={sig_min:.4f}, E[Rp]={erp_min:.4f}"
        )
        self.ax.scatter(
            sig_max, erp_max,
            color="#27ae60", s=140, zorder=5, edgecolors="black", linewidth=1.2,
            label=f"Máximo Rendimiento: σp={sig_max:.4f}, E[Rp]={erp_max:.4f}"
        )

        # Márgenes amplios (25%) para que la curva quede centrada y ningún eje se amontone
        all_x = np.append(x_pts, [sig_min, sig_max])
        all_y = np.append(y_pts, [erp_min, erp_max])
        min_x, max_x = float(np.min(all_x)), float(np.max(all_x))
        min_y, max_y = float(np.min(all_y)), float(np.max(all_y))

        pad_x = (max_x - min_x) * 0.25 if max_x > min_x else 0.005
        pad_y = (max_y - min_y) * 0.25 if max_y > min_y else 0.05

        self.ax.set_xlim(min_x - pad_x, max_x + pad_x)
        self.ax.set_ylim(min_y - pad_y, max_y + pad_y)

        # Títulos con las fórmulas en LaTeX
        self.ax.set_title(
            r"Comportamiento del Portafolio en 2D:  $E[R_p] = \mathbf{w}^T \mathbf{\mu}$  vs  $\sigma_p = \sqrt{\mathbf{w}^T \Sigma \mathbf{w}}$",
            fontsize=11, fontweight="bold", pad=14
        )
        self.ax.set_xlabel(r"Riesgo / Volatilidad del Portafolio:  $\sigma_p = \sqrt{\mathbf{w}^T \Sigma \mathbf{w}}$", fontsize=10, labelpad=8)
        self.ax.set_ylabel(r"Rendimiento Esperado:  $E[R_p] = \mathbf{w}^T \mathbf{\mu}$", fontsize=10, labelpad=8)

        # Limitar el número de divisiones en los ejes para que se lean claramente
        self.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.4f}"))
        self.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.3f}"))
        self.ax.locator_params(axis='x', nbins=6)
        self.ax.locator_params(axis='y', nbins=6)

        self.ax.grid(True, linestyle="--", alpha=0.45)
        self.ax.legend(loc="best", frameon=True, facecolor="#f8f9fa", edgecolor="#dcdde1", fontsize=9)

        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = OptimizadorMarkowitz2D(root)
    root.mainloop()