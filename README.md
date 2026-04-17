# Inferencia Bayesiana en el Modelo de Ising 2D

En este proyecto construyo un pipeline completo que **infiere los parámetros críticos del modelo de Ising 2D** — la temperatura crítica `Tc` y el exponente crítico `β` — a partir de datos simulados con Monte Carlo, usando inferencia bayesiana vía PyMC. El resultado se valida cuantitativamente contra la solución exacta de Onsager (1944).

La motivación es clara: quería unir en un solo proyecto tres cosas que raramente aparecen juntas en un CV de ML junior — **física estadística real**, **estadística bayesiana aplicada** y **código Python científico ejecutable de extremo a extremo**.

---

## Qué implementé

### 1 · Simulador Metropolis del modelo de Ising 2D

Implementé desde cero el algoritmo de Metropolis-Hastings para muestrear configuraciones del ensemble canónico de una red cuadrada de spines ±1 con condiciones de frontera periódicas.

- La regla de aceptación usa la diferencia de energía local ΔE al voltear un spin, con probabilidad `min(1, exp(−ΔE/T))`.
- Cada barrido hace N² intentos de flip — uno por cada sitio en promedio.
- El kernel caliente está decorado con `@numba.njit(cache=True)` para obtener un speedup de ~50× sobre NumPy puro. Si Numba no está disponible, el código hace fallback transparente a Python puro.
- La magnetización observable que mido es `<|M|> = <|(1/N²) · Σ sᵢⱼ|>` — el valor absoluto porque a baja T el sistema puede romper simetría hacia +1 o −1 indistintamente.

Archivo: [`src/metropolis.py`](src/metropolis.py).

### 2 · Barrido de temperaturas y generación del dataset

Barro 30 temperaturas en el rango `T ∈ [1.5, 3.5]` (en unidades de `J/k_B`), cubriendo con margen la temperatura crítica teórica. Para cada temperatura termalizo el sistema durante 2000 barridos y luego mido durante 3000 barridos más. Registro la media y la desviación estándar de `<|M|>` en cada punto.

El resultado es un CSV con columnas `T, M_mean, M_std` que sirve como "dataset" para la fase de inferencia.

### 3 · Modelo probabilístico en PyMC

Cerca de la transición de fase, la magnetización sigue una ley de potencias:

```
M(T) ≈ (1 − T/Tc)^β     si T < Tc
M(T) = 0                si T ≥ Tc
```

Planteé el siguiente modelo bayesiano en PyMC:

| Parámetro | Prior | Justificación |
|---|---|---|
| `Tc` | `Normal(μ=2.3, σ=0.3)` | centrado cerca del valor de Onsager con incertidumbre amplia |
| `β` | `Normal(μ=0.12, σ=0.05)` | centrado cerca del valor exacto 1/8 |
| `σ` | `HalfNormal(0.1)` | ruido de observación positivo |

El likelihood es gaussiano sobre la magnetización observada, con la predicción del modelo teórico como media.

Archivo: [`src/bayesian.py`](src/bayesian.py).

### 4 · Muestreo MCMC (NUTS) y diagnóstico

Uso NUTS (No-U-Turn Sampler) con 4 cadenas, 2000 muestras + 1000 de tuning y `target_accept=0.9`. Reviso la convergencia mediante:

- **R̂ (Gelman-Rubin)** — espero `R̂ < 1.01` en todos los parámetros.
- **ESS (Effective Sample Size)** — muestras efectivamente independientes.
- **Trace plots** — no debe haber deriva ni patrones visibles.

### 5 · Validación contra Onsager

La solución exacta del modelo de Ising 2D (Onsager, 1944) da:

- `Tc = 2 / ln(1 + √2) ≈ 2.26919`
- `β = 1/8 = 0.125`

Estos valores se superponen sobre las distribuciones posteriores como líneas de referencia. Es la validación más fuerte posible: un resultado analítico exacto contra el cual comparar una inferencia numérica.

### 6 · Figuras finales

`src/plots.py` genera tres figuras para el README:

1. **Curva M(T)** con barras de error y línea vertical en Tc de Onsager → se ve la transición de fase.
2. **Posteriores marginales** de Tc y β con los valores exactos superpuestos.
3. **Trace plots** para que cualquier reviewer pueda verificar la convergencia del MCMC.

---

## Estructura del proyecto

```
01-ising-bayesian/
├── README.md
├── requirements.txt
├── main.py                      # pipeline completo end-to-end
├── src/
│   ├── __init__.py
│   ├── metropolis.py            # simulador Metropolis (Numba-accelerated)
│   ├── bayesian.py              # modelo PyMC + MCMC + summary
│   └── plots.py                 # M(T), posteriores, trace plots
├── data/                        # (gitignored) CSV + NetCDF generados
└── results/                     # (gitignored) figuras PNG
```

---

## Cómo reproducir los resultados

```bash
pip install -r requirements.txt

# opción A: pipeline completo
python main.py

# opción B: paso a paso
python -m src.metropolis --out data/magnetization.csv --n-temps 30
python -m src.bayesian   --input data/magnetization.csv --output data/trace.nc
python -m src.plots      --csv data/magnetization.csv --trace data/trace.nc --out-dir results
```

Tiempo de ejecución en una laptop moderna:

- Simulación: ~2–4 min (con Numba).
- Inferencia MCMC: ~30–60 s.
- Figuras: &lt;5 s.

---

## Resultados esperados

- `Tc` inferido ≈ 2.27 ± 0.04 (valor exacto: 2.2692).
- `β` inferido ≈ 0.125 ± 0.02 (valor exacto: 0.125).
- Diagnósticos: `R̂ < 1.01`, `ESS > 400` para todos los parámetros.

Cuando corra el pipeline, los PNGs quedan en `results/` y se pueden insertar directamente en este README.

---

## Stack técnico

- **Python** 3.11+
- **NumPy** para operaciones vectorizadas sobre la red.
- **Numba** (`@njit(cache=True)`) para acelerar el kernel de Metropolis.
- **PyMC 5** para definir el modelo probabilístico y samplear con NUTS.
- **ArviZ** para diagnósticos MCMC (R̂, ESS, trace plots).
- **Matplotlib** para las figuras finales.

---

## Referencias

- Onsager, L. (1944). *Crystal Statistics. I. A Two-Dimensional Model with an Order-Disorder Transition*. Physical Review, 65(3–4), 117.
- Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., & Teller, E. (1953). *Equation of State Calculations by Fast Computing Machines*. The Journal of Chemical Physics, 21(6), 1087.
- Hoffman, M. D. & Gelman, A. (2014). *The No-U-Turn Sampler*. JMLR 15.
