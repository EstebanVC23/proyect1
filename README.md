# Mira (Señales DSP) — análisis de audio mediante ADC y DAC

Aplicación de escritorio en Python para cargar, procesar y comparar fragmentos musicales WAV. Incluye una interfaz gráfica interactiva, representación temporal en alta resolución, remuestreo antialiasing, análisis espectral mediante FFT, cuantización ADC uniforme con escala física en voltios, reconstrucción DAC (retención de orden cero e interpolación), biblioteca matemática desacoplada, histograma acústico comparativo de 7 bandas y exportación de resultados.

## 1. Qué incluye

- **Biblioteca de cinco pistas precargadas**: audios por defecto para clásica/barroca (*Capricho No. 24*), electrónica (*Fade* - Alan Walker), salsa (*Gotas de Lluvia*), música de cantina (*Repítela*) y reguetón (*Diva Virtual*), cargados automáticamente desde `data/audio/` al iniciar la app.
- **Carga y reemplazo de archivos WAV**: permite seleccionar audios propios para cualquiera de los cinco espacios.
- **Ficha musical editable**: metadatos de título, compositor/intérprete, año y fuente/referencia.
- **Caracterización física del archivo**: frecuencia de muestreo original ($f_s$), número de canales, duración total y profundidad de bits.
- **Conversión a mono**: estandarización de señales multicanal mediante promedio coherente para análisis comparativo.
- **Intervalo temporal configurable**: rango de análisis acotado (por defecto de 0 a 6 segundos, ampliable hasta 10 s) con visualización de la ventana sombreada sobre la canción completa.
- **Frecuencia de procesamiento y doble límite de Nyquist**: remuestreo polifásico antialiasing (`resample_poly`), con graficación simultánea de la frecuencia de Nyquist de procesamiento y la frecuencia de Nyquist del archivo original en el espectro FFT.
- **Cuantización ADC en escala física de tensión**: rango simétrico configurable (por defecto $\pm 5.0\text{ V}$), con actualización instantánea al cambiar la resolución en bits ($4$, $8$, $12$ o $16$ bits), cálculo del paso cuántico $\Delta V$ en milivoltios y relación señal a ruido de cuantización (SNR).
- **Reconstrucción DAC (ZOH)**: simulación por retención de orden cero y cálculo del error cuadrático medio (RMSE) y error absoluto medio (MAE).
- **Histograma comparativo de 7 bandas acústicas**: distribución porcentual de energía espectral (subgraves, graves, medios-bajos, medios, medios-altos, agudos y muy agudos) para contrastar todos los géneros.
- **Evaluación neuro-acústica especializada para TDAH**: análisis fundamentado en el *Modelo de Activación Cerebral Moderada (MBA)* y la *Teoría de Estimulación Óptima*, con ranking justificado por propiedades de la señal.
- **Exportación de resultados**: generación de reportes tabulares en formato CSV e informes técnicos detallados en TXT.
- **Biblioteca matemática modular (`dsp.calculations`)**: funciones numéricas desacopladas y reutilizables para señales continuas, muestreo, aliasing, FFT y decimación.
- **Batería de 26 pruebas unitarias automáticas**: validación de fórmulas, unidades y casos borde con el módulo nativo `unittest`.

## 2. Requisitos

- **Python 3.10 o superior** (compatible con NumPy 1.x y NumPy 2.x).
- **Sistema operativo**: Windows, macOS o Linux con soporte gráfico para Tkinter.
- **Dependencias principales**:
  - `numpy>=1.24`
  - `scipy>=1.10`
  - `matplotlib>=3.7`
  - `soundfile>=0.12.0`
- **Dependencia opcional**:
  - `sounddevice>=0.4` (requerida únicamente si se desea reproducir el audio por los altavoces desde la interfaz).

## 3. Instalación y ejecución

### Configuración en Windows (PowerShell)

Abre una terminal PowerShell en la raíz del proyecto y ejecuta:

```powershell
# 1. Crear el entorno virtual
python -m venv .venv

# 2. Activar el entorno virtual
.\.venv\Scripts\Activate.ps1

# 3. Actualizar pip e instalar dependencias
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 4. Iniciar la aplicación
python main.py
```

Si la política de ejecución de scripts de Windows restringe la activación de entornos virtuales, puedes ejecutar directamente:

```powershell
.\.venv\Scripts\python.exe main.py
```

## 4. Uso básico de la aplicación

1. **Biblioteca y pistas**: Al iniciar, las cinco canciones de prueba se cargan automáticamente. Puedes hacer clic en cualquier fila de la tabla para seleccionarla y editar su ficha musical.
2. **Reemplazo de audio**: Pulsa **Cargar / reemplazar WAV** para cargar un archivo local diferente en la pista activa.
3. **Parámetros de procesamiento**:
   - Cambia la **Resolución ADC (bits)** mediante el menú desplegable ($4$, $8$, $12$ o $16$ bits); la aplicación recalcula y actualiza las gráficas al instante.
   - Ajusta la **Frecuencia de procesamiento (Hz)** o el rango de tensión ($\pm 5\text{ V}$) y presiona Enter o pulsa **Aplicar y recalcular**.
4. **Intervalo temporal**: Escribe el segundo de **Inicio** y **Fin** (por ejemplo, `0` y `6`) y presiona Enter para acotar el análisis.
5. **Navegación por pestañas**:
   - **Señal y ADC-DAC**: Cuatro cuadrantes que muestran la señal original, las muestras remuestreadas, la escalera de cuantización en voltios con sus niveles y la reconstrucción DAC por ZOH.
   - **Canción completa**: Forma de onda íntegra con zoom inteligente por cubetas min-max y sombreado del intervalo seleccionado.
   - **Espectro FFT**: Espectro unilateral de amplitud con detección del pico dominante y líneas guía para Nyquist.
   - **Comparación de géneros**: Superposición espectral normalizada en dB respecto al pico y tabla comparativa de RMSE.
   - **Conclusiones**: Histograma de energía por bandas acústicas y reporte neuro-acústico sobre concentración y TDAH con barra de desplazamiento.
6. **Exportación**: Pulsa **Exportar CSV** o **Guardar informe TXT** para almacenar las métricas generadas.

## 5. Fundamentos y cálculos matemáticos

El núcleo científico y computacional reside en `dsp/calculations.py` y `dsp/processing.py`:

### 5.1 Generación de señales continuas simuladas
- Suma de componentes sinusoidales:
  \[
  x_c(t) = \sum_{i} A_i \sin(2\pi f_i t)
  \]
- Pulso rectangular simétrico de ancho $T$:
  \[
  x(t) = \begin{cases} A, & |t| \le \frac{T}{2} \\ 0, & |t| > \frac{T}{2} \end{cases}
  \]

### 5.2 Muestreo y criterio de Nyquist-Shannon
Para evitar aliasing, la frecuencia de muestreo $f_s$ debe satisfacer la tasa mínima:
\[
f_{s,\text{nyquist}} = 2 f_{\max}
\]
Los instantes discretos se evalúan como $t_n = n / f_s$ con período $T_s = 1 / f_s$. El remuestreo polifásico aplica un filtro antialiasing FIR simétrico de fase lineal mediante `scipy.signal.resample_poly`.

### 5.3 Frecuencia aparente por submuestreo (Aliasing)
Cuando $f_{\text{signal}} > f_s / 2$, la componente espectral se pliega en la banda base $[0, f_s / 2]$. La formulación modular general empleada es:
\[
f_{\text{alias}} = \left| \left(|f_{\text{signal}}| + \frac{f_s}{2}\right) \bmod f_s - \frac{f_s}{2} \right|
\]
Esta ecuación es exacta para armónicos de cualquier orden y maneja frecuencias positivas y negativas.

### 5.4 Transformada de Fourier por integración directa
Aproximación numérica de la integral continua de Fourier:
\[
X(f) = \int_{-\infty}^{\infty} x(t) e^{-j 2\pi ft} \, dt
\]
Se resuelve mediante la regla trapezoidal (`np.trapezoid` / `np.trapz`). Para un pulso rectangular de ancho $T$, se compara con la solución analítica exacta:
\[
X_{\text{teorica}}(f) = T \cdot \operatorname{sinc}(f T) = T \frac{\sin(\pi f T)}{\pi f T}
\]

### 5.5 Transformada Rápida de Fourier (FFT) y espectro unilateral
Calcula la magnitud unilateral normalizada en escala física:
\[
|X[k]| = \frac{2}{N} \left| \sum_{n=0}^{N-1} x[n] e^{-j 2\pi kn / N} \right|
\]
Las componentes de continua ($k=0$) y de Nyquist ($k=N/2$ para $N$ par) se multiplican por $0.5$ (factor neto $1/N$) para preservar la conservación estricta de energía.

### 5.6 Cuantización uniforme mid-rise y conversión a voltios
Para una resolución de $B$ bits y rango simétrico $[V_{\min}, V_{\max}]$:
\[
N_{\text{niveles}} = 2^B, \qquad \Delta V = \frac{V_{\max} - V_{\min}}{2^B}
\]
El cuantizador mid-rise mapea la señal normalizada $x \in [-1, 1]$ a voltios y calcula los códigos enteros:
\[
V_{\text{in}} = V_{\min} + \left(\frac{x + 1}{2}\right)(V_{\max} - V_{\min})
\]
\[
\text{código} = \operatorname{clip}\left(\left\lfloor \frac{V_{\text{in}} - V_{\min}}{\Delta V} \right\rfloor, 0, 2^B - 1\right)
\]
\[
V_{\text{cuant}} = V_{\min} + (\text{código} + 0.5) \Delta V
\]
La relación señal a ruido teórica de cuantización se evalúa como:
\[
\text{SNR} = 20 \log_{10}\left(\frac{\text{RMS}_{\text{señal}}}{\text{RMS}_{\text{error}}}\right) \quad [\text{dB}]
\]
Cuando la señal está en silencio absoluto ($\text{RMS}_{\text{señal}} = 0$) con error no nulo, la métrica reporta correctamente $-\infty\text{ dB}$, y $+\infty\text{ dB}$ si el error es nulo.

### 5.7 Reconstrucción DAC y errores
- **Retención de orden cero (ZOH)**: asigna a cada instante continuo el valor de la muestra previa: $x_{\text{ZOH}}(t) = x_s[\lfloor t \cdot f_s \rfloor]$.
- **Métricas de error**:
  \[
  \text{RMSE} = \sqrt{\frac{1}{M}\sum_{m=1}^M (x[m] - x_{\text{ZOH}}[m])^2}, \qquad \text{MAE} = \frac{1}{M}\sum_{m=1}^M |x[m] - x_{\text{ZOH}}[m]|
  \]

### 5.8 Decimación de picos min-max para visualización
Para trazar formas de onda extensas sin aliasing visual ni degradación del rendimiento gráfico, `compute_minmax_downsample` divide el segmento visible en cubetas y conserva los índices de picos mínimos y máximos locales, garantizando que los transitorios no se pierdan al hacer zoom o pan.

## 6. Música, concentración y TDAH

La pestaña de conclusiones integra un análisis neuro-acústico fundamentado en el *Modelo de Activación Cerebral Moderada* (MBA; Sikström & Söderlund, 2007) y la *Teoría de la Estimulación Óptima* (Zentall, 1975). En personas con Trastorno por Déficit de Atención e Hiperactividad (TDAH), la corteza prefrontal presenta un tono dopaminérgico basal disminuido, requiriendo estimulación sensorial continua y estructurada para prevenir la dispersión cognitiva.

A partir del histograma de 7 bandas acústicas y el análisis de armónicos:

1. **🥇 1er Lugar: Música Clásica / Barroca (*Capricho No. 24* - N. Paganini)**:
   - **Cero interferencia fonológica**: La ausencia total de letra previene la sobrecarga del bucle fonológico de la memoria de trabajo (áreas de Broca y Wernicke).
   - **Distribución espectral balanceada**: Concentra su energía en frecuencias medias ($500\text{–}2000\text{ Hz}$) y medias-altas ($2\text{k}\text{–}4\text{k}\text{ Hz}$), sin saturación de subgraves ni percusiones bruscas.
   - **Estructura métrica matemática**: Favorece la sincronización de ondas cerebrales Alfa ($8\text{–}12\text{ Hz}$), asociadas con la vigilia tranquila (*relaxed alertness*) y el estudio conceptual profundo.

2. **🥈 2º Lugar: Electrónica Instrumental (*Fade* - Alan Walker)**:
   - **Ancla rítmica y dopaminérgica**: Base continua y enérgica en graves ($60\text{–}250\text{ Hz}$) y subgraves ($<60\text{ Hz}$) que proporciona un estímulo propioceptivo constante.
   - **Resonancia estocástica**: La textura sostenida de sintetizadores actúa como una envolvente estructurada (análoga al ruido marrón o rosa), enmascarando distractores súbitos del entorno.
   - **Pulso isocrónico estable a 90 BPM**: Facilita el estado de flujo e hiperfoco en tareas de programación, diseño y trabajo operativo.

3. **❌ Géneros no recomendados para la concentración**:
   - **Salsa (*Gotas de Lluvia*)**: Polirritmia compleja, transitorios agudos en metales y síncopas que interrumpen la atención ejecutiva.
   - **Reguetón (*Diva Virtual*)**: Patrón rítmico percusivo agresivo (*Dem Bow*) con vacíos dinámicos y lírica que satura la red de saliencia auditiva.
   - **Música de cantina (*Repítela*)**: Contenido verbal altamente dramático y fluctuaciones dinámicas que desvían los recursos cognitivos.

## 7. Pruebas automáticas

El proyecto incluye 26 pruebas unitarias que cubren todas las rutinas matemáticas, casos de borde y transformaciones DSP.

Para ejecutar la batería completa con el corredor nativo de Python:

```powershell
python -m unittest discover -v tests
```

O con `pytest` si está instalado en tu entorno:

```powershell
python -m pytest -v
```

Las pruebas verifican:
- Generación de señales multitonales, dos tonos y pulso rectangular.
- Tasa de Nyquist y vector temporal discreto.
- Casos de borde en cálculo de aliasing (frecuencias cero, múltiplos exactos de $f_s$, frecuencias negativas y $f_s$ inválida).
- Transformada de Fourier numérica frente a la solución analítica $\operatorname{sinc}$.
- Espectro unilateral normalizado por FFT con señales senoidales y arrays vacíos.
- Cuantización ADC/DAC con arrays vacíos, acotamiento de error y pasos cuánticos para $4, 8, 12, 16$ bits.
- Cálculo de SNR ante silencio absoluto ($-\infty\text{ dB}$) y cero error ($+\infty\text{ dB}$).
- Interpolación lineal continua y decimación de envolvente min-max.
- Espectro en dB relativos ante magnitudes nulas y validación de frecuencias máximas.
- Remuestreo polifásico, retención ZOH con tiempos negativos y flujo integral de análisis de audio.
- Conversión y normalización de muestras PCM (`int16`, `uint8`, flotantes y arrays vacíos).

## 8. Estructura del proyecto

```text
proyect1/
├── main.py                     # Punto de entrada de la aplicación
├── requirements.txt            # Dependencias del entorno de ejecución
├── requirements-dev.txt        # Dependencias de desarrollo y pruebas
├── README.md                   # Documentación técnica completa
├── dsp/
│   ├── __init__.py             # Exportación pública del paquete dsp
│   ├── audio_io.py             # Carga y normalización de archivos de audio (WAV/MP3)
│   ├── calculations.py         # Cálculos matemáticos puros para señales, muestreo y FFT
│   ├── models.py               # Modelo de datos AudioTrack
│   ├── processing.py           # Pipeline DSP: remuestreo, espectro, ADC y DAC
│   └── reporting.py            # Generación de informes CSV y TXT
├── ui/
│   ├── __init__.py             # Exportación pública del paquete ui
│   ├── app.py                  # Controlador y ventana principal Tkinter
│   └── plots.py                # Gestor de figuras Matplotlib (PlotManager)
├── data/
│   └── audio/                  # Canciones de prueba de los cinco géneros
│       ├── capricho_24.wav         # Clásica / Barroca
│       ├── Alan-Walker-Faded.wav   # Electrónica
│       ├── gotas_de_lluvia.wav     # Salsa
│       ├── repitela.wav            # Música de cantina
│       └── diva_virtual.wav        # Reguetón
└── tests/
    ├── test_calculations.py    # Pruebas de cálculos matemáticos puros
    └── test_processing.py      # Pruebas de procesamiento de audio, ADC y DAC
```

## 9. Limitaciones conocidas

- Se admiten archivos WAV con codificación PCM estándar y punto flotante compatibles con `soundfile` y `wave`.
- La conversión de estéreo a mono se efectúa mediante promedio simple de canales, una decisión práctica orientada a la comparación espectral.
- La simulación ADC asume cuantización uniforme lineal mid-rise; no modela no-linealidades físicas de conversores de hardware real (DNL/INL).
- La salida de audio analógica se aproxima por retención de orden cero (ZOH), que representa el escalonamiento de un conversor DAC sin filtro de reconstrucción analógico posterior.
