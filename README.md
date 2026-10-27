# Mira (AudioLab DSP) — análisis de audio mediante ADC y DAC

Aplicación de escritorio en Python para cargar y comparar fragmentos musicales WAV. Incluye una interfaz gráfica interactiva, representación temporal en alta resolución, remuestreo, análisis espectral mediante FFT, cuantización ADC uniforme, reconstrucción DAC (retención de orden cero e interpolación), biblioteca de cálculos matemáticos desacoplada y exportación de resultados.

## 1. Qué incluye

- Biblioteca de cinco pistas para las categorías: clásica/barroca, electrónica, salsa, música de cantina y reguetón.
- Carga y reemplazo de archivos WAV desde el computador.
- Ficha editable con título, compositor/intérprete, año y fuente.
- Visualización de frecuencia original, canales, duración y profundidad de bits cuando se puede obtener.
- Conversión a mono para analizar pistas con uno o más canales.
- Frecuencia de muestreo de procesamiento configurable y cálculo de la frecuencia de Nyquist.
- Gráficas de forma de onda completa con zoom inteligente (decimación min-max), muestras remuestreadas, señal cuantizada y señal reconstruida.
- Espectro de magnitud por FFT y frecuencia dominante estimada.
- Resolución ADC seleccionable, número de niveles, rango de entrada, factor de escala, paso de cuantización y error.
- Reconstrucción DAC aproximada mediante retención de orden cero (ZOH) e interpolación lineal.
- Comparación espectral entre las pistas cargadas, una tabla de métricas y una sección prudente sobre música, concentración y TDAH.
- Exportación de resumen CSV e informe TXT.
- Biblioteca modular de cálculos matemáticos reutilizables (`dsp.calculations`).
- Pruebas unitarias completas tanto con `pytest` como con el módulo estándar `unittest`.

## 2. Requisitos

- Python 3.10 o superior (compatible con NumPy 1.x y NumPy 2.x).
- Windows, macOS o Linux con entorno gráfico para Tkinter.
- Paquetes: NumPy, SciPy y Matplotlib (opcional: sounddevice / pygame para reproducción).

Tkinter suele venir incluido en la instalación estándar de Python para Windows. En algunas distribuciones Linux puede requerir el paquete del sistema `python3-tk`.

## 3. Instalación en Windows

### Método recomendado

1. Extrae el archivo en una carpeta local.
2. Instala Python 3.10 o superior desde el sitio oficial y activa la opción de agregar Python al PATH si corresponde.
3. Haz doble clic en `run_windows.bat` o inicia desde PowerShell.

### Método manual (PowerShell)

Abre PowerShell en la carpeta del proyecto y ejecuta:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

Si Windows bloquea la activación del entorno virtual, puedes ejecutar directamente:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

## 4. Uso básico

1. Selecciona una de las cinco pistas en la tabla de la izquierda.
2. Pulsa **Cargar / reemplazar WAV** y selecciona un archivo `.wav`.
3. Completa la ficha musical y pulsa **Guardar ficha**.
4. Repite el proceso para los demás géneros. Puedes trabajar con una o varias pistas; no es obligatorio cargarlas todas para usar el análisis individual.
5. Modifica la frecuencia de procesamiento, la resolución ADC y el rango de entrada. Pulsa **Aplicar y recalcular** para actualizar las gráficas y métricas.
6. Usa las pestañas para explorar la señal temporal, la forma de onda de la canción completa, el espectro FFT, la comparación y las conclusiones.
7. Pulsa **Exportar CSV** para obtener la tabla de resultados o **Guardar informe TXT** para guardar el informe técnico y las referencias documentales.

La aplicación analiza hasta los primeros 10 segundos configurables de cada pista para mantener el tiempo de cálculo acotado. La duración total original sí aparece en la ficha de archivo. Si un WAV es estéreo, el análisis usa la media de sus canales; el archivo fuente no se modifica.

## 5. Fundamentos y cálculos matemáticos

El proyecto organiza sus cálculos matemáticos en `dsp/calculations.py` y `dsp/processing.py`, desacoplándolos de la presentación visual (`ui/plots.py` y `ui/app.py`).

### 5.1 Generación de señales continuas simuladas
Permite simular señales en el dominio del tiempo continuo evaluadas sobre cuadrículas de alta resolución:
- Suma de componentes sinusoidales compuestas:
  \[
  x_c(t) = \sum_{i} A_i \sin(2\pi f_i t)
  \]
- Pulso rectangular simétrico:
  \[
  x(t) = \begin{cases} 1, & |t| \le \frac{T}{2} \\ 0, & \text{en otro caso} \end{cases}
  \]

### 5.2 Muestreo y tasa de Nyquist-Shannon
Para una señal con frecuencia máxima \(f_{\max}\), la tasa teórica mínima de muestreo es:
\[
f_{s,\text{nyquist}} = 2 f_{\max}
\]
Los instantes de muestreo discretos se calculan como \(t_n = n / f_s\) para \(n \in [0, N-1]\) con período \(T_s = 1 / f_s\).
El remuestreo de audio grabado se realiza mediante `scipy.signal.resample_poly`, aplicando filtrado antialiasing polifásico.

### 5.3 Frecuencia aparente por submuestreo (Aliasing)
Cuando la frecuencia de muestreo \(f_s\) es insuficiente (\(f_s < 2 f_{\text{signal}}\)), la componente espectral se pliega en el intervalo base de Nyquist \([0, f_s / 2]\):
\[
f_{\text{alias}} = \left| \left(f_{\text{signal}} + \frac{f_s}{2}\right) \bmod f_s - \frac{f_s}{2} \right|
\]
Esta formulación modular general evita inconsistencias de redondeo y abarca armónicos de orden arbitrario.

### 5.4 Transformada de Fourier por integración directa
Aproxima numéricamente la integral continua de Fourier sin depender de la FFT:
\[
X(f) = \int_{-\infty}^{\infty} x(t) e^{-j 2\pi ft} \, dt
\]
Se implementa con soporte automático para `np.trapezoid` (NumPy 2.x) y respaldo retrocompatible en `np.trapz` (NumPy < 2.0).
Para un pulso rectangular de ancho \(T\), se valida contra la transformada analítica:
\[
X_{\text{teorica}}(f) = T \cdot \operatorname{sinc}(f T)
\]

### 5.5 Transformada Rápida de Fourier (FFT) y espectro de amplitud
Calcula el espectro discreto unilateral con eje de frecuencias calibrado en Hz y normalización por amplitud:
\[
|X[k]| = \frac{2}{N} \left| \operatorname{FFT}(x)[k] \right|
\]
Las componentes DC (\(k=0\)) y de Nyquist (\(k=N/2\)) reciben el factor \(1/N\) correspondiente para conservar la energía exacta.

### 5.6 Cuantización ADC y reconstrucción DAC
Para una resolución de \(B\) bits y rango de voltaje \([V_{\min}, V_{\max}]\):
\[
N_{\text{niveles}} = 2^B, \qquad \Delta V = \frac{V_{\max} - V_{\min}}{2^B}
\]
Se proporcionan dos modelos reutilizables:
1. Cuantizador uniforme **mid-rise** (usado en el análisis de audio de la interfaz), donde los voltajes reconstruidos se sitúan en los centros de celda:
   \[
   V_{\text{cuant}} = V_{\min} + (\text{código} + 0.5) \cdot \Delta V
   \]
2. Cuantizador directo lineal por redondeo (Snippet 6):
   \[
   \text{código} = \operatorname{clip}\left(\operatorname{round}\left(\frac{v - V_{\min}}{\Delta V}\right), 0, 2^B - 1\right)
   \]
   \[
   v_{\text{reconstruido}} = \text{código} \cdot \Delta V + V_{\min}, \qquad e_q = v - v_{\text{reconstruido}}
   \]

### 5.7 Reconstrucción ZOH e interpolación lineal
- **ZOH (Zero-Order Hold)**: mantiene constante cada muestra durante el intervalo \(T_s\).
- **Interpolación lineal**: reconstruye valores continuos entre muestras mediante `np.interp` conectando puntos adyacentes.

### 5.8 Preparación de datos y decimación para gráficas
Para renderizar formas de onda extensas sin sobrecargar la memoria ni ralentizar la interfaz, la función `compute_minmax_downsample` extrae la envolvente de picos mínimos y máximos por cubetas temporales, garantizando que los transitorios sonoros nunca desaparezcan durante el zoom.

## 6. Música, concentración y TDAH

El programa separa el análisis medible del audio de la interpretación documental. La frecuencia dominante, la distribución espectral, el rango de amplitud o el error ADC/DAC no bastan para determinar qué género mejora la atención de una persona con TDAH.

Las revisiones disponibles describen interés en el uso de música y de intervenciones musicales, pero también limitaciones relacionadas con el número de estudios, la heterogeneidad de intervenciones y participantes, y los resultados evaluados. Por ello, la aplicación no etiqueta ningún género como terapéutico ni elige una categoría ganadora de forma automática. Una evaluación de concentración requeriría medidas conductuales apropiadas y un diseño de comparación que controle la tarea, el volumen, la presencia de letras, las preferencias personales y otras variables relevantes.

Referencias iniciales para el análisis documental:

1. *Exploring the Intersection of ADHD and Music: A Systematic Review* (2025). PubMed: https://pubmed.ncbi.nlm.nih.gov/39851869/
2. *Outcomes of music therapy on children and adolescents with attention-deficit/hyperactivity disorder: a systematic review and meta-analysis* (2025). PubMed: https://pubmed.ncbi.nlm.nih.gov/40680190/
3. *Effects of Music on Attention-Deficit/Hyperactivity Disorder (ADHD) and Potential Application in Serious Video Games: Systematic Review* (2023). PubMed: https://pubmed.ncbi.nlm.nih.gov/37171837/

Estas fuentes sirven como punto de partida, no como prueba de que un género específico sea adecuado para todos los estudiantes.

## 7. Pruebas automáticas

Para ejecutar la batería completa de pruebas unitarias:

Con el ejecutor nativo de Python:

```powershell
python -m unittest discover tests
```

O si tienes instalado `pytest`:

```powershell
python -m pytest -q
```

Las pruebas verifican:
- Generación de señales multitonales y pulso rectangular.
- Tasa de Nyquist y vector de cuadrícula temporal.
- Cálculo de aliasing (fórmula general y armónicos redondeados).
- Transformada de Fourier numérica frente a la solución teórica `sinc`.
- Espectro unilateral normalizado por FFT.
- Niveles, códigos, paso y voltajes reconstruidos ADC/DAC.
- Interpolación lineal entre muestras.
- Decimación de envolvente min-max para gráficas.
- Espectros normalizados en dB relativos al pico.
- Remuestreo, retención ZOH y flujo general de análisis de audio.

## 8. Estructura del proyecto

```text
Mira (AudioLab_DSP)/
├── main.py                     # Punto de entrada de la aplicación
├── requirements.txt            # Dependencias del entorno
├── requirements-dev.txt        # Dependencias de desarrollo y pruebas
├── dsp/
│   ├── calculations.py         # Biblioteca de cálculos matemáticos para señales y gráficas
│   ├── audio_io.py             # Carga y normalización de archivos de audio (WAV/MP3)
│   ├── models.py               # Modelo de datos AudioTrack
│   ├── processing.py           # Pipeline DSP: muestreo, FFT, ADC mid-rise y DAC ZOH
│   └── reporting.py            # Exportación de informes en CSV y TXT
├── ui/
│   ├── app.py                  # Controlador principal y ventana Tkinter
│   └── plots.py                # Gestor de figuras Matplotlib (PlotManager) desacoplado
├── data/audio/                 # Audios de muestra para las categorías del proyecto
└── tests/
    ├── test_calculations.py    # Pruebas de los cálculos matemáticos puros
    └── test_processing.py      # Pruebas del procesamiento DSP y ADC/DAC
```

## 9. Limitaciones conocidas

- Se aceptan archivos WAV compatibles con SciPy y soundfile.
- Los formatos WAV flotantes o PCM no convencionales pueden no ofrecer una profundidad original exacta en sus metadatos.
- La conversión estéreo a mono usa el promedio simple de canales, una decisión práctica para la comparación de señales.
- Los resultados dependen del fragmento seleccionado, la frecuencia de muestreo, el rango ADC y la resolución en bits.
