# Mira (Señales DSP)

Aplicación de escritorio hecha en Python para cargar y analizar archivos de audio WAV. Permite ver las señales, estudiar sus frecuencias y simular procesos de conversión ADC y DAC.

## 1. Funciones principales

- Incluye cinco canciones de ejemplo de distintos géneros musicales.
- Permite cargar archivos WAV propios.
- Muestra información del audio: frecuencia de muestreo, canales, duración y bits.
- Permite elegir el fragmento de audio que se desea analizar.
- Convierte audio estéreo a mono para facilitar la comparación.
- Permite cambiar la frecuencia de procesamiento y la resolución del ADC.
- Muestra la señal original, las muestras, la cuantización y la reconstrucción DAC.
- Presenta el espectro de frecuencias mediante FFT y señala la frecuencia dominante.
- Compara las características espectrales de las canciones.
- Muestra la distribución de energía en siete bandas de frecuencia.
- Permite exportar resultados en archivos CSV y TXT.
- Incluye pruebas automáticas para revisar los cálculos principales.

## 2. Requisitos

- Python 3.10 o superior.
- Windows, macOS o Linux.
- Las siguientes bibliotecas:
  - `numpy`
  - `scipy`
  - `matplotlib`
  - `soundfile`
- `sounddevice` es opcional y se necesita para reproducir audio desde la aplicación.

## 3. Instalación y ejecución

En Windows, abre PowerShell en la carpeta del proyecto y ejecuta:

```powershell
# Crear un entorno virtual
python -m venv .venv

# Activarlo
.\.venv\Scripts\Activate.ps1

# Instalar las dependencias
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Iniciar la aplicación
python main.py
```

Si no puedes activar el entorno virtual, ejecuta la aplicación directamente:

```powershell
.\.venv\Scripts\python.exe main.py
```

## 4. Uso básico

1. **Seleccionar una canción:** al abrir el programa aparecen las cinco pistas de ejemplo.
2. **Cargar otro archivo:** usa el botón **Cargar / reemplazar WAV**.
3. **Editar la información:** puedes cambiar el título, el intérprete, el año y la referencia.
4. **Ajustar los parámetros:** selecciona los bits del ADC y modifica la frecuencia de procesamiento o el rango de voltaje.
5. **Elegir el fragmento:** indica el segundo de inicio y de fin.
6. **Revisar las pestañas:**
   - **Señal y ADC-DAC:** muestra la señal y los procesos de conversión.
   - **Canción completa:** muestra la forma de onda del audio.
   - **Espectro FFT:** muestra las frecuencias presentes y la frecuencia dominante.
   - **Comparación de géneros:** permite comparar los espectros de las canciones.
   - **Conclusiones:** muestra la energía por bandas y el análisis incluido en el proyecto.
7. **Guardar los resultados:** usa los botones de exportación CSV o TXT.

## 5. Conceptos principales

### Muestreo y frecuencia de Nyquist

El muestreo toma valores de una señal en intervalos de tiempo. Para representar correctamente una frecuencia, la frecuencia de muestreo debe ser, como mínimo, el doble de la frecuencia más alta que se desea analizar:

\[
f_s \geq 2 f_{\max}
\]

Si no se cumple esta condición, puede aparecer **aliasing**, que hace que una frecuencia se represente como otra distinta.

### Transformada de Fourier y FFT

La transformada de Fourier permite analizar las frecuencias que componen una señal. El programa utiliza la FFT para calcular y mostrar el espectro del audio.

La **frecuencia dominante** es la frecuencia que presenta el mayor pico en el espectro analizado. No necesariamente corresponde a la nota principal percibida ni al tono de toda la canción.

### ADC y cuantización

Un ADC convierte una señal analógica en valores digitales. En el programa se simula una cuantización uniforme dentro de un rango de voltaje configurable.

La cantidad de niveles depende de los bits:

\[
N_{\text{niveles}} = 2^B
\]

Por ejemplo, 8 bits representan 256 niveles y 16 bits representan 65.536 niveles. Con más bits, los niveles son más cercanos entre sí y el error de cuantización puede disminuir.

El tamaño de cada paso de voltaje se calcula así:

\[
\Delta V = \frac{V_{\max}-V_{\min}}{2^B}
\]

### DAC y reconstrucción

Un DAC convierte valores digitales en una señal reconstruida. La aplicación simula la retención de orden cero (ZOH), que mantiene cada muestra hasta que llega la siguiente.

También calcula dos medidas de error entre la señal original y la reconstruida:

- **RMSE:** mide el error cuadrático medio y da más peso a los errores grandes.
- **MAE:** calcula el promedio de los errores absolutos.

### Bandas de frecuencia

El programa divide el espectro en siete bandas para comparar cómo se distribuye la energía de las canciones, desde los subgraves hasta los muy agudos.

## 6. Música y concentración

La pestaña **Conclusiones** incluye una comparación de las canciones y un análisis relacionado con la concentración y el TDAH. Este resultado debe interpretarse como una orientación basada en las características de audio utilizadas por el programa, no como una evaluación clínica ni como una recomendación médica.

## 7. Ejecutar las pruebas

Para ejecutar las pruebas automáticas:

```powershell
python -m unittest discover -v tests
```

Si tienes `pytest` instalado, también puedes usar:

```powershell
python -m pytest -v
```

Las pruebas revisan los cálculos de señales, muestreo, aliasing, Fourier, FFT, cuantización, reconstrucción y procesamiento de audio.

## 8. Estructura del proyecto

```text
proyect1/
├── main.py
├── requirements.txt
├── requirements-dev.txt
├── README.md
├── dsp/
│   ├── audio_io.py
│   ├── calculations.py
│   ├── models.py
│   ├── processing.py
│   └── reporting.py
├── ui/
│   ├── app.py
│   └── plots.py
├── data/
│   └── audio/
│       ├── capricho_24.wav
│       ├── Alan-Walker-Faded.wav
│       ├── gotas_de_lluvia.wav
│       ├── repitela.wav
│       └── diva_virtual.wav
└── tests/
    ├── test_calculations.py
    └── test_processing.py
```

- `main.py`: inicia la aplicación.
- `dsp/`: contiene los cálculos y el procesamiento de audio.
- `ui/`: contiene la interfaz gráfica y las gráficas.
- `data/audio/`: guarda las canciones de ejemplo.
- `tests/`: contiene las pruebas automáticas.

## 9. Limitaciones

- El análisis está pensado para archivos WAV compatibles con las bibliotecas utilizadas.
- El audio estéreo se convierte a mono mediante el promedio de sus canales.
- El ADC y el DAC son simulaciones por software, no modelos completos de hardware real.
- La reconstrucción DAC usa retención de orden cero y no incluye un filtro analógico de reconstrucción.
