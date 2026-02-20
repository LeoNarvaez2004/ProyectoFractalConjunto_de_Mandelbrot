# Visualizador Interactivo del Conjunto de Mandelbrot

**Equipo:** Caetano Flores · Leonardo Narváez  
**Institución:** Universidad de las Fuerzas Armadas ESPE  

---

## Descripción

Aplicación de escritorio para la exploración interactiva del fractal Conjunto de Mandelbrot, con renderizado acelerado por GPU mediante CUDA. Permite navegar el fractal en tiempo real, personalizar paletas de colores, ajustar el nivel de zoom y la intensidad de aura, y exportar imágenes de alta calidad. Desarrollado en Python con una interfaz gráfica construida con PyQt6.

---

## Características Principales

- Renderizado en GPU con **CUDA** mediante `Numba` para alta performance
- Zoom interactivo y navegación libre por el plano complejo
- Paletas de color personalizables y múltiples modos de colorización
- Control de intensidad de aura/glow sobre el fractal
- Exportación de imágenes en alta resolución
- Interfaz gráfica intuitiva con controles de parámetros en tiempo real

---

## Tecnologías Utilizadas

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white)

| Tecnología | Uso |
|---|---|
| **Python 3** | Lenguaje principal |
| **PyQt6** | Interfaz gráfica de usuario |
| **NumPy** | Procesamiento numérico |
| **Numba (CUDA)** | Aceleración GPU del kernel de Mandelbrot |
| **colorsys** | Gestión y transformación de colores |

---

## Requisitos

```bash
pip install PyQt6 numpy numba
```

> Se requiere una GPU compatible con CUDA y los drivers de NVIDIA instalados.

---

## Ejecución

```bash
python Mandelbrot.py
```
