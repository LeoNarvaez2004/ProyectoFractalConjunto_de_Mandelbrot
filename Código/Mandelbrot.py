from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QInputDialog, QColorDialog, QComboBox, QFileDialog, QSpinBox, QSlider
from PyQt6.QtGui import QPixmap, QImage, QColor
from PyQt6.QtCore import Qt, QPoint
import numpy as np
import sys
from numba import cuda
import colorsys

@cuda.jit
def mandelbrot_kernel_with_aura(image, width, height, zoom, offset_x, offset_y, max_iter, palette, palette_size, color_mode, aura_intensity):
    x, y = cuda.grid(2)
    if x >= width or y >= height:
        return

    real = (x - width / 2.0) / zoom + offset_x
    imag = (y - height / 2.0) / zoom + offset_y
    c_real, c_imag = real, imag

    z_real, z_imag = 0.0, 0.0
    iter_count = 0

    z_mag_squared = 0.0

    while iter_count < max_iter and (z_real * z_real + z_imag * z_imag) < 4.0:
        temp = z_real * z_real - z_imag * z_imag + c_real
        z_imag = 2.0 * z_real * z_imag + c_imag
        z_real = temp
        iter_count += 1
        z_mag_squared = z_real * z_real + z_imag * z_imag

    if iter_count == max_iter:
        image[y, x, 0] = 0
        image[y, x, 1] = 0
        image[y, x, 2] = 0
        return
        
    smooth_value = iter_count
    if iter_count < max_iter:
        smooth_value = iter_count + 1.0 - min(1.0, z_mag_squared / 4.0)
        
    aura_factor = 0.0
    if z_mag_squared > 0.0:
        edge_proximity = min(1.0, z_mag_squared / 4.0) 
        
        aura_factor = edge_proximity * aura_intensity
    
    if color_mode == 0:  
        color_index = int(iter_count % palette_size)
        r = palette[color_index][0]
        g = palette[color_index][1]
        b = palette[color_index][2]
        
        r = min(255, int(r + (255 - r) * aura_factor))
        g = min(255, int(g + (255 - g) * aura_factor))
        b = min(255, int(b + (255 - b) * aura_factor))
    else:  
        t = smooth_value / max_iter
        index_float = t * (palette_size - 1)
        index = int(index_float)
        t_interp = index_float - index
        
        if index < palette_size - 1:
            r = int(palette[index][0] * (1.0 - t_interp) + palette[index + 1][0] * t_interp)
            g = int(palette[index][1] * (1.0 - t_interp) + palette[index + 1][1] * t_interp)
            b = int(palette[index][2] * (1.0 - t_interp) + palette[index + 1][2] * t_interp)
        else:
            r = palette[index][0]
            g = palette[index][1]
            b = palette[index][2]
            
        r = min(255, int(r * (1.0 + aura_factor * 0.7)))
        g = min(255, int(g * (1.0 + aura_factor * 0.7)))
        b = min(255, int(b * (1.0 + aura_factor * 0.7)))
    
    image[y, x, 0] = r
    image[y, x, 1] = g
    image[y, x, 2] = b

class MandelbrotGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Conjunto de Mandelbrot")
        self.setGeometry(100, 100, 900, 750)

        self.label = QLabel(self)
        self.label.setGeometry(0, 0, 900, 700)

        self.iterations_spinbox = QSpinBox(self)
        self.iterations_spinbox.setGeometry(20, 710, 100, 30)
        self.iterations_spinbox.setRange(1, 5000) 
        self.iterations_spinbox.setValue(200)  
        self.iterations_spinbox.valueChanged.connect(self.update_iterations)  

        self.button_zoom_in = QPushButton("+", self)
        self.button_zoom_in.setGeometry(140, 710, 50, 30)
        self.button_zoom_in.clicked.connect(self.zoom_in)

        self.button_zoom_out = QPushButton("-", self)
        self.button_zoom_out.setGeometry(200, 710, 50, 30)
        self.button_zoom_out.clicked.connect(self.zoom_out)

        self.color_combo = QComboBox(self)
        self.color_combo.setGeometry(270, 710, 120, 30)
        self.color_combo.addItems([
            "Fuego", "Océano", "Arcoíris", "Neón", "Cósmico", 
             "Esmeralda", "Psicodélico"
        ])
        self.color_combo.currentIndexChanged.connect(self.change_color_scheme)

        self.mode_combo = QComboBox(self)
        self.mode_combo.setGeometry(400, 710, 120, 30)
        self.mode_combo.addItems(["Paleta Simple", "Interpolación Suave"])
        self.mode_combo.currentIndexChanged.connect(self.change_color_mode)

        self.aura_label = QLabel("Aura:", self)
        self.aura_label.setGeometry(530, 710, 40, 30)
        
        self.aura_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.aura_slider.setGeometry(570, 710, 100, 30)
        self.aura_slider.setRange(0, 100)  
        self.aura_slider.setValue(50)      
        self.aura_slider.valueChanged.connect(self.change_aura_intensity)

        self.button_export = QPushButton("Exportar", self)
        self.button_export.setGeometry(680, 710, 100, 30)
        self.button_export.clicked.connect(self.export_high_res)

        self.zoom = 300.0
        self.offset_x = -0.5
        self.offset_y = 0.0
        self.max_iter = 200  
        self.aura_intensity = 1.0  

        self.is_dragging = False
        self.last_mouse_pos = QPoint()

        self.palette_size = 256
        self.palette = []
        self.color_scheme = 0
        self.color_mode = 1  
        
        self.create_palette()
        
        self.update_fractal()
        
    def update_iterations(self):
        """Actualiza el número de iteraciones cuando cambia el valor del SpinBox."""
        self.max_iter = self.iterations_spinbox.value()
        self.update_fractal()

    def change_aura_intensity(self):
        """Actualiza la intensidad del aura cuando cambia el valor del deslizador."""
        self.aura_intensity = self.aura_slider.value() / 50.0
        self.update_fractal()

    def create_palette(self):
        """Crea una paleta de colores basada en el esquema seleccionado."""
        self.palette = []
        
        if self.color_scheme == 0:  # Fuego
            for i in range(self.palette_size):
                t = i / (self.palette_size - 1)
                r = int(min(255, t * 3 * 255))
                g = int(min(255, t * t * 255))
                b = int(min(255, t * 0.5 * 100))
                self.palette.append((r, g, b))
        
        elif self.color_scheme == 1:  # Océano
            for i in range(self.palette_size):
                t = i / (self.palette_size - 1)
                r = int(t * 0.3 * 100)
                g = int(t * 0.8 * 200)
                b = int(min(255, t * 255))
                self.palette.append((r, g, b))
        
        elif self.color_scheme == 2:  # Arcoíris
            for i in range(self.palette_size):
                h = i / (self.palette_size - 1)
                r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
                self.palette.append((int(r * 255), int(g * 255), int(b * 255)))
        
        elif self.color_scheme == 3:  # Neón
            for i in range(self.palette_size):
                t = i / (self.palette_size - 1)
                h = (t * 0.8 + 0.7) % 1.0
                r, g, b = colorsys.hsv_to_rgb(h, 0.9, 1.0)
                self.palette.append((int(r * 255), int(g * 255), int(b * 255)))
        
        elif self.color_scheme == 4:  # Cósmico
            for i in range(self.palette_size):
                t = i / (self.palette_size - 1)
                if t < 0.5:
                    r = int(t * 2 * 255)
                    g = int(t * 100)
                    b = int(100 + t * 155)
                else:
                    r = int(255)
                    g = int(100 + (t - 0.5) * 2 * 155)
                    b = int(255)
                self.palette.append((r, g, b))
        
        
        elif self.color_scheme == 5:  # Esmeralda
            for i in range(self.palette_size):
                t = i / (self.palette_size - 1)
                r = int(t * 200)
                g = int(min(255, t * 2 * 255))
                b = int(t * 150)
                self.palette.append((r, g, b))
        
        elif self.color_scheme == 6:  # Psicodélico
            for i in range(self.palette_size):
                t = i / (self.palette_size - 1)
                phase = 6 * t
                
                if phase < 1:
                    r, g, b = 10, int(phase * 155), 0
                elif phase < 2:
                    r, g, b = int((2 - phase) * 155), 225, 0
                elif phase < 3:
                    r, g, b = 0, 255, int((phase - 2) * 225)
                elif phase < 4:
                    r, g, b = 0, int((4 - phase) * 155), 225
                elif phase < 5:
                    r, g, b = int((phase - 4) * 155), 0, 225
                else:
                    r, g, b = 0, 0, int((6 - phase) * 225)
                    
                self.palette.append((r, g, b))

    def ask_iterations(self):
        """Pide al usuario ingresar la cantidad de iteraciones."""
        text, ok = QInputDialog.getInt(self, "Iteraciones", "Ingrese el número de iteraciones:", 
                                       self.max_iter, 10, 5000, 10)
        if ok:
            self.max_iter = text
            self.update_fractal()

    def zoom_in(self):
        """Aumenta el zoom acercando la vista."""
        self.zoom *= 1.5
        self.update_fractal()

    def zoom_out(self):
        """Disminuye el zoom alejando la vista."""
        self.zoom /= 1.5
        self.update_fractal()

    def change_color_scheme(self, index):
        """Cambia el esquema de color."""
        self.color_scheme = index
        self.create_palette()
        self.update_fractal()

    def change_color_mode(self, index):
        """Cambia el modo de coloración."""
        self.color_mode = index
        self.update_fractal()

    def mousePressEvent(self, event):
        """Detecta cuando el usuario presiona el mouse."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = True
            self.last_mouse_pos = event.position()

    def mouseMoveEvent(self, event):
        """Detecta cuando el usuario mueve el mouse y arrastra la imagen."""
        if self.is_dragging:
            delta = event.position() - self.last_mouse_pos
            self.last_mouse_pos = event.position()

            self.offset_x -= delta.x() / self.zoom
            self.offset_y -= delta.y() / self.zoom

            self.update_fractal()

    def mouseReleaseEvent(self, event):
        """Detecta cuando el usuario suelta el botón del mouse."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = False

    def wheelEvent(self, event):
        """Detecta cuando el usuario gira la rueda del mouse para hacer zoom."""
        delta = event.angleDelta().y()

        if delta > 0:
            self.zoom *= 1.1 
        else:
            self.zoom /= 1.1  

        # Actualizar el fractal
        self.update_fractal()

    def generate_fractal(self, width, height, zoom=None, offset_x=None, offset_y=None):
        """Genera el fractal de Mandelbrot con los colores elegidos."""
        if zoom is None:
            zoom = self.zoom
        if offset_x is None:
            offset_x = self.offset_x
        if offset_y is None:
            offset_y = self.offset_y

        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        d_image = cuda.to_device(image)
        
        palette_array = np.array(self.palette, dtype=np.uint8)
        d_palette = cuda.to_device(palette_array)

        threads_per_block = (16, 16)
        blocks_per_grid_x = (width + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (height + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        mandelbrot_kernel_with_aura[blocks_per_grid, threads_per_block](
            d_image, width, height, zoom, offset_x, offset_y, self.max_iter, 
            d_palette, self.palette_size, self.color_mode, self.aura_intensity
        )

        image = d_image.copy_to_host()
        return image

    def update_fractal(self):
        """Actualiza el fractal en la interfaz gráfica."""
        width, height = 900, 700
        try:
            fractal = self.generate_fractal(width, height)
            q_image = QImage(fractal.data, width, height, 3 * width, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.label.setPixmap(pixmap)
        except Exception as e:
            print(f"Error al generar el fractal: {e}")

    def export_high_res(self):
        """Genera y guarda el fractal en una resolución alta."""
        export_width = 4000  
        export_height = 4000  
        
        print(f"Generando fractal en resolución {export_width}x{export_height}")
        
        scale_factor_width = export_width / 900  
        scale_factor_height = export_height / 700 
        
        high_res_zoom = self.zoom * min(scale_factor_width, scale_factor_height)
        
        high_res_offset_x = self.offset_x
        high_res_offset_y = self.offset_y
        
        image = self.generate_fractal(export_width, export_height, high_res_zoom, high_res_offset_x, high_res_offset_y)

        qimage = QImage(image.data, export_width, export_height, 3 * export_width, QImage.Format.Format_RGB888)
        
        file_path, _ = QFileDialog.getSaveFileName(self, "Guardar imagen", "", "PNG Files (*.png);;All Files (*)")
        if file_path:
            qimage.save(file_path, "PNG")
            print(f"Imagen guardada en: {file_path}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MandelbrotGUI()
    window.show()
    sys.exit(app.exec())