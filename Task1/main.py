import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


IMAGE_PATH = "9.png"

# ============================================================
# ПИКИ, КОТОРЫЕ НУЖНО УБРАТЬ
# ============================================================
PEAK_OFFSETS = [
    (-10, -12),
]

# ============================================================
# НАСТРОЙКИ ОБВОДКИ НА ТРЕТЬЕМ ГРАФИКЕ
# ============================================================
# Радиус красной окружности вокруг найденных компонент
CIRCLE_RADIUS = 8

# Толщина линии окружности
CIRCLE_LINEWIDTH = 2.0

# ============================================================
# НАСТРОЙКИ ДЕЛИКАТНОГО ЗАМАЗЫВАНИЯ
# ============================================================
# Радиус области, которую меняем:
# 1 -> область 3x3
# 2 -> область 5x5
# 3 -> область 7x7
REGION_RADIUS = 4

# Радиус внешней области, по которой берётся среднее.
# Должен быть больше REGION_RADIUS.
CONTEXT_RADIUS = 5

# Сила замазывания:
# 0.3 — очень мягко
# 0.6 — средне
# 1.0 — максимально сильно
BLEND = 1

# Плавность замазывания:
# меньше — сильнее только центр,
# больше — плавнее и шире распределение
SIGMA = 1

# ============================================================
# НАСТРОЙКИ ПРИБЛИЖЕНИЯ
# ============================================================
# Радиус окна вокруг центра, которое будет показано в zoom-версии.
# Например, 35 означает окно примерно (2*35+1)x(2*35+1)
ZOOM_RADIUS = 40

# Увеличение dpi для приближённых изображений
ZOOM_DPI = 300


def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float64) / 255.0


def save_image(filename, img):
    img = np.clip(img, 0, 1)
    arr = (img * 255).astype(np.uint8)
    Image.fromarray(arr).save(filename)


def normalize(a):
    a = a - a.min()
    mx = a.max()
    if mx > 0:
        a = a / mx
    return a


def fourier_log_image(F):
    log_abs = np.log1p(np.abs(F))
    log_abs = normalize(log_abs)

    if log_abs.ndim == 3:
        return log_abs.mean(axis=2)

    return log_abs


def smooth_peak_region(F, x0, y0, region_radius, context_radius, blend, sigma):
    h, w = F.shape[:2]

    x1_in = max(0, x0 - region_radius)
    x2_in = min(w, x0 + region_radius + 1)
    y1_in = max(0, y0 - region_radius)
    y2_in = min(h, y0 + region_radius + 1)

    x1_out = max(0, x0 - context_radius)
    x2_out = min(w, x0 + context_radius + 1)
    y1_out = max(0, y0 - context_radius)
    y2_out = min(h, y0 + context_radius + 1)

    patch = F[y1_out:y2_out, x1_out:x2_out].copy()

    inner_x1 = x1_in - x1_out
    inner_x2 = x2_in - x1_out
    inner_y1 = y1_in - y1_out
    inner_y2 = y2_in - y1_out

    ring_mask = np.ones((patch.shape[0], patch.shape[1]), dtype=bool)
    ring_mask[inner_y1:inner_y2, inner_x1:inner_x2] = False

    neighbors = patch[ring_mask]
    if neighbors.size == 0:
        return

    mean_value = neighbors.mean(axis=0)

    inner = F[y1_in:y2_in, x1_in:x2_in, :].copy()

    yy, xx = np.mgrid[y1_in:y2_in, x1_in:x2_in]
    dist2 = (xx - x0) ** 2 + (yy - y0) ** 2

    soft_mask = np.exp(-dist2 / (2 * sigma ** 2))
    soft_mask = soft_mask / soft_mask.max()
    soft_mask = (blend * soft_mask)[..., None]

    F[y1_in:y2_in, x1_in:x2_in, :] = (
        (1 - soft_mask) * inner + soft_mask * mean_value
    )


def get_peak_points(shape, offsets):
    h, w = shape
    cx = w // 2
    cy = h // 2

    points = []

    for dx, dy in offsets:
        x = cx + dx
        y = cy + dy

        x_sym = cx - dx
        y_sym = cy - dy

        points.append((x, y))
        points.append((x_sym, y_sym))

    return points


def filter_fourier_image(F_shifted, points):
    F_filtered = F_shifted.copy()
    h, w = F_filtered.shape[:2]

    for x, y in points:
        if 0 <= x < w and 0 <= y < h:
            smooth_peak_region(
                F_filtered,
                x,
                y,
                region_radius=REGION_RADIUS,
                context_radius=CONTEXT_RADIUS,
                blend=BLEND,
                sigma=SIGMA
            )

    return F_filtered


def save_fourier_plot(filename, image, points=None, zoom=False, zoom_radius=40):
    h, w = image.shape
    cx = w // 2
    cy = h // 2

    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap="gray", interpolation="nearest")

    if points is not None:
        for x, y in points:
            circle = plt.Circle(
                (x, y),
                CIRCLE_RADIUS,
                fill=False,
                edgecolor="red",
                linewidth=CIRCLE_LINEWIDTH
            )
            plt.gca().add_patch(circle)

    if zoom:
        plt.xlim(cx - zoom_radius, cx + zoom_radius)
        # инвертируем y для корректного отображения области
        plt.ylim(cy + zoom_radius, cy - zoom_radius)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=ZOOM_DPI if zoom else 200, bbox_inches="tight", pad_inches=0)
    plt.close()


# ============================================================
# 1. Исходное изображение
# ============================================================
img = load_image(IMAGE_PATH)
h, w, channels = img.shape

save_image("01_original.png", img)


# ============================================================
# 2. Фурье-образ исходного изображения
# ============================================================
F = np.fft.fft2(img, axes=(0, 1))
F_shifted = np.fft.fftshift(F, axes=(0, 1))

log_spectrum = fourier_log_image(F_shifted)
save_image("02_fourier_log_module.png", log_spectrum)


# ============================================================
# 3. Найдём нужные точки
# ============================================================
marked_points = get_peak_points((h, w), PEAK_OFFSETS)


# ============================================================
# 4. Фурье-образ с обведёнными компонентами
# ============================================================
save_fourier_plot(
    "03_fourier_log_module_with_marked_peaks.png",
    log_spectrum,
    points=marked_points,
    zoom=False
)


# ============================================================
# 5. Фурье-образ после замазывания ненужных компонент
# ============================================================
F_filtered = filter_fourier_image(F_shifted, marked_points)

log_spectrum_filtered = fourier_log_image(F_filtered)
save_image("04_fourier_log_module_filtered.png", log_spectrum_filtered)


# ============================================================
# 6. Обратное преобразование Фурье
# ============================================================
F_unshifted = np.fft.ifftshift(F_filtered, axes=(0, 1))
img_filtered = np.fft.ifft2(F_unshifted, axes=(0, 1))
img_filtered = np.real(img_filtered)
img_filtered = np.clip(img_filtered, 0, 1)

save_image("05_filtered_image.png", img_filtered)


# ============================================================
# 7. Приближённые версии Фурье-образов
# ============================================================
save_fourier_plot(
    "02_fourier_log_module_zoom.png",
    log_spectrum,
    points=None,
    zoom=True,
    zoom_radius=ZOOM_RADIUS
)

save_fourier_plot(
    "03_fourier_log_module_with_marked_peaks_zoom.png",
    log_spectrum,
    points=marked_points,
    zoom=True,
    zoom_radius=ZOOM_RADIUS
)

save_fourier_plot(
    "04_fourier_log_module_filtered_zoom.png",
    log_spectrum_filtered,
    points=None,
    zoom=True,
    zoom_radius=ZOOM_RADIUS
)


print("Готово. Сохранены файлы:")
print("01_original.png")
print("02_fourier_log_module.png")
print("03_fourier_log_module_with_marked_peaks.png")
print("04_fourier_log_module_filtered.png")
print("05_filtered_image.png")
print("02_fourier_log_module_zoom.png")
print("03_fourier_log_module_with_marked_peaks_zoom.png")
print("04_fourier_log_module_filtered_zoom.png")

print()
print("Отмеченные точки:")
for p in marked_points:
    print(p)