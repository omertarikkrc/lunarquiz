import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from numpy import cos, sin  # cos ve sin fonksiyonlarını numpy kütüphanesinden alıyoruz

# --- Parametreler ---
a = 5000  # km (semi-major axis)
e = 0.3  # eksantriklik
i = 45    #inclination
w = 60    #perigeelerin tartışması
# --- Yörünge hesaplaması (perifokal frame) ---
theta = np.linspace(0, 2 * np.pi, 100)  # Açılar
r = (a * (1 - e**2)) / (1 + e * np.cos(theta))  # Yörünge büyüklüğü

x = r * np.cos(theta)  # x koordinatları
y = r * np.sin(theta)  # y koordinatları
z = np.zeros_like(x)  # z koordinatları (çünkü perifokal düzlemde z = 0)

# --- Animasyon figürü oluştur ---
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# --- Update fonksiyonu ---
def update(frame):
    # RAAN (Omega) açısını artırıyoruz
    Omega = np.deg2rad(frame * 45 / 60)  # 60 frame'de 45 dereceyi tamamlayacak şekilde
    
    # Z ekseni etrafında RAAN dönüşüm matrisi
    Rz_Raan = np.array([
        [cos(Omega), -sin(Omega), 0],
        [sin(Omega), cos(Omega), 0],
        [0, 0, 1]
    ])
    
    # Yörüngeyi döndürüyoruz
    pos_stack = np.vstack((x, y, z))  # x, y, z'yi birleştir
    raan_rotated = Rz_Raan @ pos_stack  # Döndürme işlemi
    
    # Eski çizimleri temizle
    ax.cla()

    # Yeni döndürülmüş yörüngeyi çiz
    ax.plot(raan_rotated[0], raan_rotated[1], raan_rotated[2], color="k", label="Orbital Frame")
    ax.scatter(0, 0, 0, color="blue", label="ECI Frame")  # Dünya merkezi (ECI)

    # Eksen limitlerini ayarla
    max_val = 7378
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.set_zlim([-max_val, max_val])

    # Etiket ve başlık
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.set_title("Transition from ECI to Orbital Frame")
    ax.legend()

    return ax

# --- Animasyonu başlat ---
ani = FuncAnimation(fig, update, frames=60, interval=150, blit=False)

plt.show()
