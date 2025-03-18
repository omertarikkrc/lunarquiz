import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib.lines import Line2D

# --- Parametreler ---
radius_earth = 6371  # km
a = 10000  # Yarı-büyük eksen [km]
e = 0.3  # Eksantriklik (0: dairesel, 0<e<1: eliptik)
i_deg = 90  # Inclination [deg]
raan_deg = 45  # RAAN [deg]
arg_periapsis_deg = 0  # Argüman [deg]

num_frames = 200

# --- Temel dönüşümler ---
i = np.deg2rad(i_deg)
raan = np.deg2rad(raan_deg)
arg_periapsis = np.deg2rad(arg_periapsis_deg)

# --- Dünya modeli ---
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x_sphere = radius_earth * np.outer(np.cos(u), np.sin(v))
y_sphere = radius_earth * np.outer(np.sin(u), np.sin(v))
z_sphere = radius_earth * np.outer(np.ones(np.size(u)), np.cos(v))

# --- Grafik ayarları ---
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

max_lim = a * 1.5
ax.set_xlim([-max_lim, max_lim])
ax.set_ylim([-max_lim, max_lim])
ax.set_zlim([-max_lim, max_lim])

# Dünya çizimi
ax.plot_surface(x_sphere, y_sphere, z_sphere, color='lightblue', alpha=0.6, edgecolor='none')

# --- ECI eksenleri ---
origin = np.array([[0, 0, 0]])
ECI_axes = np.eye(3)
eci_lines = []
for i_, axis in enumerate(['X', 'Y', 'Z']):
    eci = ax.quiver(*origin.T, *(radius_earth * 0.5 * ECI_axes[:, i_]), color=['r', 'g', 'b'][i_], linewidth=2)
    eci_lines.append(Line2D([0], [0], color=['r', 'g', 'b'][i_], lw=2, label=f'ECI {axis}'))

# --- Yörüngeyi parametresel çizelim ---
theta_vals = np.linspace(0, 2 * np.pi, 500)
r_vals = (a * (1 - e**2)) / (1 + e * np.cos(theta_vals))
x_orb = r_vals * np.cos(theta_vals)
y_orb = r_vals * np.sin(theta_vals)
z_orb = np.zeros_like(x_orb)

# --- Yörüngeyi 3D'ye dönüştürmek için dönüşüm matrisi ---
def rotation_matrix(raan, inc, arg_periapsis):
    Rz_raan = np.array([[np.cos(raan), -np.sin(raan), 0],
                        [np.sin(raan),  np.cos(raan), 0],
                        [0, 0, 1]])
    Rx_inc = np.array([[1, 0, 0],
                       [0, np.cos(inc), -np.sin(inc)],
                       [0, np.sin(inc),  np.cos(inc)]])
    Rz_arg = np.array([[np.cos(arg_periapsis), -np.sin(arg_periapsis), 0],
                       [np.sin(arg_periapsis),  np.cos(arg_periapsis), 0],
                       [0, 0, 1]])
    return Rz_raan @ Rx_inc @ Rz_arg

R_total = rotation_matrix(raan, i, arg_periapsis)
orbit_eci = R_total @ np.vstack((x_orb, y_orb, z_orb))

ax.plot(orbit_eci[0], orbit_eci[1], orbit_eci[2], linestyle='dotted', color='black', label='Yörünge')

# --- Başlangıçta boş quiver'lar ---
pos_quiver = None
x_quiver = None
y_quiver = None
z_quiver = None

# --- Legend için dummy çizgiler (LVLH) ---
lvlh_lines = [
    Line2D([0], [0], color='magenta', lw=2, label='LVLH X'),
    Line2D([0], [0], color='lime', lw=2, label='LVLH Y'),
    Line2D([0], [0], color='orange', lw=2, label='LVLH Z'),
    Line2D([0], [0], color='k', lw=2, label='Pozisyon (R)'),
    Line2D([0], [0], color='black', ls='dotted', lw=1.5, label='Yörünge')
]

# --- Legend ayarı ---
ax.legend(handles=eci_lines + lvlh_lines, loc='upper left')

# --- Animasyon fonksiyonu ---
def update(frame):
    global pos_quiver, x_quiver, y_quiver, z_quiver

    theta = (2 * np.pi / num_frames) * frame
    r = (a * (1 - e**2)) / (1 + e * np.cos(theta))
    perifocal_pos = np.array([r * np.cos(theta), r * np.sin(theta), 0])
    perifocal_vel = np.array([-np.sin(theta), e + np.cos(theta), 0])  # Normalize değil

    pos = R_total @ perifocal_pos
    vel = R_total @ perifocal_vel

    # LVLH frame
    z_orbit = -pos / np.linalg.norm(pos)
    x_orbit = vel / np.linalg.norm(vel)
    y_orbit = np.cross(z_orbit, x_orbit)
    y_orbit = y_orbit / np.linalg.norm(y_orbit)

    # Öncekileri sil
    if pos_quiver:
        pos_quiver.remove()
        x_quiver.remove()
        y_quiver.remove()
        z_quiver.remove()

    # Çizimler
    pos_quiver = ax.quiver(0, 0, 0, *pos, color='k', linewidth=2)
    x_quiver = ax.quiver(*pos, *(radius_earth * 0.4 * x_orbit), color='magenta', linewidth=2)
    y_quiver = ax.quiver(*pos, *(radius_earth * 0.4 * y_orbit), color='lime', linewidth=2)
    z_quiver = ax.quiver(*pos, *(radius_earth * 0.4 * z_orbit), color='orange', linewidth=2)

    return pos_quiver, x_quiver, y_quiver, z_quiver

# --- Animasyon ---
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=150, blit=False)

plt.show()
