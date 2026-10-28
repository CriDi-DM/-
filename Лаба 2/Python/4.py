import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection

# ==========================================
# НАСТРОЙКИ АНИМАЦИИ
# ==========================================
G           = 1.0
EPS         = 0.05
M_TOTAL     = 1.0
N           = 400        # число частиц
STEPS       = 400        # число шагов интегрирования
DT          = 0.005      # шаг по времени
THETA       = 0.5        # параметр Барнса-Хата
USE_BH      = True       # True = Барнс-Хат, False = прямой метод
SAVE_EVERY  = 2          # сохранять кадр каждые SAVE_EVERY шагов
SHOW_TREE   = True       # показывать QuadTree поверх частиц
SAVE_GIF    = True       # сохранить анимацию в GIF
SHOW_WINDOW = False      # показать живое окно (если SAVE_GIF=False)
GIF_NAME    = 'lab2_nbody.gif'
FPS         = 30

# ==========================================
# МОДЕЛЬ (из предыдущих этапов)
# ==========================================
def initialize_particles(N, R=1.0):
    np.random.seed(42)
    r = R * np.sqrt(np.random.uniform(0, 1, N))
    phi = np.random.uniform(0, 2*np.pi, N)
    pos = np.column_stack((r*np.cos(phi), r*np.sin(phi)))
    v = 0.3 * np.sqrt(G * M_TOTAL * r / (r**2 + EPS**2)**0.75)
    vel = np.column_stack((-v*np.sin(phi), v*np.cos(phi)))
    mass = np.full(N, M_TOTAL / N)
    return pos, vel, mass

def compute_forces_pp(pos, mass):
    n = len(pos)
    dx = pos[:,0].reshape(1,n) - pos[:,0].reshape(n,1)
    dy = pos[:,1].reshape(1,n) - pos[:,1].reshape(n,1)
    r2 = dx*dx + dy*dy + EPS**2
    inv_r3 = 1.0 / (r2 * np.sqrt(r2))
    np.fill_diagonal(inv_r3, 0.0)
    ax = G * np.sum(mass.reshape(1,n) * inv_r3 * dx, axis=1)
    ay = G * np.sum(mass.reshape(1,n) * inv_r3 * dy, axis=1)
    return np.column_stack((ax, ay))

class QuadTreeNode:
    __slots__ = ['x','y','w','mass','cx','cy','children','body','is_external']
    def __init__(self, x, y, w):
        self.x, self.y, self.w = x, y, w
        self.mass, self.cx, self.cy = 0.0, 0.0, 0.0
        self.children, self.body, self.is_external = None, -1, True

    def insert(self, idx, pos, mass, depth=0):
        px, py = pos[idx]; pm = mass[idx]
        new_mass = self.mass + pm
        self.cx = (self.cx*self.mass + px*pm) / new_mass
        self.cy = (self.cy*self.mass + py*pm) / new_mass
        self.mass = new_mass
        if depth > 50: return
        if self.is_external:
            if self.body == -1: self.body = idx
            else:
                self.subdivide()
                old = self.body; self.body = -1
                self._into_child(old, pos, mass, depth+1)
                self._into_child(idx, pos, mass, depth+1)
        else:
            self._into_child(idx, pos, mass, depth+1)

    def _into_child(self, idx, pos, mass, depth):
        px, py = pos[idx]
        mx, my = self.x + self.w/2, self.y + self.w/2
        ix = 1 if px >= mx else 0
        iy = 1 if py >= my else 0
        self.children[ix + 2*iy].insert(idx, pos, mass, depth)

    def subdivide(self):
        self.is_external = False
        self.children = [None]*4
        h = self.w/2
        self.children[0] = QuadTreeNode(self.x,   self.y,   h)
        self.children[1] = QuadTreeNode(self.x+h, self.y,   h)
        self.children[2] = QuadTreeNode(self.x,   self.y+h, h)
        self.children[3] = QuadTreeNode(self.x+h, self.y+h, h)

    def compute_force(self, idx, pos, mass, theta):
        if self.mass == 0: return 0.0, 0.0
        dx = self.cx - pos[idx,0]
        dy = self.cy - pos[idx,1]
        d2 = dx*dx + dy*dy
        if self.is_external:
            if self.body == idx: return 0.0, 0.0
        else:
            d = np.sqrt(d2)
            if not (d > 0 and self.w / d < theta):
                ax = ay = 0.0
                for ch in self.children:
                    if ch is not None and ch.mass > 0:
                        cx, cy = ch.compute_force(idx, pos, mass, theta)
                        ax += cx; ay += cy
                return ax, ay
        r2 = d2 + EPS**2
        inv_r3 = 1.0 / (r2 * np.sqrt(r2))
        return G*self.mass*dx*inv_r3, G*self.mass*dy*inv_r3

def compute_forces_bh(pos, mass, theta):
    lo, hi = np.min(pos, axis=0), np.max(pos, axis=0)
    w = max(hi[0]-lo[0], hi[1]-lo[1]) * 1.01
    c = (lo + hi) / 2
    root = QuadTreeNode(c[0]-w/2, c[1]-w/2, w)
    for i in range(len(pos)): root.insert(i, pos, mass)
    acc = np.zeros((len(pos), 2))
    for i in range(len(pos)):
        acc[i,0], acc[i,1] = root.compute_force(i, pos, mass, theta)
    return acc

def compute_energy(pos, vel, mass):
    K = 0.5 * np.sum(mass * np.sum(vel**2, axis=1))
    dx = pos[:,0].reshape(1,-1) - pos[:,0].reshape(-1,1)
    dy = pos[:,1].reshape(1,-1) - pos[:,1].reshape(-1,1)
    r = np.sqrt(dx**2 + dy**2 + EPS**2)
    np.fill_diagonal(r, np.inf)
    U = -0.5 * G * np.sum(np.outer(mass, mass) / r)
    return K, U, K + U

def tree_segments(pos, mass):
    """Собирает границы всех ячеек дерева в список отрезков для быстрой отрисовки."""
    lo, hi = np.min(pos, axis=0), np.max(pos, axis=0)
    w = max(hi[0]-lo[0], hi[1]-lo[1]) * 1.01
    c = (lo + hi) / 2
    root = QuadTreeNode(c[0]-w/2, c[1]-w/2, w)
    for i in range(len(pos)): root.insert(i, pos, mass)
    segs = []
    stack = [root]
    while stack:
        nd = stack.pop()
        if nd.mass == 0: continue
        x, y, ww = nd.x, nd.y, nd.w
        segs += [[(x,y),(x+ww,y)], [(x+ww,y),(x+ww,y+ww)],
                 [(x+ww,y+ww),(x,y+ww)], [(x,y+ww),(x,y)]]
        if nd.children:
            stack += [ch for ch in nd.children if ch is not None]
    return segs

# ==========================================
# ПРЕДРАСЧЁТ ЭВОЛЮЦИИ СИСТЕМЫ
# ==========================================
print(f"Расчёт эволюции: N={N}, steps={STEPS}, метод={'Барнс-Хат' if USE_BH else 'прямой'}...")
pos, vel, mass = initialize_particles(N)
force = (lambda p, m: compute_forces_bh(p, m, THETA)) if USE_BH else compute_forces_pp
acc = force(pos, mass)

pos_frames  = [pos.copy()]
spd_frames  = [np.hypot(vel[:,0], vel[:,1])]
energies    = [compute_energy(pos, vel, mass)[2]]

t0 = time.time()
for step in range(1, STEPS+1):
    pos = pos + vel*DT + 0.5*acc*DT**2          # velocity Verlet: координаты
    acc_new = force(pos, mass)                  # новые ускорения
    vel = vel + 0.5*(acc + acc_new)*DT          # velocity Verlet: скорости
    acc = acc_new
    if step % SAVE_EVERY == 0:
        pos_frames.append(pos.copy())
        spd_frames.append(np.hypot(vel[:,0], vel[:,1]))
        energies.append(compute_energy(pos, vel, mass)[2])
    if step % 100 == 0:
        print(f"  шаг {step}/{STEPS}  ({time.time()-t0:.1f} с)")

pos_frames = np.array(pos_frames)   # (F, N, 2)
spd_frames = np.array(spd_frames)   # (F, N)
energies   = np.array(energies)     # (F,)
F = len(pos_frames)
print(f"Готово: {F} кадров за {time.time()-t0:.1f} с. Дрейф энергии: "
      f"{(energies[-1]-energies[0])/abs(energies[0])*100:.3f}%")

# ==========================================
# ПОДГОТОВКА СЦЕНЫ
# ==========================================
lo = np.percentile(pos_frames, 1, axis=(0,1))
hi = np.percentile(pos_frames, 99, axis=(0,1))
margin = 0.10 * (hi - lo)
lo -= margin; hi += margin
span = max(hi[0]-lo[0], hi[1]-lo[1])
cen = (hi + lo) / 2
lo, hi = cen - span/2, cen + span/2

vmin, vmax = spd_frames.min(), np.percentile(spd_frames, 99)

fig = plt.figure(figsize=(13, 7))
ax  = fig.add_axes([0.06, 0.07, 0.62, 0.86])
axE = fig.add_axes([0.76, 0.10, 0.21, 0.80])

ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
ax.set_aspect('equal'); ax.set_xlabel('X'); ax.set_ylabel('Y')

tree_lc = LineCollection([], colors='0.55', linewidths=0.4, alpha=0.45)
ax.add_collection(tree_lc)

scat = ax.scatter([], [], s=7, cmap='plasma', vmin=vmin, vmax=vmax, zorder=5)
cbar = fig.colorbar(scat, ax=ax, pad=0.015)
cbar.set_label('скорость |v|')

lineE, = axE.plot([], [], color='purple', lw=1.2)
axE.set_xlim(0, F-1)
e_pad = max((energies.max()-energies.min())*0.15, 1e-4)
axE.set_ylim(energies.min()-e_pad, energies.max()+e_pad)
axE.set_xlabel('кадр'); axE.set_ylabel('E'); axE.grid(alpha=0.3)
axE.set_title('Полная энергия')

method_name = f'Барнс-Хат, $\\theta$={THETA}' if USE_BH else 'прямой метод PP'

def update(i):
    scat.set_offsets(pos_frames[i])
    scat.set_array(spd_frames[i])
    tree_lc.set_segments(tree_segments(pos_frames[i], mass) if SHOW_TREE else [])
    lineE.set_data(np.arange(i+1), energies[:i+1])
    ax.set_title(f'Гравитационная задача N тел ({method_name}), N={N} | '
                 f'кадр {i}/{F-1}, t={i*SAVE_EVERY*DT:.2f}')
    return scat, tree_lc, lineE

anim = FuncAnimation(fig, update, frames=F, interval=1000/FPS, blit=False)

# ==========================================
# СОХРАНЕНИЕ / ПОКАЗ
# ==========================================
if SAVE_GIF:
    print("Сохранение GIF (может занять ~1 минуту)...")
    anim.save(GIF_NAME, writer=PillowWriter(fps=FPS))
    print(f"Готово! Файл: {GIF_NAME}")
if SHOW_WINDOW or not SAVE_GIF:
    plt.show()