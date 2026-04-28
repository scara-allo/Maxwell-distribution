import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tkinter as tk

# переменные
num_molecules = 100
box_size = 100
radius = 0.4
speed_limit = 100

def open_redactor():
    window = tk.Toplevel()
    window.title("Редактор переменных")

    # Поля для ввода
    tk.Label(window, text="количество молекул").grid(row=0, column=0)
    var1 = tk.Entry(window)
    var1.insert(0, str(num_molecules))
    var1.grid(row=0, column=1)

    tk.Label(window, text="размер области").grid(row=1, column=0)
    var2 = tk.Entry(window)
    var2.insert(0, str(box_size))
    var2.grid(row=1, column=1)

    tk.Label(window, text="ограничение скорости").grid(row=2, column=0)
    var4 = tk.Entry(window)
    var4.insert(0, str(speed_limit))
    var4.grid(row=2, column=1)

    def save():
        num_molecules = int(var1.get())
        box_size = int(var2.get())
        speed_limit = int(var4.get())
        window.destroy()

    tk.Button(window, text="Сохранить", command=save).grid(row=3, column=0, columnspan=2)


# Главное окно
root = tk.Tk()
tk.Button(root, text="Редактор переменных", command=open_redactor).pack()
root.mainloop()

positions = np.random.uniform(0, box_size, (num_molecules, 2))
angles = np.random.uniform(0, 2*np.pi, num_molecules)
speeds = np.random.uniform(0, speed_limit, num_molecules)
masses = np.random.uniform(1, 10, num_molecules)
velocities = np.column_stack((np.cos(angles)*speeds, np.sin(angles)*speeds))

all_speeds = []

# создаем фигуру
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(0, box_size)
ax.set_ylim(0, box_size)
ax.set_aspect('equal')
patches = [plt.Circle(positions[i], radius) for i in range(num_molecules)]
for p in patches:
    ax.add_patch(p)

def handle_collisions():
    for i in range(num_molecules):
        for j in range(i+1, num_molecules):
            delta = positions[j] - positions[i]
            dist = np.linalg.norm(delta)
            if dist <= 2*radius:
                delta_norm = delta / dist
                v1 = velocities[i]
                v2 = velocities[j]
                m1, m2 = masses[i], masses[j]
                v1_par = np.dot(v1, delta_norm)
                v2_par = np.dot(v2, delta_norm)
                v1_par_new = ((m1 - m2)*v1_par + 2*m2*v2_par) / (m1 + m2)
                v2_par_new = ((m2 - m1)*v2_par + 2*m1*v1_par) / (m1 + m2)
                velocities[i] += (v1_par_new - v1_par) * delta_norm
                velocities[j] += (v2_par_new - v2_par) * delta_norm

def update_positions():
    global positions, velocities
    positions += velocities * 0.1
    for i in range(num_molecules):
        for d in range(2):
            if positions[i, d] < radius:
                positions[i, d] = radius
                velocities[i, d] *= -1
            elif positions[i, d] > box_size - radius:
                positions[i, d] = box_size - radius
                velocities[i, d] *= -1

def animate(frame):
    handle_collisions()
    update_positions()

    current_speeds = np.linalg.norm(velocities, axis=1)
    all_speeds.append(current_speeds.copy())

    for i, p in enumerate(patches):
        p.center = positions[i]
    return patches

def plot_speed_distribution():
    speeds_array = np.concatenate(all_speeds)
    plt.figure()
    plt.hist(speeds_array, bins=20, density=True)
    plt.xlabel('скорость')
    plt.ylabel('частота')
    plt.show()

ani = animation.FuncAnimation(fig, animate, frames=200, interval=50, blit=True)

plt.show()
plot_speed_distribution()
