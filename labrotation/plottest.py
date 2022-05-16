from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

duration = 2  # in sec
refreshPeriod = 100  # in ms

fig, ax = plt.subplots()
vl = ax.axvline(0, ls='-', color='r', lw=1, zorder=10)
ax.set_xlim(0, duration)


def animate(i, vl, period):
    t = i*period / 1000
    vl.set_xdata([t, t])
    return vl,


ani = FuncAnimation(fig, animate, frames=int(
    duration/(refreshPeriod/1000)), fargs=(vl, refreshPeriod), interval=refreshPeriod)
plt.show()
