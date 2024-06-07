import matplotlib.animation as animation
import matplotlib.pyplot as plt
from background_function import generate_2d_background


def animate(solutions, func_name, n):
    fig, ax = plt.subplots()
    scat = ax.scatter(solutions[0, 0], solutions[0, 1], c="b", s=5)
    ax.set(
        xlim=[-n, n],
        ylim=[-n, n],
    )
    bg_function = generate_2d_background(func_name, n)
    plt.imshow(
        bg_function,
        extent=[
            -bg_function.shape[1] / 2.0,
            bg_function.shape[1] / 2.0,
            -bg_function.shape[0] / 2.0,
            bg_function.shape[0] / 2.0,
        ],
        cmap="viridis",
    )

    def update(frame: int):
        scat.set_offsets(solutions[frame])
        return scat

    _ = animation.FuncAnimation(fig=fig, func=update, frames=100, interval=100)

    plt.show()
