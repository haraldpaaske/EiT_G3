import matplotlib.pyplot as plt
import numpy as np


def plot_spider_chart(labels, values, title="Spider Diagram"):
    # Ensure values are in range [0,1] and close the loop
    values = np.append(values, values[0])
    labels = np.append(labels, labels[0])

    # Compute angles for each axis
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=True)

    # Create the radar chart
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='b', alpha=0.3)  # Fill area
    ax.plot(angles, values, color='b', linewidth=2)  # Outline

    # Set labels and format chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels[:-1], fontsize=10, va='bottom', position=(0.00001, 0.00001))
    ax.set_ylim(0, 50)
    ax.set_yticklabels([])  # Hide radial labels

    # Add title
    plt.title(title)

    # Show the plot
    plt.show()


# Aksel
labels = ["Ledelse", "Innorder seg", "produksjonsrettet", "Bidrar til trivsel", "Tar oppmerksomhet", "Stahet"]
values = [15, 28, 40, 44, 17, 7]
plot_spider_chart(labels, values, title="Aksel")

# Marius
values = [41, 5, 47, 43, 38, 9]
plot_spider_chart(labels, values, title="Marius")

# Harald
values = [29, 13, 41, 43, 28, 14]
plot_spider_chart(labels, values, title="Harald")

# Erlend
values = [15, 26, 43, 39, 18, 5]
plot_spider_chart(labels, values, title="Erlend")

# Callum
values = [40, 4, 47, 49, 46, 14]
plot_spider_chart(labels, values, title="Callum")

# Anders
values = [33, 7, 42, 42, 35, 13]
plot_spider_chart(labels, values, title="Anders")


aksel_ark = np.array([
    [2, 8, 5, 3, 7, 6],
    [2, 0, 0, 1, 0, 0],
    [6, 8, 8, 8, 8, 8],
    [7, 7, 8, 7, 8, 7],
    [3, 7, 6, 5, 8, 6],
    [1, 0, 0, 0, 0, 0]
])

marius_ark = np.array([
    [3, 5, 5, 3, 7, 3],
    [5, 0, 2, 5, 0, 0],
    [9, 9, 9, 9, 9, 9],
    [9, 9, 9, 9, 9, 9],
    [4, 7, 4, 4, 8, 6],
    [0, 3, 0, 0, 3, 3]
])

harald_ark = np.array([
    [3, 5, 4, 3, 4, 5],
    [5, 1, 1, 5, 1, 2],
    [6, 7, 7, 6, 7, 6],
    [7, 7, 7, 7, 8, 7],
    [3, 6, 6, 3, 7, 6],
    [1, 2, 5, 1, 4, 2]
])

erlend_ark = np.array([
    [3, 7, 5, 3, 7, 8],
    [6, 2, 4, 6, 1, 2],
    [5, 8, 6, 6, 8, 8],
    [8 ,7, 7, 6, 9, 7],
    [2, 7, 5, 2, 8, 6],
    [4, 2, 4, 4, 3, 2]
])

callum_ark = np.array([
    [4, 8, 6, 3, 7, 7],
    [4, 2, 4, 3, 2, 3],
    [8, 7, 7, 8, 7, 7],
    [7, 7, 6, 6, 7, 6],
    [3, 5, 3, 2, 7, 5],
    [1, 0, 3, 0, 2, 0]
])

anders_ark = np.array([
    [0, 8, 4, 0, 8, 4],
    [6, 0, 2, 6, 0, 0],
    [6, 8, 4, 6, 8, 4],
    [6, 6, 6, 4, 8, 6],
    [2, 6, 4, 2, 8, 6],
    [0, 2, 2, 0, 2, 6]
])


val = aksel_ark + marius_ark + harald_ark + erlend_ark + callum_ark + anders_ark
average = (aksel_ark + marius_ark + harald_ark + erlend_ark + callum_ark + anders_ark)/6
aaa = np.sum(average, axis=1)
print(aaa)

plot_spider_chart(labels, aaa, title="Gjennomsnitt")