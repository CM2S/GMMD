import matplotlib.pyplot as plt
import numpy as np

# Gráfico muito rudimentar. Se for para apresentar melhorar isto (usar funções de plot do GMMD).

files = [
    'Delete_later/Cell_new_list_times.txt',
    'Delete_later/Cell_check_intersection_times.txt',
    'Delete_later/Naive_new_list_times.txt',
    'Delete_later/Naive_check_intersection_times.txt'
]

labels = ['Cell- new list times', 'Cell- check intersection times', 'Cell-total times', 'Naive- new list times', 'Naive- check intersection times', 'Naive- total times']
colors = ['red', 'orange', 'green', 'blue']
colors = ['lightcoral', 'orange', 'red', 'lightblue', 'blue', 'darkblue']

plt.figure(figsize=(12, 7))

times=[]

for i, file_path in enumerate(files):
    with open(file_path, 'r') as file:
        line = file.read().strip()
    times.append([float(x) for x in line.split(', ')])


times_total_cell = np.array(times[0]) + np.array(times[1])
times_total_naive = np.array(times[2]) + np.array(times[3])
times_total_cell = np.cumsum(times_total_cell)
times_total_naive = np.cumsum(times_total_naive)

times = [times[0], times[1], times_total_cell, times[2], times[3], times_total_naive]

for i,values in enumerate(times):
    # Plot
    plt.plot( [v for v in range(1, len(values) + 1)], values, 
            #marker=markers[i], 
            linestyle='-', 
            linewidth=1.5, 
            #markersize=4,
            color=colors[i],
            label=labels[i],
            alpha=0.8)


# Labels and title
plt.xlabel('Step', fontsize=12)
plt.ylabel('Cell New List Time (seconds)', fontsize=12)
plt.title('Comparison of Cell New List Times Across Datasets', fontsize=14)

# Add legend
plt.legend(loc='best', fontsize=10)

# Grid for better readability
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.savefig('cell_naive_times_comparison.pdf', format='pdf', bbox_inches='tight', dpi=300)

# Display the plot
plt.tight_layout()
plt.show()