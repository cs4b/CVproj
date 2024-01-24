import matplotlib.pyplot as plt
# Generate example data
x_values = list(range(400, 4001, 400))
y_values = [0.3029, 0.1465, 0.1021, 0.0748, 0.0557, 0.0317, 0.0244, 0.0191, 0.0151]
y2_values= [0.2368, 0.116, 0.0752, 0.0505, 0.0354, 0.0259, 0.0195, 0.0149, 0.0117, 0.0094]
# Plotting the graph
plt.plot(x_values[:len(y2_values)], y2_values, marker='o', linestyle='-', color='orange', label='Loss')

for x_value in x_values[:len(y2_values)]:
    plt.axvline(x=x_value, color='black', linestyle='--', alpha=0.2)



plt.title('Loss vs. Iteration')
plt.xlabel('Iteration')
plt.ylabel('Loss')
#plt.grid(True)
plt.legend()
plt.show()