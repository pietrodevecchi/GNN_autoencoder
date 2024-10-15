import matplotlib.pyplot as plt

number_test_samples=20
samples = 10
errors = []
with open('from_hpc/paper_results/pool_0_5_batch_25/test_errors.txt') as f:
    next(f)  # Skip the first line
    for line in f:
        _, value = line.strip().split(',')
        errors.append(value)  # Add the value to the list

j=0
average=0
averages=[]
for i in range(len(errors)):
    if j<samples:
        average+=float(errors[i])
    j+=1
    if i%number_test_samples==0:
        print(f'Average error for {i/number_test_samples}: {average/samples}')
        averages.append(average/samples)
        average=0
        j=0

plt.plot(range(1, len(averages)+1), averages)
plt.xlabel('Epoch')
plt.ylabel('Average Error')
plt.title('Validation losses on 10 trajectories set')
plt.yscale('log')  # Set the y-axis to a logarithmic scale
plt.xticks(range(1, len(averages)+1))

plt.show()