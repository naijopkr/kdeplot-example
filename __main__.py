import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def main():
    dataset = np.random.randn(25)
    print(dataset)

    sns.rugplot(dataset)

    x_min = dataset.min() - 2
    x_max = dataset.max() + 2
    print(x_min)
    print(x_max)

    x_axis = np.linspace(x_min, x_max, 100)
    print(x_axis)

    bandwidth = ((4 * dataset.std()**5) / (3 * len(dataset))) ** 0.2
    print(bandwidth)

    kernel_list = []

    for data_point in dataset:
        kernel = stats.norm(data_point, bandwidth).pdf(x_axis)
        kernel_list.append(kernel)

        kernel = kernel / kernel.max()
        kernel = kernel * 0.4
        plt.plot(x_axis, kernel, color='grey', alpha=0.5)

    sum_of_kde = np.sum(kernel_list, axis=0)
    fig = plt.plot(x_axis, sum_of_kde, color='indianred')

    plt.savefig('output.pdf')

if __name__ == '__main__':
    main()
