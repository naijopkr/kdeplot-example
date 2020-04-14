import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def main():
    dataset = np.random.randn(25)

    plt.subplot(2,1,1)
    sns.rugplot(dataset)

    x_min = dataset.min() - 2
    x_max = dataset.max() + 2

    x_axis = np.linspace(x_min, x_max, 100)

    bandwidth = ((4 * dataset.std()**5) / (3 * len(dataset))) ** 0.2

    kernel_list = []
    for data_point in dataset:
        kernel = stats.norm(data_point, bandwidth).pdf(x_axis)
        kernel_list.append(kernel)

        kernel = kernel / kernel.max()
        kernel = kernel * 0.4
        plt.plot(x_axis, kernel, color='grey', alpha=0.5)

    sum_of_kde = np.sum(kernel_list, axis=0)

    plt.subplot(2,1,2)
    plt.plot(x_axis, sum_of_kde, color='indianred')

    plt.savefig('output.pdf')

if __name__ == '__main__':
    main()
