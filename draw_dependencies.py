import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # 1. Загрузка данных
    time_arr = np.load('time_series.npy')
    score_arr = np.load('gallery_score.npy') 

    plt.plot(time_arr, score_arr)
    plt.show()

    print("qqqq")