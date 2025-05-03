import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # 1. Загрузка данных
    # time_arr = np.load('results/time_series.npy')
    # score_arrs = [0] * 41
    # wave_arrs = [0] * 41
    # my_iter = 0
    # for i in range(10, 82, 2):
    #     print(i)
    #     score_arrs[my_iter] = np.load(f'results/gallery_score_w{i}.npy') 
    #     wave_arrs[my_iter] = i
        
    #     plt.plot(score_arrs[my_iter], label=f'wave {wave_arrs[my_iter]}')   

    #     my_iter += 1

    # plt.plot(time_arr, score_arrs[5], label=wave_arrs[5])


    # plt.title(f'wave = {wave_arrs[5]} ')
    # plt.show()

    n = 30
    print(np.load(f'frequently_results/gallery_score_w{n}.npy'))
    # plt.plot(np.load(f'frequently_results/time_series.npy'), np.load(f'frequently_results/gallery_score_w{n}.npy'))

    for w in range(10, 100, 10):
        plt.plot(np.load(f'frequently_results/time_series.npy'), np.load(f'frequently_results/gallery_score_w{w}.npy'))
        
    plt.show()

    print("qqqq")