import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # 1. Загрузка данных
    time_arr = np.load(f'frequently_results/time_series.npy')
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
        current = np.load(f'frequently_results/gallery_score_w{w}.npy') 
        plt.plot(time_arr, current, label=f'w{w}')

    plt.legend()
    plt.show()

    mean_gallery_score = []

    for w in range(10, 100, 2):
        current = np.load(f'frequently_results/gallery_score_w{w}.npy')
        mean_gallery_score.append(current[int(len(current) / 2):].mean()) 
        
    plt.plot(range(10, 100, 2), mean_gallery_score, '-o', label=f'mean')

    plt.legend()
    plt.show()


    for w in [90]: 
        for a in [0.05, 0.15, 0.3]:
            for b in [0.05, 0.25, 0.5]:
                current = np.load(f'm_ellipse_results/gallery_score_w{w}_a{a}_b{b}.npy')
                plt.plot(time_arr, current, label=f'w{w}_a{a}_b{b}')
    
    plt.title('ellipse params')

    plt.legend()
    plt.show()

    print("qqqq")