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

    window = 20

    for w in range(10, 100, 10):
        current = np.load(f'frequently_results/gallery_score_w{w}.npy') 
        current = np.convolve(current, np.ones(window)/window, 'same')

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

    window = 10

    for w in [90]: 
        for a in [0.05, 0.15, 0.3]:
            for b in [0.05, 0.25, 0.5]:
                current = np.load(f'm_ellipse_results/gallery_score_w{w}_a{a}_b{b}.npy')
                current = np.convolve(current, np.ones(window)/window, 'same')

                plt.plot(time_arr, current, label=f'w{w}_a{a}_b{b}')
    
    plt.title('ellipse params')

    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(10, 10))

    ax1 = fig.add_subplot(111)
    # ax2 = fig.add_subplot(122)

    window = 50  # Размер окна для скользящего среднего.

    for w in range(10, 102, 10):
        time_arr = np.load(f'leaking_out_circle_score/time_series.npy')
        current = np.load(f'leaking_out_circle_score/leaking_out_circle_score_w{w}.npy')

        # Применяем скользящее среднее.
        current = np.convolve(current, np.ones(window)/window, 'same')

        ax1.plot(time_arr, current, '.', label=f'w{w}')
        

        time_arr = np.load(f'old/leaking_out_circle_score__0.1r2/time_series.npy')
        current = np.load(f'old/leaking_out_circle_score__0.1r2/leaking_out_circle_score_w{w}.npy')

        # Применяем скользящее среднее.
        current = np.convolve(current, np.ones(window)/window, 'same')

        # ax2.plot(time_arr, current, '.', label=f'w{w}')
    
    plt.title('leaking')

    plt.legend()
    plt.show()

