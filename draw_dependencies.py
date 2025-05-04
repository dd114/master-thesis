import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':



    n = 30

    window = 20

    time_arr = np.load(f'gallery_score/time_series.npy')

    for w in range(10, 101, 10):
        current = np.load(f'gallery_score/gallery_score_w{w}.npy') 
        current = np.convolve(current, np.ones(window)/window, 'same')

        plt.title("Метрика жизни волны")
        plt.xlabel("Время")
        plt.ylabel("% энергии у границы")
        plt.plot(time_arr, current, label=f'w{w}')

    plt.legend()
    plt.show()

    mean_gallery_score = []

    for w in range(10, 101, 5):
        current = np.load(f'gallery_score/gallery_score_w{w}.npy')
        mean_gallery_score.append(current[int(len(current) / 4 * 3):].mean()) 

    plt.title("Средняя метрика жизни волны")
    plt.xlabel("Частота")
    plt.ylabel("% энергии у границы")   
    plt.plot(range(10, 101, 5), mean_gallery_score, '-o')

    # plt.legend()
    plt.show()

    window = 30

    for w in [90]: 
        for a in [0.05, 0.15, 0.3]:
            for b in [0.05, 0.25, 0.5]:
                current = np.load(f'm_ellipse_results/gallery_score_w{w}_a{a}_b{b}.npy')
                current = np.convolve(current, np.ones(window)/window, 'same')

                plt.plot(time_arr, current, label=f'p1 = {a} | p2 = {b}')
    
    plt.title(f'Метрика жизни волны, вогнутый эллипс | w = {w}')
    plt.xlabel("Время")
    plt.ylabel("% энергии у границы")  
    plt.legend()
    plt.show()

    for w in [90]: 
        for a in [0.05, 0.15, 0.3]:
            for b in [0.05, 0.25, 0.5]:
                current = np.load(f'p_ellipse_results/gallery_score_w{w}_a{a}_b{b}.npy')
                current = np.convolve(current, np.ones(window)/window, 'same')

                plt.plot(time_arr, current, label=f'p1 = {a} | p2 = {b}')
    
    plt.title(f'Метрика жизни волны, выпуклый эллипс | w = {w}')
    plt.xlabel("Время")
    plt.ylabel("% энергии у границы")  
    plt.legend()
    plt.show()


    fig = plt.figure(figsize=(10, 7))

    ax1 = fig.add_subplot(111)
    # ax2 = fig.add_subplot(122)

    window = 50  # Размер окна для скользящего среднего.

    for w in range(10, 102, 10):
        time_arr = np.load(f'leaking_out_circle_score/time_series.npy')
        current = np.load(f'leaking_out_circle_score/leaking_out_circle_score_w{w}.npy')

        # Применяем скользящее среднее.
        current = np.convolve(current, np.ones(window)/window, 'same')

        ax1.plot(time_arr, current, '.', label=f'w{w}')
        

        # time_arr = np.load(f'old/leaking_out_circle_score__0.1r2/time_series.npy')
        # current = np.load(f'old/leaking_out_circle_score__0.1r2/leaking_out_circle_score_w{w}.npy')

        # # Применяем скользящее среднее.
        # current = np.convolve(current, np.ones(window)/window, 'same')

        # ax2.plot(time_arr, current, '.', label=f'w{w}')
    
    plt.title('Соотношение количества энергии | волновод / общее_количество_энергии')
    # plt.title('Соотношение количества энергии | круг / общее_количество_энергии')
    plt.xlabel("Время")
    plt.ylabel("% энергии в волноводе")
    # plt.ylabel("% энергии в круге")
    plt.legend()
    plt.show()


    window = 65

    time_arr = np.load(f'leaking_out_circle_score/time_series.npy')
    current1 = np.load(f'leaking_out_circle_score/leaking_out_circle_score_w10.npy')
    current2 = np.load(f'leaking_out_circle_score/leaking_out_circle_score_w60.npy')

    factor = current2 / current1
    factor = np.convolve(factor, np.ones(window)/window, 'same')

    plt.title('Соотношение количества энергии | Сигнал_w60 / шум_w10')
    plt.xlabel("Время")
    plt.ylabel("Сигнал / шум")
    # plt.legend()
    
    plt.plot(time_arr, factor, ".")
    plt.show()
