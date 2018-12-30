import numpy as np

def sine_value(amplitude, arg, gap):
    return amplitude * np.sin(arg + gap)

if __name__ == '__main__':
    num_plot = 100
    num_amp = 10
    num_gap = 8
    plot_interval = 0.1
    # num_data = num_amp * num_gap
    # shape_data = (num_plot, num_data)

    count = 0
    for i_amp in range(num_amp):
        amp = float(i_amp +1) / float(num_amp)
        for i_gap in range(num_gap):
            gap = float(i_gap +1) * 2.0 * np.pi / float(num_gap)
            data = np.array([])
            #.. store sine value
            for i_plot in range(num_plot):
                plot_arg = float(i_plot +1) * plot_interval * np.pi
                data = np.append(data, sine_value(amp, plot_arg, gap))

            #.. write stored data
            column = np.atleast_2d(data).T
            plot_data = np.empty((num_plot, 0), float)
            plot_data = np.append(plot_data, column, axis=1) ## input data
            plot_data = np.append(plot_data, column, axis=1) ## target data
            filename = 'sine_{}.dat'.format(count)
            np.savetxt(filename, plot_data, comments='#')

            #.. add comment
            with open(filename) as f:
                l = f.readlines()
            ## add coment at the first line
            l.insert(0, '#.. amplitude = {}, gap = {}pi\n'.format(amp, \
                         float(i_gap +1) * 2.0 / float(num_gap)))
            with open(filename, mode='w') as f:
                f.writelines(l)

            count += 1
