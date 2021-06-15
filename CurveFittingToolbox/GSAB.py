import math
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

default_bounds = ([0.001, 0.0, 3.162277660168379e-06, 0, 3.162277660168379e-06, 0.0],
                  [0.1, 0.17453292519943295, 0.03162277660168379, 2.5, 0.01, 0.6981317007977318])


def fitGSAB(x, y, bounds=default_bounds, maxfev=10000, fmt1='b-', fmt2='r-',
            unit_y_in='dB', unit_x_in='rad', parameters_pattern='standard',
            if_plot=True, if_print_parameters=True, if_save_img=False):
    assert (-math.pi / 2 <= x).all() and (x <= math.pi / 2).all(), \
        'x is out of range([-pi / 2, pi / 2]).'
    assert unit_y_in == 'dB' or unit_y_in == 'Pa', \
        'Input y unit {} is invalid.'.format(unit_y_in)
    assert unit_x_in == 'degree' or unit_x_in == 'rad', \
        'Input x unit {} is invalid.'.format(unit_x_in)
    assert parameters_pattern == 'standard' or parameters_pattern == 'original', \
        'Parameters pattern {} is invalid.'.format(parameters_pattern)

    if unit_x_in == 'degree':
        x = xconvert(x)
    if unit_y_in == 'dB':
        y = yconvert(y)

    popt, pcov = curve_fit(GSAB, x, y, bounds=bounds, maxfev=maxfev)

    parameters = {}
    for i, key in enumerate(['A', 'B', 'C', 'D', 'E', 'F']):
        parameters[key] = popt[i]
    if parameters_pattern == 'standard':
        original2standard(parameters)

    if if_plot or if_save_img:
        plt.plot(xconvert(x, unit_in='rad'), yconvert(y, unit_in='Pa'), fmt1)
        plt.plot(xconvert(x, unit_in='rad'), yconvert(GSAB(x, *popt), unit_in='Pa'), fmt2)
        plt.xlabel('Incident Angle(°)')
        plt.ylabel('Backscatter(dB)')
        if if_save_img:
            plt.savefig('GSAB_fitting.png')
        if if_plot:
            plt.show()

    if if_print_parameters:
        print_parameters(parameters)

    return parameters


def standard2original(parameters):
    """
    Convert the unit of 6 parameters.
    Parameters 'A', 'C', 'E' convert from standard 'Pa' to 'dB'.
    Parameters 'B', 'F' convert from radian to angle.
    :param parameters: A dictionary contains 6 parameters.
    :return:
    """
    parameters['A'] = math.pow(10, parameters['A'] / 10)
    parameters['B'] = parameters['B'] * math.pi / 180
    parameters['C'] = math.pow(10, parameters['C'] / 10)
    parameters['E'] = math.pow(10, parameters['E'] / 10)
    parameters['F'] = parameters['F'] * math.pi / 180


def original2standard(parameters):
    """
    Convert the unit of 6 parameters.
    Parameters 'A', 'C', 'E' convert from 'dB' to 'Pa'.
    Parameters 'B', 'F' convert from angle to radian.
    :param parameters: A dictionary contains 6 parameters.
    :return:
    """
    parameters['A'] = 10 * math.log10(parameters['A'])
    parameters['B'] = parameters['B'] * 180 / math.pi
    parameters['C'] = 10 * math.log10(parameters['C'])
    parameters['E'] = 10 * math.log10(parameters['E'])
    parameters['F'] = parameters['F'] * 180 / math.pi


def test():
    x = np.linspace(-math.pi / 3, math.pi / 3, 60)
    y = GSAB(x, 1, 2, 3, 4, 5, 6)
    np.random.seed(2)
    y_noise = 0.2 * np.random.normal(size=x.size)
    y += y_noise
    fitGSAB(x, y)


def yconvert(y, unit_in='dB'):
    assert unit_in == 'dB' or unit_in == 'Pa', 'Input unit {} is invalid.'.format(unit_in)
    if unit_in == 'dB':
        return np.power(10, y / 10)
    elif unit_in == 'Pa':
        return 10 * np.log10(y)


def xconvert(x, unit_in='degree'):
    assert unit_in == 'degree' or unit_in == 'rad', 'Input unit {} is invalid.'.format(unit_in)
    if unit_in == 'degree':
        return x / 180 * math.pi
    elif unit_in == 'rad':
        return x * 180 / math.pi


def GSAB(x, A, B, C, D, E, F):
    p1 = A * np.exp(-np.power(x, 2) / (2 * math.pow(B, 2)))
    p2 = C * np.power(np.cos(x), D)
    p3 = E * np.exp(-np.power(x, 2) / (2 * math.pow(F, 2)))
    ret = p1 + p2 + p3
    return ret


def print_parameters(parameters):
    for key in ['A', 'B', 'C', 'D', 'E', 'F']:
        print(key, '=', parameters[key])


def get_interval(x, y):
    index = np.where(y < 0)
    return x[index], y[index]


if __name__ == '__main__':
    test()
