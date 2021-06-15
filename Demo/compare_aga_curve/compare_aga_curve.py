from ConversionToolbox import AmplitudeGrazingAngle as aga
from CurveFittingToolbox import GSAB as cf
import pandas as pd
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt


default_bounds = ([0.001, 0.0, 3.162277660168379e-06, 0, 3.162277660168379e-06, 0.0],
                  [0.1, 0.17453292519943295, 0.03162277660168379, 2.5, 0.01, 0.6981317007977318])


def test():
    """
    Compare amplitude versus angle curve with and without slope data.
    """
    file_path = 'D:/Study/mbsystem/mbbackangle/compare/'
    file_name1 = '0158_20160830_025854_EX1607_MB.all.mb58_tot.aga'
    file_name2 = '0158_20160830_025854_EX1607_MB_slope.all.mb58_tot.aga'
    table1 = aga.AGATable(file_path + file_name1)
    table2 = aga.AGATable(file_path + file_name2)
    xls_path = '../compare_aga_curve/'
    xls_name1 = table1.write_xls(xls_path)
    xls_name2 = table2.write_xls(xls_path)
    df1 = pd.read_excel(xls_name1, engine='openpyxl')
    x1 = df1['angle'].values
    y1 = df1['beam amplitude'].values
    df2 = pd.read_excel(xls_name2, engine='openpyxl')
    x2 = df2['angle'].values
    y2 = df2['beam amplitude'].values

    x1 = cf.xconvert(x1, 'degree')
    x2 = cf.xconvert(x2, 'degree')

    x1, y1 = cf.get_interval(x1, y1)
    x2, y2 = cf.get_interval(x2, y2)
    # cf.fitGSAB(x1, y1, fmt1='r--', fmt2='b-', if_plot=False)
    # cf.fitGSAB(x2, y2, fmt1='g--', fmt2='c-')
    fitGSAB(x1, y1, x2, y2, if_save_img=True)


def fitGSAB(x1, y1, x2, y2, bounds=default_bounds, maxfev=10000,
            fmt1='b--', fmt2='r-', fmt3='g--', fmt4='c-',
            unit_y_in='dB', unit_x_in='rad', if_plot=True, if_save_img=False):

    if unit_x_in == 'degree':
        x1 = cf.xconvert(x1)
        x2 = cf.xconvert(x2)
    if unit_y_in == 'dB':
        y1 = cf.yconvert(y1)
        y2 = cf.yconvert(y2)

    popt1, pcov1 = curve_fit(cf.GSAB, x1, y1, bounds=bounds, maxfev=maxfev)
    popt2, pcov2 = curve_fit(cf.GSAB, x2, y2, bounds=bounds, maxfev=maxfev)

    if if_plot or if_save_img:
        plt.plot(cf.xconvert(x1, unit_in='rad'), cf.yconvert(y1, unit_in='Pa'), fmt1)
        plt.plot(cf.xconvert(x1, unit_in='rad'), cf.yconvert(cf.GSAB(x1, *popt1), unit_in='Pa'), fmt2)
        plt.plot(cf.xconvert(x2, unit_in='rad'), cf.yconvert(y2, unit_in='Pa'), fmt3)
        plt.plot(cf.xconvert(x2, unit_in='rad'), cf.yconvert(cf.GSAB(x2, *popt2), unit_in='Pa'), fmt4)
        plt.xlabel('Incident Angle(°)')
        plt.ylabel('Backscatter(dB)')
        if if_save_img:
            plt.savefig('GSAB_fittin.png')
        if if_plot:
            plt.show()


if __name__ == '__main__':
    test()
