import pandas as pd
from ConversionToolbox import AmplitudeGrazingAngle as AGA
from CurveFittingToolbox import GSAB as cf
import time


def test_import():
    file_path2 = 'C:/Users/86598/Downloads/0158_20160830_025854_EX1607_MB.all.mb58.sga'
    file_path3 = 'C:/Users/86598/Downloads/0158_20160830_025854_EX1607_MB.all.mb58.aga'
    c = AGA.AGATable(file_path3)


def test_rdxls():
    path = 'C:/Users/86598/PycharmProjects/MultibeamTools/'
    file_name = '0158_20160830_025854_EX1607_MB.all.mb58.xlsx'
    file = path + file_name
    data = pd.read_excel(file, sheet_name='Sheet1', engine='openpyxl')
    print(data)
    print(type(data))


def test_unit():
    sup = {'A': -30,
           'B': 0,
           'C': -55,
           'D': 0,
           'E': -55,
           'F': 0}
    inf = {'A': -10,
           'B': 10,
           'C': -15,
           'D': 2.5,
           'E': -20,
           'F': 40}
    cf.standard2original(sup)
    cf.standard2original(inf)
    list1 = []
    list2 = []
    for i, key in enumerate(['A', 'B', 'C', 'D', 'E', 'F']):
        list1.append(sup[key])
        list2.append(inf[key])

    print(list1)
    print(list2)


def test_write_img():
    import numpy as np
    import matplotlib.pyplot as plt

    image = np.random.randn(100, 100)
    plt.imsave('new_1.png', image)


def test_new_curve():
    from ConversionToolbox import AmplitudeGrazingAngle as aga
    from CurveFittingToolbox import GSAB as cf
    import pandas as pd
    file_path = 'D:/Study/mbsystem/mbbackangle/compare/0158_20160830_025854_EX1607_MB_slope.all.mb58_tot.aga'
    table = aga.AGATable(file_path)
    xls_path = '../'
    xls_name = table.write_xls(xls_path)
    df = pd.read_excel(xls_name, engine='openpyxl')
    x = df['angle'].values
    y = df['beam amplitude'].values

    x = cf.xconvert(x, 'degree')
    x, y = cf.get_interval(x, y)
    cf.fitGSAB(x, y, if_save_img=True)


def test_try(type):
    start = time.time()
    a = ['0', '0.12', '5.0', '5.1', '5.12.015']
    for i in range(10):
        a = a + a

    if type:
        for index, i in enumerate(a):
            try:
                a[index] = float(i)
                try:
                    a[index] = int(i)
                except ValueError:
                    pass
            except ValueError:
                pass
    else:
        for index in range(len(a)):
            try:
                a[index] = float(a[index])
                try:
                    a[index] = int(a[index])
                except ValueError:
                    pass
            except ValueError:
                pass
    end = time.time()
    return end - start


def test_time():
    time1 = 0
    time2 = 0
    for i in range(100):
        time1 += test_try(True)
        time2 += test_try(False)
    print(time1)
    print(time2)


def test_dtype():
    import numpy as np
    a = ['a', 1, 0.5]
    x = np.array(a)
    x[1] = x[1].astype(int)
    print(type(x[1]))
    print(x)


def test_split():
    path = 'asdf/asdfee/asdfasdf'
    path = path.split('/')
    print(path)
    print(path[-1])


def test():
    dir = {1:'5',2:'6',3:'7'}
    list = [1,2,3]
    print([dir[key] for key in list])



if __name__ == '__main__':
    test()
