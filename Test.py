import pandas as pd
from ConversionToolbox import AmplitudeGrazingAngle as AGA
from CurveFittingToolbox import GSAB as cf


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


def test():
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


if __name__ == '__main__':
    test()
