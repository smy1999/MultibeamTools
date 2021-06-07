import pandas as pd
from ConversionToolbox import AmplitudeGrazingAngle as AGA
import math

def test_import():
    file_path = 'C:/Users/86598/Downloads/1.txt'
    file_path2 = 'C:/Users/86598/Downloads/0158_20160830_025854_EX1607_MB.all.mb58.sga'
    file_path3 = 'C:/Users/86598/Downloads/0158_20160830_025854_EX1607_MB.all.mb58.aga'
    file_path3 = 'sdf'
    a = AGA.AGATable(file_path2)
    b = AGA.AGATable(file_path2)
    c = AGA.AGATable(file_path3)

def test_rdxls():
    path = 'C:/Users/86598/PycharmProjects/MultibeamTools/'
    file_name = '0158_20160830_025854_EX1607_MB.all.mb58.xlsx'
    file = path + file_name
    data = pd.read_excel(file, sheet_name='Sheet1', engine='openpyxl')
    print(data)
    print(type(data))

def test():
    print(math.pow(2,3))



if __name__ == '__main__':
    test()
