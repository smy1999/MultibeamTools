from ConversionToolbox import AmplitudeGrazingAngle as aga
from CurveFittingToolbox import GSAB as cf
import pandas as pd

def test():
    """
    The standard process of loading data from .aga file
    which is produced by mbbackangle command
    and fit the data to GSAB model.
    """
    file_path = 'D:/Study/mbsystem/mbbackangle/test/0158_20160830_025854_EX1607_MB.all.mb58_tot.aga'
    table = aga.AGATable(file_path)
    xls_path = '../aga_file_fit_gsab/'
    xls_name = table.write_xls(xls_path)
    df = pd.read_excel(xls_name, engine='openpyxl')
    x = df['angle'].values
    y = df['beam amplitude'].values

    x = cf.xconvert(x, 'degree')
    x, y = cf.get_interval(x, y)
    cf.fitGSAB(x, y, if_save_img=True)


if __name__ == '__main__':
    test()