from ConversionToolbox import MBlist as ml


def test():
    file = 'D:/Study/mbsystem/mblist/testcon'
    comm = 'mblist -I0158_20160830_025854_EX1607_MB.all.mb58 -OXYTBGgV#.A.d -MA -X testcon -G*'
    SDT = ml.SwathDataTable(file, comm)
    SDT.write_xls(file_path='../mblist_file_to_xlsx/')


if __name__ == '__main__':
    test()