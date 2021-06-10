from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math


class AGATable:
    """
    This class is used to convert .aga or .sga file produced by MBbackangle of MB-System
    to other format in order to better process the data.
    """
    _metadata = {}
    _tables = []
    _is_tot = False

    def __init__(self, file_path):
        """

        :param file_path: .sga or .aga file path.
        """
        assert file_path.endswith('.sga') or file_path.endswith('.aga'), 'Only .sga and .aga file are valid.'
        self._is_tot = file_path.endswith('_tot.sga') or file_path.endswith('_tot.aga')
        with open(file_path, 'r') as f:
            for i in range(5):
                f.readline()
            range_size = 8 if self._is_tot else 9
            for i in range(range_size):
                this_line = f.readline()
                key, value = this_line.split(':')
                key = key[3:]

                while not key.find(' ') == -1:
                    blank_index = key.find(' ')
                    old_part = key[blank_index: blank_index + 2]
                    new_part = key[blank_index + 1].upper()
                    key = key.replace(old_part, new_part)

                value = value.strip()
                try:
                    value = float(value)
                except ValueError:
                    pass

                self._metadata[key] = value

            this_line = f.readline()
            while True:
                table_metadata = {}
                _, value = this_line.split(':')
                table_metadata['TableNumber'] = int("".join(value.split()))

                this_line = f.readline()
                _, value = this_line.split(':')
                table_metadata['PingQuantity'] = int("".join(value.split()))

                this_line = f.readline()
                time = {'year': int(this_line[9: 13]), 'month': int(this_line[14: 16]),
                        'day': int(this_line[17: 19]), 'hour': int(this_line[20: 22]),
                        'minute': int(this_line[23: 25]), 'second': float(this_line[26: 35])}
                table_metadata['Time'] = time

                this_line = f.readline()
                _, value = this_line.split(':')
                table_metadata['AngleQuantity'] = int("".join(value.split()))

                table_data = []
                for i in range(table_metadata['AngleQuantity']):
                    this_line = f.readline()
                    this_line = " ".join(this_line.split())
                    angle, data, standard_deviation = this_line.split(' ')
                    table_data.append((float(angle), float(data), float(standard_deviation)))
                    # print(this_line)

                self._tables.append({'metadata': table_metadata, 'data': table_data})
                f.readline()
                f.readline()

                this_line = f.readline()
                if not this_line:
                    break

    def visualization(self, width=1.2):
        assert self._is_tot, 'Total dive file is valid'
        # print(self._tables[0]['data'])
        x_list = []
        y_list = []
        for data in self._tables[0]['data']:
            x_list.append(data[0])
            y_list.append(data[1])
        plt.bar(x_list, y_list, width=width)
        plt.show()

    def write_xls(self, file_path='../', unit='dB'):
        """
        Write the angular response file to xlsx.
        Only total dive file is valid.

        :param file_path:
        :param unit: Unit of sound pressure. 'Pa' or 'dB'
        :return:
        """
        assert self._is_tot, 'Total dive file is valid'
        assert unit == 'dB' or unit == 'Pa', 'Unit {} is invalid.'.format(unit)
        file_name = self._metadata['InputFile'] + '.xlsx'

        length = int(self._metadata['NumberOfAngleBins'])
        data_array = np.ones((length, 2))
        if unit == 'dB':
            for i, data in enumerate(self._tables[0]['data']):
                data_array[i] = data[0: 2]
        elif unit == 'Pa':
            for i, data in enumerate(self._tables[0]['data']):
                data_array[i][0] = data[0]
                data_pa = 0 if data[1] == 0 else math.pow(10, data[1] / 10)
                data_array[i][1] = data_pa

        frame = pd.DataFrame(data_array, columns=['angle', self._metadata['DataType']])
        frame.to_excel(file_path + file_name, index=False)
        return file_path + file_name


if __name__ == '__main__':
    file = 'D:/Study/mbsystem/mbbackangle/0158_20160830_025854_EX1607_MB.all.mb58_tot.aga'
    # file_path = 'D:/Study/mbsystem/mbbackangle/0158_20160830_025854_EX1607_MB.all.mb58.aga'
    a = AGATable(file)

    # a.visualization()
    a.write_xls(unit='Pa')
