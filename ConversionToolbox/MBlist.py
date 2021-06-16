import numpy as np
import pandas as pd

FORMAT = {'B': 'amplitude',
          'G': 'flat bottom grazing angle (degrees)',
          'g': 'grazing angle using seafloor slope (degrees)',
          'T': 'a time string (yyyy/mm/dd/hh/mm/ss)',
          'V': 'ping interval (decimal seconds)',
          'X': 'longitude (decimal degrees)',
          'Y': 'latitude (decimal degrees)',
          '#': 'beam or pixel number',
          '.d': 'Beam depression angle',
          '.A': 'Amplitude (backscatter) in dB'
          }

TYPE = {'int': ['#'],
        'float': ['B', 'G', 'g', 'V', 'X', 'Y', '.d', '.A'],
        'String': ['T']}


class SwathDataTable:

    def __init__(self, file_path, command):
        command = ''.join(command.split())
        command = command.split('-')[1:]
        self.metadata = {}
        for data in command:
            self.metadata[data[0]] = data[1:]

        delimiter = self.metadata['G']

        self.output_format = self.metadata['O']
        self.output_format = list(self.output_format)

        length = len(self.output_format)
        for i in range(length):
            if self.output_format[length - i - 1] == '.':
                item = self.output_format[length - i]
                self.output_format[length - i - 1: length - i + 1] = ""
                self.output_format.insert(length - i - 1, '.' + item)

        self.data = []

        with open(file_path, 'r') as f:
            this_line = f.readline()
            while True:
                this_line = ''.join(this_line.split())
                this_line = this_line.split(delimiter)
                this_line = list(filter(None, this_line))
                for i in range(len(this_line)):
                    if self.output_format[i] in TYPE['int']:
                        this_line[i] = int(this_line[i])
                    elif self.output_format[i] in TYPE['float']:
                        this_line[i] = float(this_line[i])
                self.data.append(this_line)
                this_line = f.readline()
                if not this_line:
                    break

    def write_xls(self, file_path='../'):
        file_name = self.metadata['I'] + '_mblist.xlsx'
        data_array = np.array(self.data)

        columns = [FORMAT[key] for key in self.output_format]
        frame = pd.DataFrame(data_array, columns=columns)

        int_key = [FORMAT[key] for key in TYPE['int']]
        float_key = [FORMAT[key] for key in TYPE['float']]
        frame[int_key] = frame[int_key].apply(pd.to_numeric)
        frame[float_key] = frame[float_key].apply(pd.to_numeric)

        frame.to_excel(file_path + file_name, index=False)
        return file_path + file_name


if __name__ == '__main__':
    file = 'D:/Study/mbsystem/mblist/testcon1'
    comm = 'mblist -I0158_20160830_025854_EX1607_MB.all.mb58 -OXYTBGgV#.A.d -MA -X testcon -G*'
    SDT = SwathDataTable(file, comm)
    # print(SDT.data)
    SDT.write_xls()

