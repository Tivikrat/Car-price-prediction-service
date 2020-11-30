import getopt
import sys

import pandas


def prepare(data_path, output_path, ignore_columns=None):
    data = pandas.read_csv(data_path, index_col=0)
    data = data.drop(columns=set(ignore_columns).intersection(data.columns))
    first_columns = [column
                     for column in ['insurance_price', 'registration_year', 'engine_capacity', 'model']
                     if column not in ignore_columns]
    data = data[first_columns + [column for column in data.columns if column not in first_columns]]
    data.to_csv(output_path)


blacklist = ['engine_capacity', 'zipcode']


def main(argv):
    input_file = None
    output_file = None
    help_message = ' -i <input_data_path> -o <output_data_path>'
    try:
        opts, args = getopt.getopt(argv, "hi:o:", [])
    except getopt.GetoptError:
        print(help_message)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(help_message)
            sys.exit()
        elif opt == "-i":
            input_file = arg
        elif opt == "-o":
            output_file = arg
    if input_file is not None and output_file is not None:
        prepare(input_file, output_file, ignore_columns=blacklist)
    else:
        if input_file is None:
            print("Wrong -i")
        if output_file is None:
            print("Wrong -o")
        print(help_message)


if __name__ == "__main__":
    main(sys.argv[1:])
