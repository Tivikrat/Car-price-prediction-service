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
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print(' -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(' -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-o", "--ofile"):
            output_file = arg
    if input_file is not None and output_file is not None:
        prepare(input_file, output_file, ignore_columns=blacklist)


if __name__ == "__main__":
    main(sys.argv[1:])
