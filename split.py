import getopt
import sys

import pandas


def split(input_data_path, output_train_data_path, output_valid_data_path, target: str):
    data = pandas.read_csv(input_data_path, index_col=0)
    data = data.sort_values(target)
    validating = data.iloc[lambda x: x.index % 5 == 0]
    validating.to_csv(output_valid_data_path)
    training = data.iloc[lambda x: x.index % 5 != 0]
    training.to_csv(output_train_data_path)


def main(argv):
    input_file = None
    train_file = None
    valid_file = None
    target = None
    help_message = ' -i <input_data_path> -t <output_training_data_path>' \
                   ' -v <output_validating_data_path> -g <target_column_name>'
    try:
        opts, args = getopt.getopt(argv, "hi:t:v:g:", [])
    except getopt.GetoptError:
        print(help_message)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(help_message)
            sys.exit()
        elif opt == "-i":
            input_file = arg
        elif opt == "-t":
            train_file = arg
        elif opt == "-v":
            valid_file = arg
        elif opt == "-g":
            target = arg
    if input_file is not None and train_file is not None and valid_file is not None and target is not None:
        split(input_file, train_file, valid_file, target)
    else:
        if input_file is None:
            print("Wrong -i")
        if train_file is None:
            print("Wrong -t")
        if valid_file is None:
            print("Wrong -v")
        if target is None:
            print("Wrong -g")
        print(help_message)


if __name__ == "__main__":
    main(sys.argv[1:])
