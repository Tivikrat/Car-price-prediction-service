import getopt
import sys

import numpy
import pandas


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = numpy.array(y_true), numpy.array(y_pred)
    return numpy.mean(numpy.abs((y_true - y_pred) / y_true)) * 100


def check(input_valid_data_path, input_predicted_targets_path, target: str):
    validating_data = pandas.read_csv(input_valid_data_path, index_col=0)
    predicted_data_frame = pandas.read_csv(input_predicted_targets_path, index_col=0)
    original_prices = validating_data[target].values
    predicted_prices = predicted_data_frame["Predicted"].values
    error = mean_absolute_percentage_error(original_prices, predicted_prices)
    print(f'{error}% error validation')


def main(argv):
    input_valid_file = None
    predicted_targets_file = None
    target = None
    help_message = ' -v <input_validating_data_path>' \
                   ' -p <input_validating_predicted_targets_path>' \
                   ' -g <target_column_name>'
    try:
        opts, args = getopt.getopt(argv, "hv:p:g:", [])
    except getopt.GetoptError:
        print(help_message)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(help_message)
            sys.exit()
        elif opt == "-v":
            input_valid_file = arg
        elif opt == "-p":
            predicted_targets_file = arg
        elif opt == "-g":
            target = arg
    if input_valid_file is not None and predicted_targets_file is not None and target is not None:
        check(input_valid_file, predicted_targets_file, target)
    else:
        if input_valid_file is None:
            print("Wrong -v")
        if predicted_targets_file is None:
            print("Wrong -p")
        if target is None:
            print("Wrong -g")
        print(help_message)


if __name__ == "__main__":
    main(sys.argv[1:])
