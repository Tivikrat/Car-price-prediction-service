import math
from collections import defaultdict, Counter
from time import time
import getopt
import sys
import pandas
import numpy as np

from pandas import notnull


def sign(x):
    return x and (-1 if x < 0 else 1)


class Predictor:
    def __init__(self):
        self.additions = None
        self.multipliers = None
        self.rows = None

    @staticmethod
    def read_data(data_path):
        dataFrame = pandas.read_csv(data_path, index_col=0)
        return dataFrame.where(notnull(dataFrame), None)

    def predict_target_value(self, car):
        target = 0.0
        multiplier = 1.0
        for column in car:
            factor = car[column]
            if column in self.additions and factor in self.additions[column]:
                target += self.additions[column][factor]
            if column in self.multipliers and factor in self.multipliers[column]:
                multiplier *= self.multipliers[column][factor]
        return target * multiplier

    def predict_target_values(self, data=None):
        if data is None:
            return [self.predict_target_value(instance) for instance in self.rows]
        else:
            return [self.predict_target_value({column: factor for column, factor in instance.items()}) for
                    index, instance in data.iterrows()]


class Entity:
    def __init__(self, target_value, predicted_value, multiplier):
        self.target_value = target_value
        self.predicted_value = predicted_value
        self.multiplier = multiplier

    def get_difference(self):
        return (self.target_value - self.predicted_value) / self.multiplier

    def get_coefficient(self):
        return self.target_value / self.predicted_value

    def get_mape_addition_derivative(self):
        return self.multiplier * sign(self.get_difference()) / self.target_value

    def get_mape_multiplier_derivative(self, current_factor_multiplier):
        return self.predicted_value / current_factor_multiplier * sign(self.get_difference()) / self.target_value


class TrainingPredictor(Predictor):
    def __init__(self, data_path, target_column, rarity_ignore: float = 0.0):
        super().__init__()
        print('Extracting data...', flush=True)
        data = self.read_data(data_path)
        self.ignore = {'additions': [], 'multipliers': []}
        self.target_values = data[target_column].to_numpy()
        data_no_target = data.drop(columns=[target_column])

        print('Setting default values for factors...', flush=True)
        minimum = len(data_no_target) * rarity_ignore
        factors_instances_counter = {column: Counter(data_no_target[column]) for column in data_no_target}
        self.additions = {column: {factor: 0.0
                                   for factor in set(data_no_target[column]).union({None})
                                   if factor is None or factors_instances_counter[column][factor] >= minimum}
                          for column in data_no_target}
        self.multipliers = {column: {factor: 1.0
                                     for factor in set(data_no_target[column]).union({None})
                                     if factor is None or factors_instances_counter[column][factor] >= minimum}
                            for column in data_no_target}
        for column in factors_instances_counter:
            for item in factors_instances_counter[column]:
                if factors_instances_counter[column][item] < minimum:
                    data_no_target[column].replace({item: None}, inplace=True)

        for column in self.additions:
            print(f'{column} has {len(self.additions[column])} factors')

        print('Evaluating instances with the same factor...', flush=True)
        self.factors_instances = {}
        for column in data_no_target:
            column_values = data_no_target[column]
            self.factors_instances[column] = defaultdict(list)
            list_index = 0
            for index, factor in column_values.items():
                if not isinstance(factor, float) or not math.isnan(factor):
                    self.factors_instances[column][factor] += [list_index]
                list_index += 1

        self.rows = [{column: factor for column, factor in data.items()} for index, data in
                     data_no_target.iterrows()]
        self.current_targets = self.predict_target_values()

        self.instance_multipliers = [1 for _ in range(0, len(self.rows))]
        self.refresh_instance_multipliers()

    def refresh_current_targets(self):
        self.current_targets = self.predict_target_values()

    def refresh_instance_multipliers(self):
        self.instance_multipliers = [1 for _ in range(0, len(self.rows))]
        for column in self.factors_instances:
            for factor in self.factors_instances[column]:
                for index in self.factors_instances[column][factor]:
                    self.instance_multipliers[index] *= self.multipliers[column][factor]

    def refresh_temporal_values(self):
        self.refresh_current_targets()
        self.refresh_instance_multipliers()

    @staticmethod
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def find_prediction_error(self):
        predicted_values = self.predict_target_values()
        return self.mean_absolute_percentage_error(self.target_values, predicted_values)

    def _load_factors_values(self, filename):
        [values, multipliers] = np.load(filename, allow_pickle=True)
        for column in self.additions:
            if column in values:
                for factor in self.additions[column]:
                    if factor in values[column]:
                        self.additions[column][factor] = values[column][factor]
        for column in self.multipliers:
            if column in multipliers:
                for factor in self.multipliers[column]:
                    if factor in multipliers[column]:
                        self.multipliers[column][factor] = multipliers[column][factor]

    def _get_optimal_factor_addition(self, column, factor):
        entities = [Entity(self.target_values[instance],
                           self.current_targets[instance],
                           self.instance_multipliers[instance])
                    for instance
                    in self.factors_instances[column][factor]]

        entities.sort(key=lambda entity: entity.get_difference())
        derivative = sum([entity.get_mape_addition_derivative() for entity in entities])
        difference = 0
        direction = 1
        if derivative < 0:
            direction = -1
            entities.reverse()
        if derivative != 0:
            for entity in entities:
                if derivative * direction < 0:
                    break
                if entity.get_difference() * direction > 0:
                    difference = entity.get_difference()
                    derivative -= entity.get_mape_addition_derivative()
                    entity.predicted_value = entity.target_value
                if entity.get_difference() == 0:
                    entity.predicted_value += direction
                    derivative += entity.get_mape_addition_derivative()
        return difference

    def _get_optimal_factor_multiplier(self, column, factor):
        entities = [Entity(self.target_values[instance],
                           self.current_targets[instance],
                           self.instance_multipliers[instance])
                    for instance
                    in self.factors_instances[column][factor]]

        entities.sort(key=lambda entity: entity.get_coefficient())
        derivative = sum([entity.get_mape_multiplier_derivative(1) for entity in entities])
        coefficient = 1
        direction = 1
        if derivative < 0:
            direction = -1
            entities.reverse()
        if derivative != 0:
            for entity in entities:
                if derivative * direction < 0:
                    break
                if entity.get_coefficient() * direction > 1 * direction:
                    coefficient = entity.get_coefficient()
                    derivative -= entity.get_mape_multiplier_derivative(1)
                    entity.predicted_value = entity.target_value
                    entity.multiplier *= coefficient
                if entity.get_coefficient() == 1:
                    entity.predicted_value *= 2
                    derivative += entity.get_mape_multiplier_derivative(coefficient * 2) * direction
        return coefficient

    def _train_once(self):
        for column in self.additions:
            if column in self.ignore['additions']:
                continue
            column_additions = self.additions[column]
            print(f'Calculating addition: {column}', flush=True)
            for factor in column_additions:
                factor_value = column_additions[factor]
                for instance in self.factors_instances[column][factor]:
                    self.current_targets[instance] -= factor_value * self.instance_multipliers[instance]
                new_value = self._get_optimal_factor_addition(column, factor)
                column_additions[factor] = new_value
            self.refresh_temporal_values()
        for column in self.multipliers:
            if column in self.ignore['multipliers']:
                continue
            column_multipliers = self.multipliers[column]
            print(f'Calculating multiplier: {column}', flush=True)
            for factor in column_multipliers:
                factor_multiplier = column_multipliers[factor]
                for instance in self.factors_instances[column][factor]:
                    self.current_targets[instance] /= factor_multiplier
                    self.instance_multipliers[instance] /= factor_multiplier
                new_multiplier = self._get_optimal_factor_multiplier(column, factor)
                column_multipliers[factor] = new_multiplier
            self.refresh_temporal_values()

    def train(self, times=1, factors_values_filename: str = None, ignore: dict = None,
              save_period=1, models_directory=''):
        if len(models_directory) > 0:
            if models_directory[-1] != '/' or models_directory[-1] != '\\':
                models_directory += '/'
        if ignore is not None:
            self.ignore = ignore
        if factors_values_filename is not None:
            self._load_factors_values(factors_values_filename)
            if ignore is not None:
                if 'additions' in ignore:
                    for column in ignore['additions']:
                        if column in ignore['multipliers']:
                            self.additions.pop(column, None)
                        else:
                            for factor in self.additions[column]:
                                self.additions[column][factor] = 0
                if 'multipliers' in ignore:
                    for column in ignore['multipliers']:
                        if column in ignore['additions']:
                            self.multipliers.pop(column, None)
                        else:
                            for factor in self.multipliers[column]:
                                self.multipliers[column][factor] = 1
        self.current_targets = self.predict_target_values()
        for index in range(times):
            start = time()
            self._train_once()
            error = self.find_prediction_error()
            print(f'Train #{index} for {time() - start} s with {error}% error', flush=True)
            if save_period != 0 and (index + 1) % save_period == 0:
                path = f'{models_directory}{int(time())}_{error}.npy'
                np.save(path, [self.additions, self.multipliers])
                print(f' was saved in {path}')
            else:
                print()
        if save_period == 0 or times % save_period != 0:
            error = self.find_prediction_error()
            path = f'{models_directory}{int(time())}_{error}.npy'
            np.save(path, [self.additions, self.multipliers])
            print(f'Train result was saved in {path}', flush=True)


def main(argv):
    blacklist = ['engine_capacity', 'zipcode', 'city']
    ignore = {'additions': [column for column in blacklist], 'multipliers': [column for column in blacklist]}
    ignore['additions'] += ['brand', 'type', 'mileage', 'power']
    ignore['multipliers'] += ['fuel']
    input_train_data_path = None
    target = None
    training_kwargs = {}
    help_message = ' -i <input_train_data_path> -g <target> [-t <iterations_count>' \
                   ' -f <factors_values_filename> -s <save_period>]'
    try:
        opts, args = getopt.getopt(argv, "hi:g:t:f:s:", [])
    except getopt.GetoptError:
        print(help_message)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(help_message)
            sys.exit()
        elif opt == "-i":
            input_train_data_path = arg
        elif opt == "-g":
            target = arg
        elif opt == "-t":
            training_kwargs["times"] = int(arg)
        elif opt == "-f":
            training_kwargs["factors_values_filename"] = arg
        elif opt == "-s":
            training_kwargs["save_period"] = int(arg)
    if input_train_data_path is not None and target is not None:
        predictor = TrainingPredictor(input_train_data_path, target)
        predictor.train(ignore=ignore, **training_kwargs)
    else:
        if input_train_data_path is None:
            print("Wrong -i")
        if target is None:
            print("Wrong -g")
        print(help_message)


if __name__ == "__main__":
    main(sys.argv[1:])
