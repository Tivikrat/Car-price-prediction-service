from math import isnan

import numpy
import pandas
from flask import Flask, request, render_template
from pandas import DataFrame, notnull

app = Flask(__name__)


class Predictor:
    def __init__(self):
        self.additions = None
        self.multipliers = None
        self.rows = None

    @staticmethod
    def _read_data(data_path):
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
            elif isinstance(factor, float) and not isnan(factor):
                if column == 'insurance_price':
                    target += factor * 8
                    multiplier *= (0.000002356781372541904 * factor + 0.9677092756497888)
                elif column == 'power':
                    multiplier *= (0.0011040866703426102 * factor + 0.8993661804773361)
                elif column == 'engine_capacity':
                    target += (60.84013433324293 * factor + 1084.2797944627148)
        return target * multiplier

    def _predict_target_values(self, data=None):
        if data is None:
            return [self.predict_target_value(instance) for instance in self.rows]
        else:
            return [self.predict_target_value({column: factor for column, factor in instance.items()}) for
                    index, instance in data.iterrows()]

    def predict(self, data_path, prediction_path, column_name='Predicted', index_name='Id'):
        print('\rReading data...', end='')
        data = self._read_data(data_path)
        print('\rPredicting prices...', end='')
        prices = self._predict_target_values(data)
        indexes = [index for index, data in data.iterrows()]
        cars_prices = DataFrame(prices, index=indexes, columns=[column_name]).rename_axis(index_name)
        print('\rWriting results...', end='')
        cars_prices.to_csv(prediction_path)


class LoadingPredictor(Predictor):
    def __init__(self, filename, ignore_columns: list):
        super().__init__()
        print('Loading values for factors...', end='')
        [additions, multipliers] = numpy.load(filename, allow_pickle=True)
        self.additions = {column: {factor: additions[column][factor]
                                   for factor in additions[column]}
                          for column in additions
                          if column not in ignore_columns}
        self.multipliers = {column: {factor: multipliers[column][factor]
                                     for factor in multipliers[column]}
                            for column in multipliers
                            if column not in ignore_columns}


@app.route('/')
def index_page():
    return render_template("index.html", gearbox=2)


def int_try_parse(value):
    try:
        return int(value)
    except ValueError:
        return None


def float_try_parse(value):
    try:
        return float(value)
    except ValueError:
        return None


@app.route('/version/')
def version():
    return render_template("version.html")


@app.route('/predict/')
def calculate_page():
    car = {'engine_capacity': float_try_parse(request.args.get('engine_capacity', None)),
           'type': request.args.get('type', None),
           'registration_year': int_try_parse(request.args.get('registration_year', None)),
           'gearbox': request.args.get('gearbox', None),
           'power': int_try_parse(request.args.get('power', None)),
           'model': request.args.get('model', None),
           'mileage': int_try_parse(request.args.get('mileage', None)),
           'fuel': request.args.get('fuel', None),
           'brand': request.args.get('brand', None),
           'damage': float_try_parse(request.args.get('damage', None)),
           'insurance_price': float_try_parse(request.args.get('insurance_price', None))}
    try:
        return render_template("index.html", price=str(LoadingPredictor('model.npy', []).predict_target_value(car)),
                               **car)
    except FileNotFoundError:
        return "Error loading model.npy"


if __name__ == '__main__':
    app.run()
