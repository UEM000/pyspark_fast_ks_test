from __future__ import annotations
import pyspark.sql.functions as F
from pyspark.ml.feature import StringIndexer
from pyspark.sql import DataFrame
from math import sqrt, log, ceil
from numpy import linspace, interp, arange, exp, sum as np_sum
from abc import ABC
from typing import List, Union


class StaticTest(ABC):
    def __init__(self,
                 data: DataFrame,
                 target_columns: Union[str, List[str]],
                 label_column: str,
                 reliability: float = 0.05):
        self.data = data.na.drop()
        self.target_columns = target_columns
        self.label_column = label_column
        self.reliability = reliability
        self._prepare_label_column()

    def _prepare_label_column(self):
        label_type = dict(self.data.dtypes)[self.label_column]
        if label_type in ['string', 'varchar']:
            stringIndexer = StringIndexer(inputCol=self.label_column, 
                                          outputCol=f"{self.label_column}_indexed")
            self.data = stringIndexer.fit(self.data).transform(self.data)
            self.label_column = f"{self.label_column}_indexed"

    def _target_column_extracting(self, target_column: str) -> tuple:
        test_column = self.data.filter(F.col(self.label_column) == 2).select(F.col(target_column))
        control_column = self.data.filter(F.col(self.label_column) == 1).select(F.col(target_column))
        return test_column, control_column


class MyKStest(StaticTest):
    def __init__(self, data, target_columns, label_column, reliability=0.05, error=0.01):
        super().__init__(data, target_columns, label_column, reliability)
        self.error = error

    def calculate(self) -> dict:
        if isinstance(self.target_columns, (str, int, float)):
            test_col, control_col = self._target_column_extracting(self.target_columns)
            return {self.target_columns: self._single_calc(test_col, control_col)}

        if isinstance(self.target_columns, list):
            result = {}
            for target_column in self.target_columns:
                test_col, control_col = self._target_column_extracting(target_column)
                result[target_column] = self._single_calc(test_col, control_col)
            return result 
        
    def _single_calc(self, test_column: DataFrame, control_column: DataFrame) -> dict:
        test_count = test_column.count()
        control_count = control_column.count()

        delta = self._error_bound(test_count, control_count) / 2
        if delta <= 0:
            delta = 1e-5

        ksi_test = max(0.0, delta - sqrt(delta / test_count))
        ksi_control = max(0.0, delta - sqrt(delta / control_count))

        sketch_test = min(ceil(sqrt(test_count / delta) + 1), test_count)
        sketch_control = min(ceil(sqrt(control_count / delta) + 1), control_count)

        rangs_test = linspace(1 / test_count, 1, sketch_test)
        rangs_control = linspace(1 / control_count, 1, sketch_control)

        col_name = test_column.columns[0]
        
        F_test = test_column.approxQuantile(col_name, list(rangs_test), ksi_test)
        F_control = control_column.approxQuantile(col_name, list(rangs_control), ksi_control)

        F_xi = rangs_test
        F_yi = interp(F_test, F_control, rangs_control)
        
        F_yj = rangs_control
        F_xj = interp(F_control, F_test, rangs_test)

        D_1 = max(abs(F_xi - F_yi))
        D_2 = max(abs(F_xj - F_yj))
        D_ks = max(D_1, D_2)

        n_eff = (test_count * control_count) / (test_count + control_count)
        lambda_val = sqrt(n_eff) * D_ks
        p_value = self._kolmogorov_sf(lambda_val)

        return {
            "p-value": p_value,
            "statistic": D_ks,
            "pass": p_value < self.reliability,
            "test_n": test_count,
            "control_n": control_count,
            "sketch_test_size": sketch_test,
            "sketch_control_size": sketch_control
        }

    def _error_bound(self, n: int, m: int) -> float:
        alpha_up = self.reliability + self.error
        alpha_down = max(1e-10, self.reliability - self.error)
        alpha = self.reliability
        
        crit_D_alpha = self._D_crit(n, m, alpha)
        crit_D_alpha_up = self._D_crit(n, m, alpha_up)
        crit_D_alpha_down = self._D_crit(n, m, alpha_down)

        return min(abs(crit_D_alpha - crit_D_alpha_down), abs(crit_D_alpha - crit_D_alpha_up))

    @staticmethod
    def _D_crit(n: int, m: int, alpha: float) -> float:
        if alpha >= 2.0 or alpha <= 0:
            return 0.0
        c_alpha = sqrt(-log(alpha / 2) * 0.5)
        multiplier = sqrt((n + m) / (n * m))
        return c_alpha * multiplier

    @staticmethod
    def _kolmogorov_sf(lambda_val: float, num_terms: int = 100) -> float:
        """
        Вычисляет survival function (1 - CDF) распределения Колмогорова.
        Формула (3) из статьи:
        Q_KS(lambda) = 2 * sum_{k=1}^{inf} (-1)^{k-1} * exp(-2*k^2*lambda^2)
        """
        k = arange(1, num_terms + 1)
        terms = (-1) ** (k - 1) * exp(-2 * k ** 2 * lambda_val ** 2)
        return 2 * np_sum(terms)