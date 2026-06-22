from __future__ import annotations

import pyspark.sql.functions as F
import numpy as np

from pyspark.ml.feature import StringIndexer
from pyspark.sql import DataFrame
from math import sqrt, log, ceil
from numpy import linspace, interp, arange, exp, sum as np_sum
from abc import ABC
from typing import List, Union, Dict, Any
from scipy.stats import ks_2samp, kstwobign


class StaticTest(ABC):
    """
    Базовый абстрактный класс для проведения статистических тестов на Spark DataFrame.
    
    Отвечает за базовую подготовку данных: удаление пропусков (NULL) и 
    кодирование строковых меток классов в числовые индексы.
    """

    def __init__(self,
                 data: DataFrame,
                 target_columns: Union[str, List[str]],
                 label_column: str,
                 reliability: float = 0.05):
        """
        Инициализация базового тестового класса.

        Args:
            data (DataFrame): Входной Spark DataFrame с данными.
            target_columns (Union[str, List[str]]): Имя целевой колонки или список имен 
                колонок для проведения тестов.
            label_column (str): Имя колонки, содержащей метки групп (например, 1 для control, 2 для test).
            reliability (float, optional): Уровень значимости (alpha) для проверки гипотезы. 
                По умолчанию 0.05.
        """
        # СЕЙЧАС ВСЕ NULL ВЫБРАСЫВАЮТСЯ !!!!!!!!!!!!!!!!!!
        self.data = data.na.drop()
        self.target_columns = target_columns
        self.label_column = label_column
        self.reliability = reliability
        
        self._prepare_label_column()

    def _prepare_label_column(self) -> None:
        """
        Преобразует строковую колонку меток в числовую с помощью StringIndexer, 
        если это необходимо. Модифицирует self.data и self.label_column inplace.
        """
        label_type = dict(self.data.dtypes)[self.label_column]
        if label_type in ['string', 'varchar']:
            stringIndexer = StringIndexer(
                inputCol=self.label_column, 
                outputCol=f"{self.label_column}_indexed"
            )
            self.data = stringIndexer.fit(self.data).transform(self.data)
            self.label_column = f"{self.label_column}_indexed"

    def _target_column_extracting(self, target_column: str) -> tuple:
        """
        Разделяет данные на две выборки (test и control) на основе значений в label_column.

        Args:
            target_column (str): Имя колонки с числовыми данными для тестирования.

        Returns:
            tuple: Кортеж из двух DataFrame (test_column, control_column), 
                   где test соответствует label == 2, а control == 1.
        """
        test_column = self.data.filter(F.col(self.label_column) == 2).select(F.col(target_column))
        control_column = self.data.filter(F.col(self.label_column) == 1).select(F.col(target_column))
        return test_column, control_column


class ApproxKStest(StaticTest):
    """
    Реализация приближенного двухвыборочного теста Колмогорова-Смирнова (KS-test) для Apache Spark.
    
    Основан на статье "Two-sample KS test with approxQuantile in Apache Spark" (Eck et al., 2023).
    Вместо полной сортировки данных (что дорого в Big Data), метод использует 
    ε-приближенные квантили (approxQuantile) для построения аппроксимированных 
    эмпирических функций распределения (CDF) и вычисления статистики D_KS с гарантированными 
    теоретическими границами ошибки.
    """

    def __init__(self, 
                 data: DataFrame, 
                 target_columns: Union[str, List[str]], 
                 label_column: str, 
                 reliability: float = 0.05, 
                 error: float = 0.01):
        """
        Инициализация KS-теста.

        Args:
            data (DataFrame): Входной Spark DataFrame.
            target_columns (Union[str, List[str]]): Целевая колонка(и) для анализа.
            label_column (str): Колонка с метками групп (1 и 2).
            reliability (float, optional): Уровень значимости (alpha). По умолчанию 0.05.
            error (float, optional): Допустимая погрешность (beta) для расчета границ 
                критического расстояния D_crit. По умолчанию 0.01.
        """
        super().__init__(data, target_columns, label_column, reliability)
        self.error = error

    def calculate(self) -> Dict[str, Dict[str, Any]]:
        """
        Запускает расчет KS-теста для одной или нескольких целевых колонок.

        Returns:
            Dict[str, Dict[str, Any]]: Словарь, где ключи — это имена колонок, 
            а значения — словари с результатами теста (p-value, statistic, pass, размеры выборок и скетчей).
        """
        if isinstance(self.target_columns, (str, int, float)):
            test_col, control_col = self._target_column_extracting(self.target_columns)
            return {
                str(self.target_columns): self._single_calc(test_col, control_col)
            }

        if isinstance(self.target_columns, list):
            result = {}
            for target_column in self.target_columns:
                test_col, control_col = self._target_column_extracting(target_column)
                result[target_column] = self._single_calc(test_col, control_col)
            return result 
        
        raise ValueError("target_columns должен быть строкой, числом или списком")

    def _single_calc(self, test_column: DataFrame, control_column: DataFrame) -> Dict[str, Any]:
        """
        Выполняет расчет приближенного KS-теста для одной пары выборок.

        Args:
            test_column (DataFrame): Выборка тестовой группы (1 колонка).
            control_column (DataFrame): Выборка контрольной группы (1 колонка).

        Returns:
            Dict[str, Any]: Словарь с метриками:
                - "p-value" (float): Рассчитанное p-value.
                - "statistic" (float): Статистика D_KS (максимальное расстояние между CDF).
                - "pass" (bool): True, если p-value < reliability (гипотеза H0 отвергается).
                - "test_n" (int): Размер тестовой выборки.
                - "control_n" (int): Размер контрольной выборки.
                - "sketch_test_size" (int): Количество точек в скетче тестовой выборки.
                - "sketch_control_size" (int): Количество точек в скетче контрольной выборки.
        """
        test_count = test_column.count()
        control_count = control_column.count()

        # Расчет допустимой ошибки дельта  (δ)
        delta = self._error_bound(test_count, control_count) / 2
        if delta <= 0:
            delta = 1e-5 

        # формула "единичного наклона" из статьи
        ksi_test = max(0.0, delta - sqrt(delta / test_count))
        ksi_control = max(0.0, delta - sqrt(delta / control_count))

        # Расчет размера скетча для аппроксимации CDF
        # Упрощенная форма: a = sqrt(N / δ) + 1
        sketch_test = min(ceil(sqrt(test_count / delta) + 1), test_count)
        sketch_control = min(ceil(sqrt(control_count / delta) + 1), control_count)

        # Расчет вероятносте 
        rangs_test = linspace(1 / test_count, 1, sketch_test)
        rangs_control = linspace(1 / control_count, 1, sketch_control)

        col_name = test_column.columns[0]
        
        F_test = test_column.approxQuantile(col_name, list(rangs_test), ksi_test)
        F_control = control_column.approxQuantile(col_name, list(rangs_control), ksi_control)

        # Линейная интерполяция CDF
        F_xi = rangs_test
        F_yi = interp(F_test, F_control, rangs_control)
        
        F_yj = rangs_control
        F_xj = interp(F_control, F_test, rangs_test)

        # Кс-статиситика 
        D_1 = max(abs(F_xi - F_yi))
        D_2 = max(abs(F_xj - F_yj))
        D_ks = max(D_1, D_2)

        #p-value через асимптотическое распределение Колмогорова (формула 2-3 из статьи)
        n_eff = (test_count * control_count) / (test_count + control_count)
        lambda_val = sqrt(n_eff) * D_ks
        p_value = self._kolmogorov_sf(lambda_val)

        return {
            "p-value": float(p_value),
            "statistic": float(D_ks),
            "pass": bool(p_value < self.reliability),
            "test_n": int(test_count),
            "control_n": int(control_count),
            "sketch_test_size": int(sketch_test),
            "sketch_control_size": int(sketch_control)
        }

    def _error_bound(self, n: int, m: int) -> float:
        """
        Вычисляет допустимую погрешность φ (phi) для статистики D_KS.
        Основано на разнице критических значений D_crit для уровня значимости alpha 
        и его границ (alpha ± error).

        Args:
            n (int): Размер первой выборки.
            m (int): Размер второй выборки.

        Returns:
            float: Значение phi, гарантирующее, что ошибка аппроксимации не ухудшит 
                   качество статистического вывода.
        """
        alpha_up = self.reliability + self.error
        alpha_down = max(1e-10, self.reliability - self.error)  # Защита от log(0)
        alpha = self.reliability
        
        crit_D_alpha = self._D_crit(n, m, alpha)
        crit_D_alpha_up = self._D_crit(n, m, alpha_up)
        crit_D_alpha_down = self._D_crit(n, m, alpha_down)

        return float(min(abs(crit_D_alpha - crit_D_alpha_down), abs(crit_D_alpha - crit_D_alpha_up)))

    @staticmethod
    def _D_crit(n: int, m: int, alpha: float) -> float:
        """
        Вычисляет критическое значение статистики D для заданного уровня значимости alpha.

        Args:
            n (int): Размер первой выборки.
            m (int): Размер второй выборки.
            alpha (float): Уровень значимости.

        Returns:
            float: Критическое расстояние D_crit.
        """
        if alpha >= 2.0 or alpha <= 0:
            return 0.0
        c_alpha = sqrt(-log(alpha / 2) * 0.5)
        multiplier = sqrt((n + m) / (n * m))
        return float(c_alpha * multiplier)

    @staticmethod
    def _kolmogorov_sf(lambda_val: float, num_terms: int = 100) -> float:
        """
        Вычисляет функцию выживания (Survival Function, 1 - CDF) распределения Колмогорова.
        Использует разложение в ряд (формула 3 из статьи Eck et al.):
        Q_KS(λ) = 2 * Σ_{k=1}^{∞} (-1)^{k-1} * exp(-2 * k^2 * λ^2)

        Args:
            lambda_val (float): Значение λ = D_KS * sqrt(N*M / (N+M)).
            num_terms (int, optional): Количество членов ряда для суммирования. 
                По умолчанию 100 (достаточно для машинной точности при λ > 0.01).

        Returns:
            float: Рассчитанное p-value.
        """
        k = arange(1, num_terms + 1)
        terms = (-1) ** (k - 1) * exp(-2 * k ** 2 * lambda_val ** 2)
        return float(2 * np_sum(terms))
    

class ScipyKStest(StaticTest):

    def __init__(self, data, target_columns, label_column, reliability = 0.05):
        super().__init__(data, target_columns, label_column, reliability)

    def _single_calc(self, test_column: DataFrame, control_column: DataFrame) -> Dict[str, Any]:
        test_column_arr = np.array(test_column.collect())
        control_column_arr = np.array(control_column.collect())

        statistic, p_value, *_ = ks_2samp(test_column_arr, control_column_arr)
        return {
            "p-value": float(p_value),
            "statistic": float(statistic),
            "pass": bool(p_value < self.reliability),
            "test_n": len(test_column_arr),
            "control_n": int(control_column_arr),
        }

    def calculate(self) -> Dict[str, Dict[str, Any]]:
        """
        Запускает расчет KS-теста для одной или нескольких целевых колонок.

        Returns:
            Dict[str, Dict[str, Any]]: Словарь, где ключи — это имена колонок, 
            а значения — словари с результатами теста (p-value, statistic, pass, размеры выборок и скетчей).
        """
        if isinstance(self.target_columns, (str, int, float)):
            test_col, control_col = self._target_column_extracting(self.target_columns)
            return {
                str(self.target_columns): self._single_calc(test_col, control_col)
            }

        if isinstance(self.target_columns, list):
            result = {}
            for target_column in self.target_columns:
                test_col, control_col = self._target_column_extracting(target_column)
                result[target_column] = self._single_calc(test_col, control_col)
            return result 
        
        raise ValueError("target_columns должен быть строкой, числом или списком")
    

class GroupKSTestExtension(StaticTest):
    """
    Реализация двухвыборочного теста Колмогорова-Смирнова (KS-test) на основе гистограмм.
    
    Метод дискретизирует данные в фиксированное количество бинов (n_bins),
    строит эмпирические функции распределения (ECDF) через кумулятивные суммы
    и вычисляет максимальное расстояние между ними. Использует поправку Стефенса
    для точного расчета p-value.
    """
    
    def __init__(self, 
                 data: DataFrame, 
                 target_columns: Union[str, List[str]], 
                 label_column: str, 
                 reliability: float = 0.05,
                 n_bins: int = 2000):
        """
        Инициализация гистограммного KS-теста.

        Args:
            data (DataFrame): Входной Spark DataFrame.
            target_columns (Union[str, List[str]]): Целевая колонка(и) для анализа.
            label_column (str): Колонка с метками групп (1 и 2).
            reliability (float, optional): Уровень значимости (alpha). По умолчанию 0.05.
            n_bins (int, optional): Количество бинов для гистограммы. По умолчанию 2000.
        """
        super().__init__(data, target_columns, label_column, reliability)
        self.n_bins = n_bins

    def calculate(self) -> Dict[str, Dict[str, Any]]:
        """
        Запускает расчет KS-теста для одной или нескольких целевых колонок.

        Returns:
            Dict[str, Dict[str, Any]]: Словарь, где ключи — это имена колонок, 
            а значения — словари с результатами теста (p-value, statistic, pass).
        """
        if isinstance(self.target_columns, (str, int, float)):
            test_col, control_col = self._target_column_extracting(self.target_columns)
            return {
                str(self.target_columns): self._single_calc(test_col, control_col)
            }

        if isinstance(self.target_columns, list):
            result = {}
            for target_column in self.target_columns:
                test_col, control_col = self._target_column_extracting(target_column)
                result[target_column] = self._single_calc(test_col, control_col)
            return result 
        
        raise ValueError("target_columns должен быть строкой, числом или списком")

    def _single_calc(self, test_column: DataFrame, control_column: DataFrame) -> Dict[str, Any]:
        """
        Выполняет расчет KS-теста через гистограммы для одной пары выборок.

        Args:
            test_column (DataFrame): Выборка тестовой группы (1 колонка).
            control_column (DataFrame): Выборка контрольной группы (1 колонка).

        Returns:
            Dict[str, Any]: Словарь с метриками:
                - "p-value" (float): Рассчитанное p-value.
                - "statistic" (float): Статистика D_KS (максимальное расстояние между ECDF).
                - "pass" (bool): True, если p-value < reliability (гипотеза H0 отвергается).
                - "test_n" (int): Размер тестовой выборки.
                - "control_n" (int): Размер контрольной выборки.
        """
        col_name = test_column.columns[0]
        
        # Получаем размеры выборок
        n1 = test_column.count()
        n2 = control_column.count()
        
        # Обработка граничных случаев: пустые выборки
        if n1 == 0 or n2 == 0:
            return {
                "p-value": None,
                "statistic": None,
                "pass": None,
                "test_n": n1,
                "control_n": n2,
            }

        # Вычисляем глобальные минимум и максимум для общих границ бинов
        bounds1 = test_column.agg(
            F.min(col_name).alias("min1"), 
            F.max(col_name).alias("max1")
        ).collect()[0]
        
        bounds2 = control_column.agg(
            F.min(col_name).alias("min2"), 
            F.max(col_name).alias("max2")
        ).collect()[0]

        global_min = min(bounds1["min1"], bounds2["min2"])
        global_max = max(bounds1["max1"], bounds2["max2"])

        # Граничный случай: все значения одинаковы (нулевая дисперсия)
        if global_min == global_max:
            return {
                "p-value": 1.0,
                "statistic": 0.0,
                "pass": 1.0 < self.reliability,
                "test_n": n1,
                "control_n": n2,
            }

        # Функция для добавления колонки с номером бина
        def _add_bucket_column(df: DataFrame, 
                               global_min: float, 
                               global_max: float) -> DataFrame:
            """
            Присваивает каждой строке номер дискретного гистограммного бина.
            Использует глобальные min/max для обеспечения одинаковых границ бинов.
            """
            width = (global_max - global_min) / self.n_bins
            return df.withColumn(
                "bucket",
                F.least(
                    F.floor((F.col(col_name) - global_min) / width),
                    F.lit(self.n_bins - 1)  # Ограничиваем последним бином
                ).cast("int")
            )

        # Строим гистограммы (подсчет частот по бинам) для обеих выборок
        hist1 = (_add_bucket_column(test_column, global_min, global_max)
            .groupBy("bucket")
            .count()
            .withColumnRenamed("count", "c1")
        )
        
        hist2 = (_add_bucket_column(control_column, global_min, global_max)
            .groupBy("bucket")
            .count()
            .withColumnRenamed("count", "c2")
        )

        # Объединяем гистограммы через outer join, заполняя пропущенные бины нулями
        combined = hist1.join(hist2, on="bucket", how="outer").fillna(0, subset=["c1", "c2"])
        
        # Конвертируем в Pandas для эффективного вычисления кумулятивных сумм (ECDF)
        # Количество строк <= n_bins, поэтому безопасно помещается в память драйвера
        pdf = combined.toPandas().sort_values("bucket").reset_index(drop=True)
        pdf["cum1"] = pdf["c1"].cumsum()
        pdf["cum2"] = pdf["c2"].cumsum()
        
        # Вычисляем статистику KS: максимальная абсолютная разница между ECDF
        d_stat = float((pdf["cum1"]/n1 - pdf["cum2"]/n2).abs().max())

        try:
            # Вычисляем эффективный размер выборки для асимптотического распределения
            en = np.sqrt(n1 * n2 / (n1 + n2))
            
            # Применяем поправку Стефенса (1970) для лучшей точности на конечных выборках.
            # Это та же формула, что используется внутри scipy.stats.ks_2samp для больших выборок.
            p_value = float(kstwobign.sf((en + 0.12 + 0.11 / en) * d_stat))
        except Exception:
            # Запасной вариант: 0.0 если статистическая функция не сработала
            p_value = 0.0 

        return {
            "p-value": p_value,
            "statistic": d_stat,
            "pass": p_value < self.reliability,
            "test_n": n1,
            "control_n": n2,
        }