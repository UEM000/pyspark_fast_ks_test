import os
import time
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
from pyspark.sql.functions import rand, col
from scipy.stats import ks_2samp
from typing import Dict, Any, List, Optional, Union
from tqdm import tqdm

from statistic_spark import ApproxKStest, ScipyKStest, GroupKSTestExtension


class KSExperiment:
    """
    Эксперимент для сравнения трех реализаций KS-теста:
    1. ApproxKStest - приближенный через approxQuantile (Eck et al., 2023)
    2. ScipyKStest - точный через scipy (baseline)
    3. GroupKSTestExtension - гистограммный метод
    
    Конфигурация подается как параметр (словарь), что позволяет 
    работать из Jupyter Notebook без перезагрузки модулей.
    """
    
    def __init__(self, spark: SparkSession):
        """
        Args:
            spark: Активная сессия Spark
        """
        self.spark = spark
        self.spark.sparkContext.setLogLevel("ERROR")
    
    def run(self, 
            data: SparkDataFrame,
            config: Dict[str, Any]) -> pd.DataFrame:
        """
        Запуск эксперимента сравнения методов KS-теста.
        
        Args:
            data: Входной Spark DataFrame с данными
            config: Словарь конфигурации эксперимента. Пример:
                {
                    "columns": ["feature1", "feature2", "feature3"],
                    "label_column": "label",
                    "group_split_ratio": 0.5,
                    "iterations": 5,
                    "error_params": [0.005, 0.01, 0.05],
                    "n_bins_params": [1000, 2000, 5000],
                    "reliability": 0.05,
                    "random_state": 42,
                    "output_file": "ks_experiment_results.csv",
                    "write_header": True,
                    "create_label": True,
                    "label_test_value": 2,
                    "label_control_value": 1,
                    "show_progress": True,  # НОВОЕ: показывать прогресс-бары
                }
        
        Returns:
            pd.DataFrame: DataFrame с результатами эксперимента
        """
        # Параметры по умолчанию
        default_config = {
            "columns": [],
            "label_column": "label",
            "group_split_ratio": 0.5,
            "iterations": 3,
            "error_params": [0.01],
            "n_bins_params": [2000],
            "reliability": 0.05,
            "random_state": 42,
            "output_file": "ks_experiment_results.csv",
            "write_header": True,
            "create_label": True,
            "label_test_value": 2,
            "label_control_value": 1,
            "show_progress": True,  # НОВОЕ: по умолчанию показывать прогресс
        }
        
        # Объединяем с дефолтными значениями
        config = {**default_config, **config}
        
        # Проверка обязательных параметров
        if not config["columns"]:
            raise ValueError("Параметр 'columns' не может быть пустым")
        
        show_progress = config["show_progress"]
        
        # Подготовка данных: разбиение на группы
        prepared_data = self._split_into_groups(
            data=data,
            label_column=config["label_column"],
            split_ratio=config["group_split_ratio"],
            create_label=config["create_label"],
            test_value=config["label_test_value"],
            control_value=config["label_control_value"],
            random_state=config["random_state"]
        )
        
        # Проверяем, нужно ли писать заголовок
        write_header = (
            config["write_header"] and 
            (not os.path.exists(config["output_file"]) or 
             os.path.getsize(config["output_file"]) == 0)
        )
        
        all_results = []
        
        # Открываем файл для записи
        with open(config["output_file"], 'a', encoding='utf-8') as f:
            if write_header:
                f.write(
                    "column;method;error_param;n_bins;iteration;"
                    "time_sec;compression_pct;stat_diff;pvalue_diff;pass_match\n"
                )
            
            # Получаем размеры групп для информации
            test_count = prepared_data.filter(
                col(config["label_column"]) == config["label_test_value"]
            ).count()
            control_count = prepared_data.filter(
                col(config["label_column"]) == config["label_control_value"]
            ).count()
            
            print(f"📊 Размеры групп: Test={test_count}, Control={control_count}")
            print(f"📋 Колонки для анализа: {config['columns']}")
            
            # ===== БАЗОВЫЙ РАСЧЕТ ЧЕРЕZ SCIPY (эталон) =====
            scipy_results = {}
            columns_iter = tqdm(
                config["columns"], 
                desc="🔬 Scipy baseline", 
                unit="col",
                colour='blue',
                disable=not show_progress
            )
            for column in columns_iter:
                scipy_results[column] = self._run_scipy_baseline(
                    df=prepared_data,
                    column=column,
                    label_column=config["label_column"],
                    reliability=config["reliability"],
                    test_value=config["label_test_value"],
                    control_value=config["label_control_value"]
                )
            
            # ===== ТЕСТИРОВАНИЕ МЕТОДОВ =====
            columns_iter = tqdm(
                config["columns"], 
                desc="🧪 Тестирование методов", 
                unit="col",
                colour='green',
                disable=not show_progress
            )
            
            for column in columns_iter:
                # --- ApproxKStest с разными error ---
                error_iter = tqdm(
                    config["error_params"], 
                    desc=f"   ApproxKStest [{column}]", 
                    unit="error",
                    colour='cyan',
                    leave=False,  # Не оставлять после завершения
                    disable=not show_progress
                )
                
                for error_param in error_iter:
                    iter_iter = tqdm(
                        range(config["iterations"]), 
                        desc=f"    Итерации (error={error_param})", 
                        unit="iter",
                        colour='cyan',
                        leave=False,
                        disable=not show_progress
                    )
                    
                    for iteration in iter_iter:
                        result = self._run_approx_ks_test(
                            df=prepared_data,
                            column=column,
                            label_column=config["label_column"],
                            error_param=error_param,
                            reliability=config["reliability"],
                            scipy_reference=scipy_results[column],
                            test_value=config["label_test_value"],
                            control_value=config["label_control_value"]
                        )
                        
                        # Записываем в файл
                        f.write(
                            f"{column};ApproxKStest;{error_param};N/A;"
                            f"{iteration};{result['time_sec']:.4f};"
                            f"{result['compression_pct']:.2f};"
                            f"{result['stat_diff']:.6f};"
                            f"{result['pvalue_diff']:.6f};"
                            f"{result['pass_match']}\n"
                        )
                        
                        all_results.append({
                            "column": column,
                            "method": "ApproxKStest",
                            "error_param": error_param,
                            "n_bins": None,
                            "iteration": iteration,
                            **result
                        })
                
                # --- GroupKSTestExtension с разными n_bins ---
                bins_iter = tqdm(
                    config["n_bins_params"], 
                    desc=f"   GroupKSTest [{column}]", 
                    unit="bins",
                    colour='red',
                    leave=False,
                    disable=not show_progress
                )
                
                for n_bins in bins_iter:
                    iter_iter = tqdm(
                        range(config["iterations"]), 
                        desc=f"    Итерации (n_bins={n_bins})", 
                        unit="iter",
                        colour='red',
                        leave=False,
                        disable=not show_progress
                    )
                    
                    for iteration in iter_iter:
                        result = self._run_group_ks_test(
                            df=prepared_data,
                            column=column,
                            label_column=config["label_column"],
                            n_bins=n_bins,
                            reliability=config["reliability"],
                            scipy_reference=scipy_results[column],
                            test_value=config["label_test_value"],
                            control_value=config["label_control_value"]
                        )
                        
                        # Записываем в файл
                        f.write(
                            f"{column};GroupKSTest;N/A;{n_bins};"
                            f"{iteration};{result['time_sec']:.4f};"
                            f"{result['compression_pct']:.2f};"
                            f"{result['stat_diff']:.6f};"
                            f"{result['pvalue_diff']:.6f};"
                            f"{result['pass_match']}\n"
                        )
                        
                        all_results.append({
                            "column": column,
                            "method": "GroupKSTest",
                            "error_param": None,
                            "n_bins": n_bins,
                            "iteration": iteration,
                            **result
                        })
        
        print(f"\n✓ Эксперимент завершен. Результаты сохранены в: {os.path.abspath(config['output_file'])}")
        
        return pd.DataFrame(all_results)
    
    def _split_into_groups(self,
                          data: SparkDataFrame,
                          label_column: str,
                          split_ratio: float,
                          create_label: bool,
                          test_value: int,
                          control_value: int,
                          random_state: int) -> SparkDataFrame:
        """
        Разбивает DataFrame на test/control группы с заданным соотношением.
        """
        np.random.seed(random_state)
        
        if create_label:
            data_with_label = data.withColumn(
                label_column,
                (rand(seed=random_state) < split_ratio).cast("int") * (test_value - control_value) + control_value
            )
            return data_with_label
        else:
            if label_column not in data.columns:
                raise ValueError(f"Колонка '{label_column}' не найдена в DataFrame. "
                               f"Установите create_label=True или укажите существующую колонку.")
            return data
    
    def _run_scipy_baseline(self, 
                           df: SparkDataFrame, 
                           column: str,
                           label_column: str,
                           reliability: float,
                           test_value: int,
                           control_value: int) -> Dict[str, Any]:
        """
        Запуск точного KS-теста через scipy (эталон для сравнения).
        """
        test_data = (df.filter(col(label_column) == test_value)
                    .select(column)
                    .toPandas()[column]
                    .values)
        
        control_data = (df.filter(col(label_column) == control_value)
                       .select(column)
                       .toPandas()[column]
                       .values)
        
        start_time = time.time()
        statistic, p_value = ks_2samp(test_data, control_data, method='asymp')
        elapsed = time.time() - start_time
        
        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "pass": bool(p_value < reliability),
            "test_n": len(test_data),
            "control_n": len(control_data),
            "time_sec": elapsed
        }
    
    def _run_approx_ks_test(self,
                           df: SparkDataFrame,
                           column: str,
                           label_column: str,
                           error_param: float,
                           reliability: float,
                           scipy_reference: Dict[str, Any],
                           test_value: int,
                           control_value: int) -> Dict[str, Any]:
        """
        Запуск ApproxKStest и расчет метрик сравнения со scipy.
        """
        ks_test = ApproxKStest(
            data=df,
            target_columns=column,
            label_column=label_column,
            reliability=reliability,
            error=error_param
        )
        
        start_time = time.time()
        result = ks_test.calculate()
        elapsed = time.time() - start_time
        
        col_result = result[column]
        
        test_compression = (1 - col_result["sketch_test_size"] / col_result["test_n"]) * 100
        control_compression = (1 - col_result["sketch_control_size"] / col_result["control_n"]) * 100
        avg_compression = (test_compression + control_compression) / 2
        
        stat_diff = abs(col_result["statistic"] - scipy_reference["statistic"])
        pvalue_diff = abs(col_result["p-value"] - scipy_reference["p_value"])
        pass_match = col_result["pass"] == scipy_reference["pass"]
        
        return {
            "time_sec": elapsed,
            "compression_pct": avg_compression,
            "stat_diff": stat_diff,
            "pvalue_diff": pvalue_diff,
            "pass_match": pass_match
        }
    
    def _run_group_ks_test(self,
                          df: SparkDataFrame,
                          column: str,
                          label_column: str,
                          n_bins: int,
                          reliability: float,
                          scipy_reference: Dict[str, Any],
                          test_value: int,
                          control_value: int) -> Dict[str, Any]:
        """
        Запуск GroupKSTestExtension и расчет метрик сравнения со scipy.
        """
        ks_test = GroupKSTestExtension(
            data=df,
            target_columns=column,
            label_column=label_column,
            reliability=reliability,
            n_bins=n_bins
        )
        
        start_time = time.time()
        result = ks_test.calculate()
        elapsed = time.time() - start_time
        
        col_result = result[column]
        
        total_n = scipy_reference["test_n"] + scipy_reference["control_n"]
        compression = (1 - n_bins / total_n) * 100
        
        stat_diff = abs(col_result["statistic"] - scipy_reference["statistic"])
        pvalue_diff = abs(col_result["p-value"] - scipy_reference["p_value"])
        pass_match = col_result["pass"] == scipy_reference["pass"]
        
        return {
            "time_sec": elapsed,
            "compression_pct": compression,
            "stat_diff": stat_diff,
            "pvalue_diff": pvalue_diff,
            "pass_match": pass_match
        }


# ============================================================================
# Пример использования из Jupyter Notebook
# ============================================================================

if __name__ == "__main__":
    # Инициализация Spark
    spark = SparkSession.builder \
        .appName("KSExperiment") \
        .master("local[*]") \
        .getOrCreate()
    
    # Пример: создание тестовых данных
    import numpy as np
    import pandas as pd
    
    np.random.seed(42)
    n = 10000
    pdf = pd.DataFrame({
        'feature1': np.concatenate([np.random.normal(0, 1, n), np.random.normal(0.5, 1, n)]),
        'feature2': np.concatenate([np.random.normal(0, 1, n), np.random.normal(0, 1, n)]),
        'feature3': np.concatenate([np.random.normal(0, 1, n), np.random.normal(1, 1, n)]),
    })
    data = spark.createDataFrame(pdf)
    
    # Конфигурация эксперимента
    experiment_config = {
        # "columns": ["feature1", "feature2", "feature3"],
        # "label_column": "label",
        # "group_split_ratio": 0.5,
        # "iterations": 3,
        # "error_params": [0.005, 0.01, 0.05],
        # "n_bins_params": [1000, 2000],
        # "reliability": 0.05,
        # "random_state": 42,
        # "output_file": "ks_experiment_results.csv",
        # "write_header": True,
        # "create_label": True,
        # "label_test_value": 2,
        # "label_control_value": 1,
        # "show_progress": True,  # Включить прогресс-бары
        "columns": [f"feature_{i}" for i in range(1, 91)],

        "label_column": "label",
        "group_split_ratio": 0.5,

        "iterations": 3,  # 3 итерации для стабильности

        # Разные уровни точности
        "error_params": [0.005, 0.01, 0.05],

        # Разные размеры бинов
        "n_bins_params": [1000, 2000, 5000],

        "reliability": 0.05,
        "random_state": 42,
        "output_file": "ks_experiment_full.csv",

        "create_label": True,
        "label_test_value": 2,
        "label_control_value": 1,
        "show_progress": True,
        "write_header": True
    }
    
    # Запуск эксперимента
    experiment = KSExperiment(spark)
    results_df = experiment.run(data=data, config=experiment_config)
    
    # Вывод сводной статистики
    print("\n" + "=" * 80)
    print("СВОДНАЯ СТАТИСТИКА ПО МЕТОДАМ")
    print("=" * 80)
    
    for method in ["ApproxKStest", "GroupKSTest"]:
        method_data = results_df[results_df["method"] == method]
        
        print(f"\n📊 {method}:")
        print(f"  Среднее время: {method_data['time_sec'].mean():.4f} сек")
        print(f"  Среднее сжатие: {method_data['compression_pct'].mean():.2f}%")
        print(f"  Средняя разница статистик: {method_data['stat_diff'].mean():.6f}")
        print(f"  Средняя разница p-value: {method_data['pvalue_diff'].mean():.6f}")
        print(f"  Совпадение решений (pass): {method_data['pass_match'].mean() * 100:.1f}%")
    
    spark.stop()