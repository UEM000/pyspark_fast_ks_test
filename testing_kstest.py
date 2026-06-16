import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from scipy.stats import ks_2samp
from statistic_spark import MyKStest

# Создание Spark сессии
spark = SparkSession.builder \
    .appName("KSTest") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")


def print_sketch_reduction(result_dict: dict):
    """Выводит информацию о том, насколько скетч сжал исходную выборку"""
    print("\n  📊 ЭФФЕКТИВНОСТЬ СЖАТИЯ (SKETCH):")
    for col, res in result_dict.items():
        test_n = res.get("test_n", 0)
        sketch_test = res.get("sketch_test_size", 0)
        control_n = res.get("control_n", 0)
        sketch_control = res.get("sketch_control_size", 0)
        
        if test_n > 0 and sketch_test > 0:
            test_reduction = (1 - sketch_test / test_n) * 100
        else:
            test_reduction = 0.0
            
        if control_n > 0 and sketch_control > 0:
            control_reduction = (1 - sketch_control / control_n) * 100
        else:
            control_reduction = 0.0
            
        print(f"    [{col}]")
        print(f"      Test:     {test_n:>7,} -> {sketch_test:>5,} точек (сжато на {test_reduction:>5.1f}%)")
        print(f"      Control:  {control_n:>7,} -> {sketch_control:>5,} точек (сжато на {control_reduction:>5.1f}%)")


def generate_test_data(n1, n2, mean1=0, mean2=0, std1=1, std2=1, seed=42):
    np.random.seed(seed)
    control = np.random.normal(mean1, std1, n1)
    test = np.random.normal(mean2, std2, n2)
    
    df = pd.DataFrame({
        'value': np.concatenate([control, test]),
        'label': np.concatenate([np.ones(n1), np.full(n2, 2)])
    })
    return spark.createDataFrame(df)


def test_same_distribution():
    print("=" * 80)
    print("ТЕСТ 1: Одинаковые распределения (H0: распределения одинаковы)")
    print("=" * 80)
    
    n = 10000
    df = generate_test_data(n, n, mean1=0, mean2=0, std1=1, std2=1)
    
    ks_test = MyKStest(df, 'value', 'label', reliability=0.05, error=0.01)
    result = ks_test.calculate()
    
    control_data = df.filter(df['label'] == 1).select('value').toPandas()['value'].values
    test_data = df.filter(df['label'] == 2).select('value').toPandas()['value'].values
    scipy_stat, scipy_pvalue = ks_2samp(control_data, test_data, method='asymp')
    
    print(f"\nSpark KS Test:")
    print(f"  Statistic: {result['value']['statistic']:.6f}")
    print(f"  P-value:   {result['value']['p-value']:.6f}")
    print(f"  Reject H0: {result['value']['pass']}")
    
    print(f"\nScipy KS Test:")
    print(f"  Statistic: {scipy_stat:.6f}")
    print(f"  P-value:   {scipy_pvalue:.6f}")
    
    print(f"\nРазница в p-value: {abs(result['value']['p-value'] - scipy_pvalue):.6f}")
    print(f"Разница в statistic: {abs(result['value']['statistic'] - scipy_stat):.6f}")
    
    # НОВЫЙ ВЫВОД
    print_sketch_reduction(result)
    
    assert not result['value']['pass'], "H0 должна быть НЕ отвергнута для одинаковых распределений"
    print("\n✓ ТЕСТ 1 ПРОЙДЕН: H0 корректно не отвергнута")


def test_different_distributions():
    print("\n" + "=" * 80)
    print("ТЕСТ 2: Разные распределения (H0: распределения различны)")
    print("=" * 80)
    
    n = 10000
    df = generate_test_data(n, n, mean1=0, mean2=2, std1=1, std2=1)
    
    ks_test = MyKStest(df, 'value', 'label', reliability=0.05, error=0.01)
    result = ks_test.calculate()
    
    control_data = df.filter(df['label'] == 1).select('value').toPandas()['value'].values
    test_data = df.filter(df['label'] == 2).select('value').toPandas()['value'].values
    scipy_stat, scipy_pvalue = ks_2samp(control_data, test_data, method='asymp')
    
    print(f"\nSpark KS Test:")
    print(f"  Statistic: {result['value']['statistic']:.6f}")
    print(f"  P-value:   {result['value']['p-value']:.6f}")
    print(f"  Reject H0: {result['value']['pass']}")
    
    print(f"\nScipy KS Test:")
    print(f"  Statistic: {scipy_stat:.6f}")
    print(f"  P-value:   {scipy_pvalue:.6f}")
    
    # НОВЫЙ ВЫВОД
    print_sketch_reduction(result)
    
    assert result['value']['pass'], "H0 должна быть отвергнута для разных распределений"
    print("\n✓ ТЕСТ 2 ПРОЙДЕН: H0 корректно отвергнута")


def test_multiple_columns():
    print("\n" + "=" * 80)
    print("ТЕСТ 3: Тестирование нескольких колонок")
    print("=" * 80)
    
    np.random.seed(42)
    n = 5000
    
    df = pd.DataFrame({
        'col1': np.concatenate([np.random.normal(0, 1, n), np.random.normal(0, 1, n)]),
        'col2': np.concatenate([np.random.normal(0, 1, n), np.random.normal(2, 1, n)]),
        'label': np.concatenate([np.ones(n), np.full(n, 2)])
    })
    
    spark_df = spark.createDataFrame(df)
    
    ks_test = MyKStest(spark_df, ['col1', 'col2'], 'label', reliability=0.05, error=0.01)
    result = ks_test.calculate()
    
    print(f"\nКолонка 'col1' (одинаковые распределения):")
    print(f"  P-value: {result['col1']['p-value']:.6f}")
    print(f"  Reject H0: {result['col1']['pass']}")
    
    print(f"\nКолонка 'col2' (разные распределения):")
    print(f"  P-value: {result['col2']['p-value']:.6f}")
    print(f"  Reject H0: {result['col2']['pass']}")
    
    # НОВЫЙ ВЫВОД
    print_sketch_reduction(result)
    
    assert not result['col1']['pass'], "col1: H0 должна быть НЕ отвергнута"
    assert result['col2']['pass'], "col2: H0 должна быть отвергнута"
    print("\n✓ ТЕСТ 3 ПРОЙДЕН: Множественное тестирование работает корректно")


def test_accuracy_comparison():
    print("\n" + "=" * 80)
    print("ТЕСТ 4: Сравнение точности и сжатия на разных размерах выборок")
    print("=" * 80)
    
    sizes = [1000, 5000, 10000, 50000, 100000, 150000, 200000, 250000, 300000]
    
    for n in sizes:
        df = generate_test_data(n, n, mean1=0, mean2=0.5, std1=1, std2=1)
        
        ks_test = MyKStest(df, 'value', 'label', reliability=0.05, error=0.01)
        result = ks_test.calculate()
        
        control_data = df.filter(df['label'] == 1).select('value').toPandas()['value'].values
        test_data = df.filter(df['label'] == 2).select('value').toPandas()['value'].values
        scipy_stat, scipy_pvalue = ks_2samp(control_data, test_data, method='asymp')
         
        stat_diff = abs(result['value']['statistic'] - scipy_stat)
        pval_diff = abs(result['value']['p-value'] - scipy_pvalue)
        
        print(f"\nn={n:6d}:")
        print(f"  Spark:  stat={result['value']['statistic']:.6f}, p={result['value']['p-value']:.6f}")
        print(f"  Scipy:  stat={scipy_stat:.6f}, p={scipy_pvalue:.6f}")
        print(f"  Diff:   stat={stat_diff:.6f}, p={pval_diff:.6f}")
        
        # НОВЫЙ ВЫВОД
        print_sketch_reduction(result)
        
        assert pval_diff < 0.05, f"Разница в p-value слишком велика для n={n}"
    
    print("\n✓ ТЕСТ 4 ПРОЙДЕН: Точность и сжатие корректны на всех размерах")


if __name__ == "__main__":
    try:
        test_same_distribution()
        test_different_distributions()
        test_multiple_columns()
        test_accuracy_comparison()
        
        print("\n" + "=" * 80)
        print("✓ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("=" * 80)
        
    except AssertionError as e:
        print(f"\n✗ ТЕСТ ПРОВАЛЕН: {str(e)}")
        raise
    finally:
        spark.stop()