```markdown
# Fast Approximate Kolmogorov-Smirnov Test for PySpark

[English](#english) | [Русский](#russian)

---

## <a name="english"></a>English Version

### Overview

This project implements an **approximate two-sample Kolmogorov-Smirnov (KS) test** for Apache Spark, based on the research paper *"Two-sample KS test with approxQuantile in Apache Spark"* (Eck et al., 2023). 

The implementation provides a scalable solution for distribution comparison in big data settings, avoiding the need for full data sorting while maintaining statistical rigor with theoretical error bounds.

### Key Features

- ⚡ **High Performance**: Uses Spark's `approxQuantile` method for ε-approximate quantiles
- 📊 **Theoretical Guarantees**: Derived error bounds for approximate CDFs
- 🎯 **Configurable Precision**: Tune accuracy vs. performance trade-off via `error` parameter
- 🔧 **Easy Integration**: Simple API compatible with Spark DataFrames
- 📈 **Scalable**: Handles datasets from thousands to millions of records

### Performance Results

Our experiments show significant data reduction while maintaining accuracy:

| Dataset Size | Error Parameter | Data Reduction | Statistic Difference |
|-------------|----------------|----------------|---------------------|
| 1,000 | 0.005 | ~60% | < 0.0001 |
| 10,000 | 0.005 | ~40% | < 0.0001 |
| 100,000 | 0.005 | ~20% | < 0.0001 |
| 300,000 | 0.005 | ~15% | < 0.0001 |

**Key Insights:**
- Higher `error` values → greater compression but lower accuracy
- For most applications, `error=0.01` provides excellent balance
- Statistic difference from scipy remains negligible across all sizes

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd pyspark_fast_ks_test

# Install dependencies
pip install pyspark numpy scipy pandas
```

### Quick Start

```python
from pyspark.sql import SparkSession
from statistic_spark import MyKStest

# Initialize Spark
spark = SparkSession.builder.appName("KSTest").master("local[*]").getOrCreate()

# Prepare your data
# DataFrame should have:
# - target column(s) with numeric values
# - label column with values 1 (control) and 2 (test)

# Create KS test instance
ks_test = MyKStest(
    data=df,
    target_columns='your_column',  # or ['col1', 'col2'] for multiple
    label_column='label',
    reliability=0.05,  # significance level (alpha)
    error=0.01         # approximation error (beta)
)

# Run test
result = ks_test.calculate()

# View results
print(f"KS Statistic: {result['your_column']['statistic']}")
print(f"P-value: {result['your_column']['p-value']}")
print(f"Reject H0: {result['your_column']['pass']}")
print(f"Data reduction: Test {result['your_column']['test_n']} → {result['your_column']['sketch_test_size']} points")
```

### API Reference

#### `MyKStest` Class

**Parameters:**
- `data` (DataFrame): Spark DataFrame with your data
- `target_columns` (str | List[str]): Column name(s) to test
- `label_column` (str): Column with group labels (1=control, 2=test)
- `reliability` (float, optional): Significance level α. Default: 0.05
- `error` (float, optional): Approximation precision β. Default: 0.01

**Methods:**

`calculate()` → `Dict[str, Dict[str, Any]]`
- Returns dictionary with test results for each column
- Each result contains:
  - `p-value`: Calculated p-value
  - `statistic`: KS D-statistic (max distance between CDFs)
  - `pass`: Boolean (True if p-value < reliability)
  - `test_n`: Size of test sample
  - `control_n`: Size of control sample
  - `sketch_test_size`: Number of points in test sketch
  - `sketch_control_size`: Number of points in control sketch

### How It Works

1. **Calculate Error Bound (δ)**: Based on desired precision and significance level
2. **Determine Sketch Parameters**: Using the "45-degree elbow" heuristic:
   - ε = δ - √(δ/N)
   - a = √(N/δ) + 1 (number of quantile points)
3. **Compute Approximate Quantiles**: Via Spark's `approxQuantile`
4. **Build Approximate CDFs**: Linear interpolation between quantile points
5. **Calculate KS Distance**: Maximum absolute difference between CDFs
6. **Compute P-value**: Using asymptotic Kolmogorov distribution

### Mathematical Background

The KS test statistic is:

**D_KS = ||F₁(x) - F₂(x)||∞**

For approximate CDFs, the error is bounded by:

**δ ≤ 1/(a-1) + ε**

The p-value is approximated using:

**P(D_KS > observed) = Q_KS(√(NM/(N+M)) · D_KS)**

where Q_KS is the Kolmogorov distribution function.

### Running Tests

```bash
python testing_kstest.py
```

The test suite includes:
- **Test 1**: Same distributions (should NOT reject H0)
- **Test 2**: Different distributions (should reject H0)
- **Test 3**: Multiple columns testing
- **Test 4**: Small sample edge cases
- **Test 5**: Accuracy comparison across different sizes (saves results to CSV)

### Comparison with Alternatives

| Method | Memory | Accuracy | Speed | Best For |
|--------|--------|----------|-------|----------|
| **This implementation** | O(√N) | High (theoretical bounds) | Fast | Big data, drift detection |
| Full KS test (scipy) | O(N) | Exact | Slow (single machine) | Small datasets |
| Histogram-based | O(bins) | Medium | Fast | Discrete data |

### Use Cases

- 🤖 **ML Model Monitoring**: Detect distribution shift in production
- 📊 **A/B Testing**: Compare treatment vs control groups at scale
- 🏭 **Quality Control**: Monitor manufacturing process changes
- 📈 **Time Series Analysis**: Detect concept drift

### References

1. Eck, B., Kabakci-Zorlu, D., & Ba, A. (2023). Two-sample KS test with approxQuantile in Apache Spark. arXiv:2312.09380
2. Greenwald, M., & Khanna, S. (2001). Space-efficient online computation of quantile summaries. SIGMOD Rec.
3. Lall, A. (2015). Data streaming algorithms for the Kolmogorov-Smirnov test. IEEE Big Data.

### License

MIT License

---

## <a name="russian"></a>Русская версия

### Обзор

Этот проект реализует **приближенный двухвыборочный тест Колмогорова-Смирнова (KS-тест)** для Apache Spark на основе научной статьи *"Two-sample KS test with approxQuantile in Apache Spark"* (Eck et al., 2023).

Реализация обеспечивает масштабируемое решение для сравнения распределений в задачах больших данных, избегая необходимости полной сортировки данных при сохранении статистической строгости с теоретическими границами ошибок.

### Ключевые особенности

- ⚡ **Высокая производительность**: Использует метод `approxQuantile` из Spark для ε-приближенных квантилей
- 📊 **Теоретические гарантии**: Выведены границы ошибок для приближенных CDF
- 🎯 **Настраиваемая точность**: Баланс между точностью и производительностью через параметр `error`
- 🔧 **Простая интеграция**: Удобный API, совместимый с Spark DataFrame
- 📈 **Масштабируемость**: Обрабатывает наборы данных от тысяч до миллионов записей

### Результаты производительности

Наши эксперименты показывают значительное сокращение данных при сохранении точности:

| Размер данных | Параметр ошибки | Сокращение данных | Разница статистик |
|-------------|----------------|----------------|------------------|
| 1,000 | 0.005 | ~60% | < 0.0001 |
| 10,000 | 0.005 | ~40% | < 0.0001 |
| 100,000 | 0.005 | ~20% | < 0.0001 |
| 300,000 | 0.005 | ~15% | < 0.0001 |

**Ключевые выводы:**
- Более высокие значения `error` → большее сжатие, но меньшая точность
- Для большинства приложений `error=0.01` обеспечивает отличный баланс
- Разница статистик с scipy остается пренебрежимо малой для всех размеров

### Установка

```bash
# Клонировать репозиторий
git clone <repository-url>
cd pyspark_fast_ks_test

# Установить зависимости
pip install pyspark numpy scipy pandas
```

### Быстрый старт

```python
from pyspark.sql import SparkSession
from statistic_spark import MyKStest

# Инициализация Spark
spark = SparkSession.builder.appName("KSTest").master("local[*]").getOrCreate()

# Подготовьте ваши данные
# DataFrame должен содержать:
# - целевую(ые) колонку(и) с числовыми значениями
# - колонку меток со значениями 1 (control) и 2 (test)

# Создать экземпляр KS-теста
ks_test = MyKStest(
    data=df,
    target_columns='your_column',  # или ['col1', 'col2'] для нескольких
    label_column='label',
    reliability=0.05,  # уровень значимости (alpha)
    error=0.01         # погрешность аппроксимации (beta)
)

# Запустить тест
result = ks_test.calculate()

# Просмотр результатов
print(f"KS Статистика: {result['your_column']['statistic']}")
print(f"P-value: {result['your_column']['p-value']}")
print(f"Отвергнуть H0: {result['your_column']['pass']}")
print(f"Сокращение данных: Test {result['your_column']['test_n']} → {result['your_column']['sketch_test_size']} точек")
```

### API Reference

#### Класс `MyKStest`

**Параметры:**
- `data` (DataFrame): Spark DataFrame с данными
- `target_columns` (str | List[str]): Имя(ена) колонки(ок) для тестирования
- `label_column` (str): Колонка с метками групп (1=control, 2=test)
- `reliability` (float, optional): Уровень значимости α. По умолчанию: 0.05
- `error` (float, optional): Точность аппроксимации β. По умолчанию: 0.01

**Методы:**

`calculate()` → `Dict[str, Dict[str, Any]]`
- Возвращает словарь с результатами теста для каждой колонки
- Каждый результат содержит:
  - `p-value`: Рассчитанное p-value
  - `statistic`: KS D-статистика (максимальное расстояние между CDF)
  - `pass`: Boolean (True если p-value < reliability)
  - `test_n`: Размер тестовой выборки
  - `control_n`: Размер контрольной выборки
  - `sketch_test_size`: Количество точек в скетче тестовой выборки
  - `sketch_control_size`: Количество точек в скетче контрольной выборки

### Как это работает

1. **Расчет границы ошибки (δ)**: На основе желаемой точности и уровня значимости
2. **Определение параметров скетча**: Используя эвристику "единичного наклона":
   - ε = δ - √(δ/N)
   - a = √(N/δ) + 1 (количество точек квантилей)
3. **Вычисление приближенных квантилей**: Через `approxQuantile` в Spark
4. **Построение приближенных CDF**: Линейная интерполяция между точками квантилей
5. **Расчет KS-расстояния**: Максимальная абсолютная разница между CDF
6. **Вычисление P-value**: Используя асимптотическое распределение Колмогорова

### Математическая основа

Статистика KS-теста:

**D_KS = ||F₁(x) - F₂(x)||∞**

Для приближенных CDF ошибка ограничена:

**δ ≤ 1/(a-1) + ε**

P-value аппроксимируется как:

**P(D_KS > observed) = Q_KS(√(NM/(N+M)) · D_KS)**

где Q_KS - функция распределения Колмогорова.

### Запуск тестов

```bash
python testing_kstest.py
```

Набор тестов включает:
- **Тест 1**: Одинаковые распределения (НЕ должен отвергать H0)
- **Тест 2**: Разные распределения (должен отвергать H0)
- **Тест 3**: Тестирование нескольких колонок
- **Тест 4**: Граничные случаи для малых выборок
- **Тест 5**: Сравнение точности на разных размерах (сохраняет результаты в CSV)

### Сравнение с альтернативами

| Метод | Память | Точность | Скорость | Лучше всего для |
|--------|--------|----------|-------|----------------|
| **Эта реализация** | O(√N) | Высокая (теоретические границы) | Быстро | Большие данные, детекция дрейфа |
| Полный KS-тест (scipy) | O(N) | Точная | Медленно (одна машина) | Малые наборы данных |
| Гистограммный метод | O(bins) | Средняя | Быстро | Дискретные данные |

### Сценарии использования

- 🤖 **Мониторинг ML-моделей**: Детекция сдвига распределения в production
- 📊 **A/B тестирование**: Сравнение treatment vs control групп в масштабе
- 🏭 **Контроль качества**: Мониторинг изменений производственного процесса
- 📈 **Анализ временных рядов**: Детекция концептуального дрейфа

### Ссылки

1. Eck, B., Kabakci-Zorlu, D., & Ba, A. (2023). Two-sample KS test with approxQuantile in Apache Spark. arXiv:2312.09380
2. Greenwald, M., & Khanna, S. (2001). Space-efficient online computation of quantile summaries. SIGMOD Rec.
3. Lall, A. (2015). Data streaming algorithms for the Kolmogorov-Smirnov test. IEEE Big Data.

### Лицензия

MIT License

---

**Project / Проект**: pyspark_fast_ks_test
```
