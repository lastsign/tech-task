![](Aspose.Words.e99ffdc8-656f-4763-a1e5-a17e371231d8.001.png)![](Aspose.Words.e99ffdc8-656f-4763-a1e5-a17e371231d8.002.jpeg)

**Copy of Intern Test (round #1)**

Первый раунд тестового задания. Проделаем простое упражнения с фит-предиктом моделей, визуализацией и матричной алгеброй.

Задание необходимо выполнить в Jupyter Notebook либо в Google[ Colab](https://colab.research.google.com/notebooks/intro.ipynb)

**Дедлайн: 27.08 (пт) 23:59**

решение и возникающие вопросы [присылать Мне](http://t.me/uberkinder)![](Aspose.Words.e99ffdc8-656f-4763-a1e5-a17e371231d8.003.png)

**Notation**

- **[+1]** Обязательное к выполнению задание
  - **[+2]** *Теоретическая подзадача, ответ написать комментарием в коде*

![](Aspose.Words.e99ffdc8-656f-4763-a1e5-a17e371231d8.004.png) **[+3]** **"задача со (\*)"**, необязательна к выполнению, но даёт дополнительные баллы ![](Aspose.Words.e99ffdc8-656f-4763-a1e5-a17e371231d8.005.png) **[+4]** Не обязательная к выполнению задача на визуализацию данных![](Aspose.Words.e99ffdc8-656f-4763-a1e5-a17e371231d8.006.png)

1. **Dataset**

Будем работать с time-series данными

**[+2]** Загрузите файл  energydata\_complete.csv из![](Aspose.Words.e99ffdc8-656f-4763-a1e5-a17e371231d8.007.png)

[(for simple exercises) Time Series Forecasting This dataset was created for time series exercises. ](https://www.kaggle.com/bulentsiyah/for-simple-exercises-time-series-forecasting?select=energydata_complete.csv)![](Aspose.Words.e99ffdc8-656f-4763-a1e5-a17e371231d8.008.png)![](Aspose.Words.e99ffdc8-656f-4763-a1e5-a17e371231d8.009.jpeg)

![](Aspose.Words.e99ffdc8-656f-4763-a1e5-a17e371231d8.010.png)[ https://www.kaggle.com/bulentsiyah/for-simple-exercises-time-series-forecasting?select=energydata _complete.csv ](https://www.kaggle.com/bulentsiyah/for-simple-exercises-time-series-forecasting?select=energydata_complete.csv)

![](Aspose.Words.e99ffdc8-656f-4763-a1e5-a17e371231d8.011.png) колонка  date – временные метки![](Aspose.Words.e99ffdc8-656f-4763-a1e5-a17e371231d8.012.png)

- переведите её в тип  datetime![](Aspose.Words.e99ffdc8-656f-4763-a1e5-a17e371231d8.013.png)
  - из  date извлечь *weekofyear* в колонку  week ![](Aspose.Words.e99ffdc8-656f-4763-a1e5-a17e371231d8.014.png)![](Aspose.Words.e99ffdc8-656f-4763-a1e5-a17e371231d8.015.png)![](Aspose.Words.e99ffdc8-656f-4763-a1e5-a17e371231d8.016.png) должны получиться недели со 2 по 21

![](Aspose.Words.e99ffdc8-656f-4763-a1e5-a17e371231d8.017.png) оставить недели 3...20

- сделать  date индексом![](Aspose.Words.e99ffdc8-656f-4763-a1e5-a17e371231d8.018.png)
- все остальные колонки воспринимаем как  float ![](Aspose.Words.e99ffdc8-656f-4763-a1e5-a17e371231d8.019.png)![](Aspose.Words.e99ffdc8-656f-4763-a1e5-a17e371231d8.020.png) колонка  Appliances – целевая переменная  ![](Aspose.Words.e99ffdc8-656f-4763-a1e5-a17e371231d8.021.png)y ![](Aspose.Words.e99ffdc8-656f-4763-a1e5-a17e371231d8.022.png)![](Aspose.Words.e99ffdc8-656f-4763-a1e5-a17e371231d8.023.png) всё остальное –  X, т.е. признаки![](Aspose.Words.e99ffdc8-656f-4763-a1e5-a17e371231d8.024.png)

**[+1]** Визуализируйте последнюю неделю  (убедитесь, что датасет отсортирован по дате)

должен получиться следующий ряд:

![](Aspose.Words.e99ffdc8-656f-4763-a1e5-a17e371231d8.025.jpeg)

2. **Validation**

**[+1]** Разобъём датасет на 2 части: 

![](Aspose.Words.e99ffdc8-656f-4763-a1e5-a17e371231d8.026.png) данные за недели 19-20 – тестовая выборка ![](Aspose.Words.e99ffdc8-656f-4763-a1e5-a17e371231d8.027.png) недели 3-18 – обучающая выборка

**[+1]** *Как называется данный вид валидации?*

**Альтернативное задание**

**[+2]** Вместо разбиения на тренировочную и валидационную использовать кросс-валидацию на 5 фолдов

![](Aspose.Words.e99ffdc8-656f-4763-a1e5-a17e371231d8.028.png) *фолд №1 – недели 11-12 для теста, всё, что раньше – трейн ![](Aspose.Words.e99ffdc8-656f-4763-a1e5-a17e371231d8.029.png) фолд №2 – недели 13-14*

![](Aspose.Words.e99ffdc8-656f-4763-a1e5-a17e371231d8.030.png) *... ![](Aspose.Words.e99ffdc8-656f-4763-a1e5-a17e371231d8.031.png) фолд №5 – недели 19-20*

![](Aspose.Words.e99ffdc8-656f-4763-a1e5-a17e371231d8.032.png)

3. **Feature Extraction**

**Gradient Boosting Machine**

**[+2]** Обучим градиентный бустинг (выбрать фреймфорк, с которым привыкли работать – CatBoost, XGBoost, LGBM, H2O и т.д.). Возьмём глубину **D=4**, и количество деревьев **N=200**, остальные параметры по умолчанию / на ваш выбор.

**[+1]** *В чём отличие между leaf-wise tree и depth-wise tree?*

**[+1]** Визуализировать feature importances (оставить топ-10) в виде:

![](Aspose.Words.e99ffdc8-656f-4763-a1e5-a17e371231d8.033.jpeg)

**[+1]** *Как считается feature importance в выбранном вами бустинге?*

**[+1]** Если для валидации выбрана кросс-валидация, мы имеем 5 моделей. У каждой модели своя важность признаков, значит, мы получили некоторое распределение. дополнительно визуализировать его std (черные полоски как на графике ниже)

![](Aspose.Words.e99ffdc8-656f-4763-a1e5-a17e371231d8.034.jpeg)

**Leafs**

![](Aspose.Words.e99ffdc8-656f-4763-a1e5-a17e371231d8.035.jpeg)

**[+1]** *Сколько суммарно листьев у depth-wise tree глубины D? (обозначим ответ за L)*

**[+2]** Для тренировочной и валидационной части мы сделаем предсказания. Но не в виде прогноза целевой переменной, а в виде перевода в новое признаковое пространство.  

Каждый объект, когда прогоняется через бустинг, попадает в конкретный лист в каждом дереве. Если в каждом 

дереве L листьев, тогда всего у бустинга (N \* L) листьев. Таким образом, объект можно представить как вектор из N индексов, где на i-й позиции номер листа, в который попал объект в i-ом дереве. 

Нужно в выбранном фреймворке для бустинга использовать метод, выдающий предсказания в виде индекса листа по каждому дереву (пример на диаграмме)

Например,

- <https://catboost.ai/docs/concepts/python-reference_catboost_calc_leaf_indexes.html>
  - [https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html#lightgbm.LGBMRegressor.predict ](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html#lightgbm.LGBMRegressor.predict) (параметр  pred\_leaf )![](Aspose.Words.e99ffdc8-656f-4763-a1e5-a17e371231d8.036.png)
    - [https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster.predict (параме](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster.predict)тр  pred\_leaf ) ![](Aspose.Words.e99ffdc8-656f-4763-a1e5-a17e371231d8.037.png)

**[+2]** Перевести индексы листьев из пространства  [0...L) (индексы уникальны в рамках дерева) в формат ![](Aspose.Words.e99ffdc8-656f-4763-a1e5-a17e371231d8.038.png)

[0...N\*L) (индексы уникальны в рамках всего ансамбля). ![](Aspose.Words.e99ffdc8-656f-4763-a1e5-a17e371231d8.039.png)

- допустим мы имеем 3 дерева, объект попал во 2-й лист в первом дереве, в 5-й во втором и в 0-й в третьем.
  - на выходе получаем вектор **[2, 5, 0]** (индексы - локальные по каждому дереву)
    - его необходимо перевести в **[2, 1\*L+5, 2\*L+0]** (индексы - глобальные по всему бустингу)
      - пример результата на диаграмме

Вывести первые 10 строк и 10 стобцов старой и новой матрицы индексов (если использована кросс-валидация, по каждому фолду) 

+ размерности старой и новой матрицы индексов 
+ комментарием в ячейке полученные L и N\*L 

**[+2]** предыдущее задание должно быть выполнено с помощью матричной алгебры numpy

**One-Hot Encoding**

**[+2]** Имея глобальные индексы листьев, можем перевести их в пространство нулей и единиц:

- пусть на предыдущем шаге мы получили индексы [1, 4, 7), а всего индексов 8
  - перевести их в формат [0, 1, 0, 0, 1, 0, 0, 1]

Это и будет новым признаковым пространством, в которое мы перевели датасет с помощью градиентного бустинга

4. **Linear Regression**

**LinReg**

**[+1]** поверх полученной матрицы из нулей и единиц обучаем линейную регрессию **[+1]** добавить L1 регуляризацию, λ=0.666

- **[+1]** *какой % признаков отключился?*
  - **[+1]** *в чём разница между L1 и L2 регуляризацией? почему L1 можно использовать для отбора признаков, а L2 нельзя?*

**[+1]** визуализируйте scatter plot весов регрессии для первых 10 деревьев (сами веса + их модули) – должно получиться 10 \* L точек

**[+1]** Визуализировать реальный таргет и его прогноз (последние 100 точек, сортировка по дате). если использована кросс-валидация, то по график нужен для каждого фолда 

**Confidence Bound**

- **[+2]** На веса модели (исключая bias) накинуть случайный шум 10000 раз. например, *w*′ = *wz*, *r* ∼ N (1, 0.2). 
  - **[+1]** с помощью каждого инстанса зашумлённой модели делаем предикт
- **[+1]** *какие квантили отвечают за* нижнюю и верхнюю границу 95%-го доверительного интервала? (LCB и UCB)
  - **[+1]** взять данные LCB и UCB оценку по каждому объекту (таймстемпу) валидационной выборки
    - **[+1]** посчитать % случаев (по каждому фолду), когда реальное значение попадает в область доверительного интервала
      - **[+1]** визуализировать (таргет + прогноз + прозрачным цветом область доверительного интервала). если выбрана кросс-валидация, визуализировать каждый фолд. пример графика:

![](Aspose.Words.e99ffdc8-656f-4763-a1e5-a17e371231d8.040.jpeg)

**Quantile Regression**

В dev версии sklearn добавлена квантильная регрессия

<https://scikit-learn.org/dev/modules/generated/sklearn.linear_model.QuantileRegressor.html>![](Aspose.Words.e99ffdc8-656f-4763-a1e5-a17e371231d8.041.png)

**[+3]** Установить dev-пакет sklearn, обучить **QuantileRegressor** на 1) медиану, 2-3) нижнюю и верхнюю границу 95%- го доверительного интервала

**[+1]** Визуализировать все 3 регрессии аналогично предыдущему заданию

**[+1]** аналогично, посчитать % случаев (по каждому фолду), когда реальное значение попадает в область доверительного интервала
Copy of Intern T est (round #1) PAGE5
