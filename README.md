Этот проект обучает модель CatBoost и предоставляет REST API на FastAPI для получения предсказаний по реальным признакам визита/хита. Скрипт обучения формирует целевую переменную, удаляет утечки, обучает модель с GPU, выводит метрики и важности, затем сохраняет модель. API загружает модель и принимает JSON с теми же признаками, возвращая класс и вероятность.
Training script details (script_diploma.py)
Purpose and flow
Загрузка данных: читает prikoluha.csv.

Цель: бинарный таргет из event_action == "sub_submit_success".

Категориальные признаки: строго перечислены и задаются CatBoost (без событийных полей, чтобы исключить утечки).

Очистка: каст к строке и заполнение unknown.

Анти‑утечки: удаление event_action, event_category, event_label, рефереры и служебных столбцов.

Сплит: стратифицированный train_test_split.

Обучение: CatBoostClassifier с GPU, ранней остановкой, class_weights.

Оценка: ROC‑AUC, confusion matrix, classification report, feature importance.

Сохранение: модель в loan_catboost.pkl, важности в feature_importance.csv.
API
Загружает loan_catboost.pkl, принимает JSON с признаками одного наблюдения, проводит предобработку (каст к строке для категорий, проставление unknown), подаёт в модель, возвращает класс и вероятность.
