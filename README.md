# heart-attack-risk
Проект по предсказанию рисков сердечных приступов

## Возможности
- исследование и обучение модели в Jupyter Notebook;
- обучение модели из командной строки;
- предсказание для тестового CSV;
- FastAPI сервис для инференса;
- сохранение результата в формате `id,prediction`.

## Структура
См. дерево проекта в корне репозитория.

## Установка
```bash
python -m venv .venv
source .venv/bin/activate  # Linux / Mac
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```
## Для Windows:

.venv\Scripts\activate
pip install -r requirements.txt
"""Core package for heart attack risk prediction project."""

## Обучение модели
```python scripts/train.py --train-path data/heart_train.csv --model-path models/model.cbm --metadata-path models/metadata.json ```
## Предсказание
``` python scripts/predict.py --test-path data/heart_test.csv --model-path models/model.cbm --metadata-path models/metadata.json --output-path data/predictions.csv ```

## Запуск API

```uvicorn api.main:app --reload```

## Пример запроса
```curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "csv_path": "data/heart_test.csv",
    "output_path": "data/predictions.csv"
  }'
```
## Формат ответа

``{
  "status": "ok",
  "rows": 966,
  "output_path": "data/predictions.csv"
}

---
