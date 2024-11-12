FROM python:3.11

# Устанавливаем рабочую директорию
WORKDIR /workdir

COPY requirements.txt /workdir/
COPY myapp/ /workdir/myapp/
COPY static/ /workdir/static/
COPY data/ /workdir/data/  

RUN pip install -r requirements.txt

# Запускаем приложение
CMD ["uvicorn", "myapp.main:app", "--host", "0.0.0.0", "--port", "8000"]
