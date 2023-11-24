FROM python:3.8

# Create app directory
WORKDIR /app

# Install app dependencies
COPY ./requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt

# Bundle app source
COPY . /app

EXPOSE 5000

ENTRYPOINT [ "flask", "--app", "app:create_app()", "run","--host","0.0.0.0","--port","5000"]