# simple_RNN_predictor

This project implements a Recurrent Neural Network model using PyTorch for predicting a simple time series sequence.
It also supports inference through an UI developed using Streamlit and a backend web service developed using flask.

If you face any problems during setup steps detailed below, you can reach out to me at `deeraj.nachimuthu@gmail.com` 

Clone this project and go to the root directory of the project.

## Assumptions

- Predictions will be requested for months in the future with respect to the end date in the data i.e., 12/31/2021
- The predictions will be made for a single month for each request i.e., there are no endpoints to predict for a year / x predictions from y date. The way the code is written, it is easy to enable these endpoints but from reading the assignment, assumed that this was the expected behavior. To get an year of predictions, we can call the `predict_yy_mm` endpoint repeatedly with different month values.
- When providing past data points, it is expected to provide 15 values because the model is trained with this sequence length. The window was selected after several trial and errors, with minimal changes the model can be retrained to handle different window size.

## Starting the backend prediction service

There are multiple ways the service can be started. If you face a problem with one approach, use the other one. Once the web service is up, move to the next section to run the UI application. 

**Note: all these methods start the model by loading the pretrained weights stored in the file `weights_only_2_16_15.pth`. If you want the model to be trained on startup / while running the program, just delete this file from the project and follow the steps.**

### Running the application through Docker

---------------
#### Prerequisites:
- Docker

1. From the root directory of the project, build the docker image `docker build -t backend_flask_image .`
2. Start a container with the image `docker run -p 5000:5000 --name backend_flask_app backend_flask_image`
3. To verify if the service is working, use the curl commands in the file `curl_win_commands.txt` (this is formatted for windows cmd prompts, edit as necessary for other terminals) or alternatively create a similar request in postman or other tools. The server should respond with results for the requested month.

### Running the application through flask directly

---------------
If you prefer to not use Docker to start the service, you can start the flask application locally.

#### Prerequisites:
- Install the dependencies in requirements.txt using `pip install -r requirements.txt`

1. From the root directory of the project, run the flask app `flask --app "app:create_app()" run --port 5000`
2. To verify if the service is working, use the curl commands in the file `curl_win_commands.txt` (this is formatted for windows cmd prompts, edit as necessary for other terminals) or alternatively create a similar request in postman or other tools. The server should repond with results for the requested month.

### If both approaches failed do this

---------------
Run the model training script directly

#### Prerequisites:
- Install the dependencies in requirements.txt using `pip install -r requirements.txt`

1. From the root directory of the project, run the file lstmregr.py using `python lstmregr.py`.
2. The script should tune the model and create a weights file, which can be manually loaded and used for inference as shown in `demo.ipynb`

## Starting the frontend application

Note:
- If you weren't successful in creating the flask app, then you can't use the UI because it depends on the web service. The results in this case can be found in the `demo.ipynb` notebook (you can try more operations in the notebook).
- If you were successful in creating the flask app, but don't want to install the below prerequisites then the results are the responses from the curl commands.

#### Prerequisites:
- Streamlit and altair packages, install using `pip install streamlit`, `pip install altair`

1. From the root directory of the project, run the application using `streamlit run front_end_app.py`
2. If successful, a browser window should open where you can interact with the service by entering "year",  "month", "past data" (Optional: at least 15 values must be entered as comma separated values to make predictions for the "year" and "month" requested.)


