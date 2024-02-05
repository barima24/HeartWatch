from flask import Flask,request,render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging

## creating a flask web application instance
app = Flask(__name__)

##Define a route to the home page

@app.route("/" ,methods = ["GET","POST"])
def home():
    if request.method == "POST":
        try:
            data=CustomData(
                age=int(request.form.get("age")),
                sex=int(request.form.get("sex")),
                chest_pain_type=int(request.form.get("chest_pain_type")),
                cholestrol=int(request.form.get("cholestrol")),
                resting_blood_pressure=int(request.form.get("resting_blood_pressure")),
                fasting_blood_sugar=int(request.form.get("fasting_blood_sugar")),
                resting_electrocardiogram=int(request.form.get("resting_electrocardiogram")),
                max_heart_rate_achieved=int(request.form.get("max_heart_rate_achieved")),
                exercise_induced_angina=int(request.form.get("exercise_induced_angina")),
                st_depression=float(request.form.get("st_depression")),
                st_slope=int(request.form.get("st_slope")),
                num_major_vessels=int(request.form.get("num_major_vessels")),
                thalassemia=int(request.form.get("thalassemia"))
            )
            logging.info("data is submitted")

            final_data = data.get_data_as_dataframe()
            print("Final data for prediction:", final_data)
            ## make prediction
            predict_pipeline = PredictPipeline()
            pred = predict_pipeline.predict(final_data)
            result = round(pred[0],2)
            logging.info("result is predicted")
            return render_template("result.html",final_result=result)

        except Exception as e:
            error_message = f"Error during prediction: {str(e)}"
            return render_template("error.html", error_message=error_message)
    else:
        return render_template("index.html")

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080, debug=True)


