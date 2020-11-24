from flask import Flask


app = Flask(__name__)


@app.route("/")
def runpipeline():
    return "ran the job"


if __name__ == "__main__":
    app.run(port=5000)
