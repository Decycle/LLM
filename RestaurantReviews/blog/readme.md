# Uses Dash to generate a dynamic blog with figures

Run the following commands in your terminal to create a virtual environment and install the required packages:

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Then, run the following command to start the Dash app:

```bash
python -m gunicorn app:server
```
