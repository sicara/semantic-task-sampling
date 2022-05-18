black:
		black src st_scripts streamlit_app.py

dev-install:
		pip install -r dev_requirements.txt

soft-exp-clean:
		dvc exp gc -w
		dvc gc -w
