black:
		black src st_scripts 🎷_Hello.py pages

make install:
		sudo apt-get install graphviz graphviz-dev libpython3.8-dev
		pip install -r dev_requirements.txt;

soft-exp-clean:
		dvc exp gc -w
		dvc gc -w
