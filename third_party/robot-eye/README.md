# robot-eye
A python package centered around a ego-centric robot and using computer vision and multi-view geometry to allow for intuitive processing of the world through different camera in different coordinate frames

### Set configuration for poetry
```bash
pip install poetry
poetry config virtualenvs.in-project true # puts .venv in the current directory
poetry lock --no-update 
poetry install
```
poetry lock reason: https://stackoverflow.com/questions/76327419/valueerror-libcublas-so-0-9-not-found-in-the-system-path

### Starting from scratch if necessary
```bash
poetry env list  # shows the name of the current environment
poetry env remove <current environment>
poetry install  # will create a new environment using your updated configuration
```