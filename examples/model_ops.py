import pandas as pd

from gretel_client import get_project
from gretel_client.config import RunnerMode

project = get_project(create=True)

# create a synthetic model using a default synthetic config from
#   https://github.com/gretelai/gretel-blueprints/blob/main/config_templates/gretel/synthetics/default.yml
model = project.create_model(model_config="synthetics/default")

# this will override the datasource set from the template or blueprint
model.data_source = "https://gretel-public-website.s3.us-west-2.amazonaws.com/datasets/USAdultIncome5k.csv"

# submit the model to Gretel Cloud for training
model.create(runner_mode=RunnerMode.CLOUD, upload_data_source=True)

# read out a preview data from the synthetic model
pd.read_csv(model.get_artifact_link("data_preview"), compression='gzip')
