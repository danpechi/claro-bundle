.PHONY: setup deploy run

# Full setup from scratch in a new workspace:
#   1. bundle deploy  — uploads files + creates the setup job + app resource
#   2. bundle run     — registers the ML model and kicks off endpoint deployment
#   3. bundle run app — starts the Streamlit app
#
# The emotion endpoint takes ~10-15 min to become READY after step 2.

setup: deploy run

deploy:
	databricks bundle deploy

run:
	databricks bundle run claro_setup
	databricks bundle run claro_app
