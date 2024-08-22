# Azure Machine Learning - Monitoring Demo - August 2024

Sample notebooks that facilitate the following operations
- Load data from open-source datasets and store in Delta format in ADLS Gen2 (Databricks)
- Retrieve Delta data from ADLS Gen2 and use to train & register a custom ML model (MLflow)
- Develop a custom inferencing environment/container based on a base Azure ML inferencing image and ML model dependencies. Push to target registry.
- Deploy model to a managed online endpoint (using custom docker image) with model data capture configured
- Set up out-of-box and advanced model montioring to identify data quality issues with inferencing data, and identify drift away from serving data.
- Send data to the endpoint for capture and testing with monitoring capabilities.