# SRG
This is a pytorch implementation of the paper: Super Resolution Graph with Conditional Normalizing Flows for Temporal Link Prediction. The baselines used in the paper will be released as a toolbox soon. 

# Requirements
torch (1.10.1)

numpy (1.21.6)
# Prepare the data
unzip the data.zip into ./data;

for other dataset, use ./generate_data/preprocess_LR.py to split the original data, and the settings including node_num are setted in the  ./generate_data/baseconfig.yml.

# Run
```python run.py```
