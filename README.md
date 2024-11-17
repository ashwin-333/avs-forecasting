# Pedestrian Detection with Event Camera Data
```
Using PEDRo events dataset: https://zenodo.org/records/13331985
```

## Project Structure

```plaintext
PEDRo-dataset/
├── numpy/               # Contains numpy files for processed data
├── xml/                 # XML files for annotations
├── yolo/                # YOLO-specific files and configurations
src/
├── baseline-model/       # Baseline model implementation
│   ├── main.py           # Main script to run the baseline model
│   ├── train.py          # Script for training the baseline model
│   ├── preprocessing.py  # Script for data preprocessing for baseline model
│   ├── model.py          # Baseline model architecture
├── spike-yolo/           # SpikeYOLO implementation
│   ├── main.py           # Main script to run the SpikeYOLO model
│   ├── preprocessing.py  # Script for data preprocessing for SpikeYOLO
README.md                # Project documentation
