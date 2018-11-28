## Installation

Required Python Version: 3.7

`pip` installation: `pip install kaggle requests clint numpy scipy`

### Kaggle API Key
This project uses the Kaggle API to download the Human Protein Atlas Image Classification dataset.

You must download by navigating to Kaggle.com > My Account; then find the subheading "API" and click "Create New API Token"

Store this API key at `~/.kaggle/kaggle.json` so that the Kaggle API knows where
to find it.

For a better description on getting Kaggle authenticated, see the following resources:
- https://www.kaggle.com/docs/api#getting-started-installation-&-authentication
- https://github.com/Kaggle/kaggle-api#datasets

## Project Directory Structure
```
├── data          - data directory
│ └── raw         - raw, unprocessed data download from kaggle
│       ├── test  - test images
│       └── train - train images
└── protein-atlas - Python Module for this project
```
