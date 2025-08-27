# ASL Auto-QC and CBF Estimator

## Overview
This service runs automated quality checks and cerebral blood flow estimation on arterial spin labeling data using existing tools. It outputs summary reports for each region.

## Setup
Install dependencies using:

```bash
pip install -r requirements.txt
```

Run the development server with:

```bash
uvicorn app.main:app --reload
```

## Test Data
A sample dataset is provided in the `data` folder to help you get started with input formats. You can replace it with your own data.