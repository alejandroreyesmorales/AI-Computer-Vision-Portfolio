# Clinical Data Classification with PySpark

End-to-end Machine Learning pipeline using PySpark MLlib in local mode, developed entirely in VS Code. This project focuses on distributed data processing, feature engineering, and model tuning for tabular clinical data.

## Environment Setup

- Python 3.10+
- Java 17 (OpenJDK)
- PySpark installed via `conda` and `pip`
- Spark runs locally using all available CPU cores (`local[*]`)

### Note on Windows Compatibility
To ensure Spark uses the correct Python environment (and not the Windows Store alias), the following environment variables are set at the start of every script:
```python
import os
import sys
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable