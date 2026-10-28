import os
import sys
# Forzar a Spark a usar el Python de este entorno (spark-env)
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
from pyspark.sql import SparkSession

# Crear la sesión de Spark en modo local usando todos los núcleos
spark = SparkSession.builder \
    .master("local[*]") \
    .appName("TestPySpark") \
    .getOrCreate()

# Crear un DataFrame de prueba
data = [("benign", 1), ("malignant", 2), ("benign", 3)]
columns = ["diagnosis", "id"]
df = spark.createDataFrame(data, columns)

print("\n--- Datos de prueba ---")
df.show()

print(f"Versión de Spark: {spark.version}")
print("¡Entorno configurado correctamente!")

spark.stop()