import os
import sys
import time
import logging
import traceback
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as _sum, count

# ------------------------------------------------------------------
# LOGGING SETUP
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("gold_reporting_p3")

# ------------------------------------------------------------------
# ENVIRONMENT VARIABLES
# ------------------------------------------------------------------
# Nessie/Iceberg Config
ICEBERG_WAREHOUSE = os.getenv("ICEBERG_WAREHOUSE", "s3a://promotionengine-search")
NESSIE_URI = os.getenv("NESSIE_URI", "http://nessie:19120/api/v2")
NESSIE_REF = os.getenv("NESSIE_REF", "main")

# AWS Config
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "eu-north-1")

# Postgres Config (Destination)
# We use the internal container name "postgres"
PG_URL = "jdbc:postgresql://postgres:5432/airflow"
PG_USER = "airflow"
PG_PASS = "airflow"
PG_TABLE = "public.city_sales_report"

JAR_DIR = "/opt/airflow/jars"

# ADDED POSTGRES JAR HERE
JARS = [
    os.path.join(JAR_DIR, "iceberg-spark-runtime-3.4_2.12-1.5.2.jar"),
    os.path.join(JAR_DIR, "iceberg-nessie-1.5.2.jar"),
    os.path.join(JAR_DIR, "nessie-client-0.99.0.jar"),
    os.path.join(JAR_DIR, "nessie-spark-extensions-3.4_2.12-0.105.7.jar"),
    os.path.join(JAR_DIR, "hadoop-aws-3.3.4.jar"),
    os.path.join(JAR_DIR, "aws-java-sdk-bundle-1.12.772.jar"),
    # NEW JAR
    os.path.join(JAR_DIR, "postgresql-42.7.3.jar")
]

# INPUT (SILVER)
SOURCE_TABLE = "nessie.sales_silver.mongo_orders_silver"

# ------------------------------------------------------------------
def now():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

# ------------------------------------------------------------------
def create_spark():
    logger.info("Creating SparkSession")
    spark = (
        SparkSession.builder
        .appName("GOLD_REPORTING_PROJECT3")
        .config("spark.jars", ",".join(JARS))
        
        # Iceberg Config
        .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
        .config("spark.sql.defaultCatalog", "nessie")
        .config("spark.sql.catalog.nessie", "org.apache.iceberg.spark.SparkCatalog")
        .config("spark.sql.catalog.nessie.catalog-impl", "org.apache.iceberg.nessie.NessieCatalog")
        .config("spark.sql.catalog.nessie.uri", NESSIE_URI)
        .config("spark.sql.catalog.nessie.ref", NESSIE_REF)
        .config("spark.sql.catalog.nessie.warehouse", ICEBERG_WAREHOUSE)
        
        # AWS Config
        .config("spark.hadoop.fs.s3a.access.key", AWS_ACCESS_KEY_ID)
        .config("spark.hadoop.fs.s3a.secret.key", AWS_SECRET_ACCESS_KEY)
        .config("spark.hadoop.fs.s3a.endpoint.region", AWS_REGION)
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        
        .getOrCreate()
    )
    return spark

# ------------------------------------------------------------------
def main():
    logger.info("========== JOB STARTED at %s ==========", now())

    spark = create_spark()

    try:
        logger.info("Reading SILVER table: %s", SOURCE_TABLE)
        df_silver = spark.read.format("iceberg").load(SOURCE_TABLE)
        
        # -------------------------------------------------------
        # GOLD AGGREGATION LOGIC
        # -------------------------------------------------------
        logger.info("Aggregating Sales by City and Country...")
        
        # We access the struct fields using dot notation
        df_gold = (
            df_silver
            .groupBy(
                col("shipping_address.city").alias("city"),
                col("shipping_address.country").alias("country")
            )
            .agg(
                _sum("total_amount").alias("total_revenue"),
                count("order_id").alias("order_count")
            )
            .orderBy(col("total_revenue").desc())
        )

        logger.info("Gold Report Preview:")
        df_gold.show()

        # -------------------------------------------------------
        # WRITE TO POSTGRES
        # -------------------------------------------------------
        logger.info("Writing to Postgres Table: %s", PG_TABLE)
        
        (
            df_gold.write
            .format("jdbc")
            .option("url", PG_URL)
            .option("dbtable", PG_TABLE)
            .option("user", PG_USER)
            .option("password", PG_PASS)
            .option("driver", "org.postgresql.Driver")
            .mode("overwrite")  # Replaces the table each time
            .save()
        )

        logger.info("Postgres write SUCCESS")

        logger.info("Stopping Spark")
        spark.stop()
        logger.info("========== JOB SUCCESS at %s ==========", now())

    except Exception as e:
        logger.error("JOB FAILED: %s", str(e))
        logger.error("TRACEBACK:\n%s", traceback.format_exc())
        raise

if __name__ == "__main__":
    main()