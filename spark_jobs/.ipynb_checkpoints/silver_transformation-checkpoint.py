import os
import sys
import time
import logging
import traceback
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, struct

# ------------------------------------------------------------------
# LOGGING SETUP (MATCHING PROJECT 2 TEMPLATE)
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("silver_transformation_p3")

# ------------------------------------------------------------------
# ENVIRONMENT VARIABLES
# ------------------------------------------------------------------
ICEBERG_WAREHOUSE = os.getenv("ICEBERG_WAREHOUSE", "s3a://promotionengine-search")
NESSIE_URI = os.getenv("NESSIE_URI", "http://nessie:19120/api/v2")
NESSIE_REF = os.getenv("NESSIE_REF", "main")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "eu-north-1")

JAR_DIR = "/opt/airflow/jars"

# We only need Iceberg/Nessie/AWS jars for this step (No Mongo needed here, but keeping list safe)
JARS = [
    os.path.join(JAR_DIR, "iceberg-spark-runtime-3.4_2.12-1.5.2.jar"),
    os.path.join(JAR_DIR, "iceberg-nessie-1.5.2.jar"),
    os.path.join(JAR_DIR, "nessie-client-0.99.0.jar"),
    os.path.join(JAR_DIR, "nessie-spark-extensions-3.4_2.12-0.105.7.jar"),
    os.path.join(JAR_DIR, "hadoop-aws-3.3.4.jar"),
    os.path.join(JAR_DIR, "aws-java-sdk-bundle-1.12.772.jar"),
]

# INPUT (BRONZE)
SOURCE_TABLE = "nessie.sales.mongo_orders"

# OUTPUT (SILVER)
TARGET_NAMESPACE = "sales_silver"
TARGET_TABLE = "mongo_orders_silver"
TARGET_IDENT = f"nessie.{TARGET_NAMESPACE}.{TARGET_TABLE}"

# ------------------------------------------------------------------
def now():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

# ------------------------------------------------------------------
def create_spark():
    logger.info("Creating SparkSession")
    spark = (
        SparkSession.builder
        .appName("SILVER_TRANSFORMATION_PROJECT3")
        .config("spark.jars", ",".join(JARS))
        .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
        .config("spark.sql.defaultCatalog", "nessie")
        .config("spark.sql.catalog.nessie", "org.apache.iceberg.spark.SparkCatalog")
        .config("spark.sql.catalog.nessie.catalog-impl", "org.apache.iceberg.nessie.NessieCatalog")
        .config("spark.sql.catalog.nessie.uri", NESSIE_URI)
        .config("spark.sql.catalog.nessie.ref", NESSIE_REF)
        .config("spark.sql.catalog.nessie.warehouse", ICEBERG_WAREHOUSE)
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
        logger.info("Reading from BRONZE table: %s", SOURCE_TABLE)
        df_bronze = spark.read.format("iceberg").load(SOURCE_TABLE)
        
        logger.info("Bronze Row Count: %d", df_bronze.count())
        df_bronze.printSchema()

        # -------------------------------------------------------
        # TRANSFORMATION LOGIC
        # -------------------------------------------------------
        logger.info("Applying Silver Transformation (Adding Country: INDIA)...")
        
        # We rebuild the 'shipping_address' struct adding the new column
        df_silver = df_bronze.withColumn("shipping_address", 
            struct(
                col("shipping_address.city"),
                col("shipping_address.state"),
                col("shipping_address.zip"),
                lit("INDIA").alias("country")
            )
        )

        logger.info("Silver Schema Preview:")
        df_silver.printSchema()

        # -------------------------------------------------------
        # WRITE TO SILVER
        # -------------------------------------------------------
        logger.info("Ensuring namespace exists: %s", TARGET_NAMESPACE)
        spark.sql(f"CREATE NAMESPACE IF NOT EXISTS nessie.{TARGET_NAMESPACE}")

        logger.info("Writing to Silver Iceberg Table: %s", TARGET_IDENT)
        df_silver.writeTo(TARGET_IDENT).createOrReplace()

        logger.info("Silver write SUCCESS")

        # -------------------------------------------------------
        # VERIFICATION
        # -------------------------------------------------------
        logger.info("Verifying Silver Data...")
        result = spark.sql(f"SELECT shipping_address FROM {TARGET_IDENT} LIMIT 1").collect()
        logger.info("Sample Record: %s", result)

        logger.info("Stopping Spark")
        spark.stop()
        logger.info("========== JOB SUCCESS at %s ==========", now())

    except Exception as e:
        logger.error("JOB FAILED: %s", str(e))
        logger.error("TRACEBACK:\n%s", traceback.format_exc())
        raise

if __name__ == "__main__":
    main()