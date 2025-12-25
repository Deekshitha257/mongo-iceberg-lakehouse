import os
import sys
import time
import logging
import traceback

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

# ------------------------------------------------------------------
# LOGGING SETUP
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("bronze_to_silver_p2")

# ------------------------------------------------------------------
# ENVIRONMENT VARIABLES
# ------------------------------------------------------------------
ICEBERG_WAREHOUSE = os.getenv("ICEBERG_WAREHOUSE")
NESSIE_URI = os.getenv("NESSIE_URI")
NESSIE_REF = os.getenv("NESSIE_REF", "main")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

JAR_DIR = "/opt/airflow/jars"

JARS = [
    os.path.join(JAR_DIR, "iceberg-spark-runtime-3.5_2.12-1.5.2.jar"),
    os.path.join(JAR_DIR, "iceberg-nessie-1.5.2.jar"),
    os.path.join(JAR_DIR, "nessie-client-0.99.0.jar"),
    os.path.join(JAR_DIR, "nessie-spark-extensions-3.4_2.12-0.105.7.jar"),
    os.path.join(JAR_DIR, "hadoop-aws-3.3.4.jar"),
    os.path.join(JAR_DIR, "aws-java-sdk-bundle-1.12.772.jar"),
]

BRONZE_TABLE = "nessie.sales.sample_sales"
SILVER_TABLE = "nessie.sales.sample_sales_silver"

# ------------------------------------------------------------------
def now():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

# ------------------------------------------------------------------
def check_jars():
    logger.info("Checking Spark JARs...")
    missing = []
    for jar in JARS:
        exists = os.path.exists(jar)
        size = os.path.getsize(jar) if exists else None
        logger.info("JAR=%s exists=%s size=%s", jar, exists, size)
        if not exists:
            missing.append(jar)
    return missing

# ------------------------------------------------------------------
def create_spark():
    logger.info("Creating SparkSession (Silver layer)")

    spark = (
        SparkSession.builder
        .appName("BRONZE_TO_SILVER_PROJECT2")
        .config("spark.jars", ",".join(JARS))

        # Iceberg + Nessie
        .config(
            "spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions"
        )
        .config("spark.sql.defaultCatalog", "nessie")
        .config("spark.sql.catalog.nessie", "org.apache.iceberg.spark.SparkCatalog")
        .config(
            "spark.sql.catalog.nessie.catalog-impl",
            "org.apache.iceberg.nessie.NessieCatalog"
        )
        .config("spark.sql.catalog.nessie.uri", NESSIE_URI)
        .config("spark.sql.catalog.nessie.ref", NESSIE_REF)
        .config("spark.sql.catalog.nessie.warehouse", ICEBERG_WAREHOUSE)

        # AWS S3
        .config("spark.hadoop.fs.s3a.access.key", AWS_ACCESS_KEY_ID)
        .config("spark.hadoop.fs.s3a.secret.key", AWS_SECRET_ACCESS_KEY)
        .config("spark.hadoop.fs.s3a.endpoint.region", AWS_REGION)
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider"
        )

        .getOrCreate()
    )

    logger.info("Spark version=%s", spark.version)
    return spark

# ------------------------------------------------------------------
def main():
    logger.info("========== SILVER JOB STARTED at %s ==========", now())

    logger.info("BRONZE_TABLE=%s", BRONZE_TABLE)
    logger.info("SILVER_TABLE=%s", SILVER_TABLE)

    missing = check_jars()
    if missing:
        raise RuntimeError(f"Missing JARs: {missing}")

    spark = create_spark()

    try:
        logger.info("Reading Bronze Iceberg table...")
        bronze_df = spark.table(BRONZE_TABLE)

        bronze_count = bronze_df.count()
        logger.info("Bronze row count=%d", bronze_count)

        if bronze_count == 0:
            raise RuntimeError("Bronze table is EMPTY")

        logger.info("Bronze schema:")
        bronze_df.printSchema()

        # ----------------------------------------------------------
        # DEDUPLICATION LOGIC (BUSINESS KEY = category_key)
        # ----------------------------------------------------------
        logger.info("Applying deduplication on category_key")

        window = Window.partitionBy("category_key").orderBy(col("category_key"))

        silver_df = (
            bronze_df
            .withColumn("rn", row_number().over(window))
            .filter(col("rn") == 1)
            .drop("rn")
        )

        silver_count = silver_df.count()
        dup_count = bronze_count - silver_count

        logger.info("Duplicate rows removed=%d", dup_count)
        logger.info("Silver row count=%d", silver_count)

        logger.info("Sample Silver rows:")
        for r in silver_df.take(5):
            logger.info("ROW=%s", r)

        logger.info("Writing Silver Iceberg table...")
        silver_df.writeTo(SILVER_TABLE).createOrReplace()

        logger.info("Silver Iceberg write SUCCESS")

        logger.info("Verifying Silver table...")
        verify = spark.sql(
            f"SELECT COUNT(*) AS cnt FROM {SILVER_TABLE}"
        ).collect()
        logger.info("Verification COUNT=%s", verify)

        spark.stop()
        logger.info("========== SILVER JOB SUCCESS at %s ==========", now())

    except Exception as e:
        logger.error("SILVER JOB FAILED: %s", str(e))
        logger.error("TRACEBACK:\n%s", traceback.format_exc())
        raise

# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
