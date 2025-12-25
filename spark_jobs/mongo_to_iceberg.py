import os
import sys
import time
import logging
import traceback
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# ------------------------------------------------------------------
# LOGGING SETUP (MATCHING PROJECT 2 TEMPLATE)
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("mongo_to_iceberg_p3")

# ------------------------------------------------------------------
# ENVIRONMENT VARIABLES
# ------------------------------------------------------------------
# Connection for MongoDB (Internal Docker Network)
MONGO_URI = "mongodb://mongo_user:mongo_pass@local-mongodb:27017/airflow_db?authSource=admin"

ICEBERG_WAREHOUSE = os.getenv("ICEBERG_WAREHOUSE", "s3a://promotionengine-search")
NESSIE_URI = os.getenv("NESSIE_URI", "http://nessie:19120/api/v2")
NESSIE_REF = os.getenv("NESSIE_REF", "main")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "eu-north-1")

JAR_DIR = "/opt/airflow/jars"

# ALL REQUIRED JARS (Added MongoDB specific ones)
JARS = [
    os.path.join(JAR_DIR, "iceberg-spark-runtime-3.4_2.12-1.5.2.jar"),
    os.path.join(JAR_DIR, "iceberg-nessie-1.5.2.jar"),
    os.path.join(JAR_DIR, "nessie-client-0.99.0.jar"),
    os.path.join(JAR_DIR, "nessie-spark-extensions-3.4_2.12-0.105.7.jar"),
    os.path.join(JAR_DIR, "hadoop-aws-3.3.4.jar"),
    os.path.join(JAR_DIR, "aws-java-sdk-bundle-1.12.772.jar"),
    # NEW MONGO JARS
    os.path.join(JAR_DIR, "mongo-spark-connector_2.12-10.1.1.jar"),
    os.path.join(JAR_DIR, "mongodb-driver-core-4.11.2.jar"),
    os.path.join(JAR_DIR, "mongodb-driver-sync-4.11.2.jar"),
    os.path.join(JAR_DIR, "bson-4.11.2.jar"),
]

NAMESPACE = "sales"
TABLE = "mongo_orders"
TABLE_IDENT = f"nessie.{NAMESPACE}.{TABLE}"

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
    logger.info("Creating SparkSession")

    spark = (
        SparkSession.builder
        .appName("MONGO_TO_ICEBERG_PROJECT3")
        .config("spark.jars", ",".join(JARS))

        # Iceberg + Nessie
        .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
        .config("spark.sql.defaultCatalog", "nessie")
        .config("spark.sql.catalog.nessie", "org.apache.iceberg.spark.SparkCatalog")
        .config("spark.sql.catalog.nessie.catalog-impl", "org.apache.iceberg.nessie.NessieCatalog")
        .config("spark.sql.catalog.nessie.uri", NESSIE_URI)
        .config("spark.sql.catalog.nessie.ref", NESSIE_REF)
        .config("spark.sql.catalog.nessie.warehouse", ICEBERG_WAREHOUSE)

        # AWS S3
        .config("spark.hadoop.fs.s3a.access.key", AWS_ACCESS_KEY_ID)
        .config("spark.hadoop.fs.s3a.secret.key", AWS_SECRET_ACCESS_KEY)
        .config("spark.hadoop.fs.s3a.endpoint.region", AWS_REGION)
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        
        # MONGODB CONFIGURATION
        .config("spark.mongodb.read.connection.uri", MONGO_URI)
        .config("spark.mongodb.write.connection.uri", MONGO_URI)

        .getOrCreate()
    )

    logger.info("Spark version=%s", spark.version)
    return spark

# ------------------------------------------------------------------
def main():
    logger.info("========== JOB STARTED at %s ==========", now())

    missing = check_jars()
    if missing:
        raise RuntimeError(f"Missing JARs: {missing}")

    spark = create_spark()

    try:
        logger.info("Reading from MongoDB...")
        # READ FROM MONGO
        df = (
            spark.read
            .format("mongodb")
            .option("database", "airflow_db")
            .option("collection", "orders")
            .load()
        )

        row_count = df.count()
        logger.info("MongoDB row count=%d", row_count)
        
        logger.info("Schema from MongoDB:")
        df.printSchema()

        if row_count == 0:
            raise RuntimeError("MongoDB collection is EMPTY — stopping job")

        logger.info("Ensuring namespace exists: %s", NAMESPACE)
        spark.sql(f"CREATE NAMESPACE IF NOT EXISTS nessie.{NAMESPACE}")

        logger.info("Writing to Iceberg table: %s", TABLE_IDENT)
        # We use createOrReplace because we want to overwrite old tests
        df.writeTo(TABLE_IDENT).createOrReplace()

        logger.info("Iceberg write SUCCESS")

        logger.info("Verifying Iceberg read...")
        result = spark.sql(f"SELECT COUNT(*) AS cnt FROM {TABLE_IDENT}").collect()
        logger.info("Verification COUNT=%s", result)

        logger.info("Stopping Spark")
        spark.stop()

        logger.info("========== JOB SUCCESS at %s ==========", now())

    except Exception as e:
        logger.error("JOB FAILED: %s", str(e))
        logger.error("TRACEBACK:\n%s", traceback.format_exc())
        raise

# ------------------------------------------------------------------
if __name__ == "__main__":
    main()