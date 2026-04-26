"""Generate face blur ML system architecture diagram."""

from pathlib import Path

from diagrams import Cluster, Diagram, Edge
from diagrams.onprem.client import Users
from diagrams.onprem.compute import Server
from diagrams.onprem.container import Docker
from diagrams.onprem.database import PostgreSQL
from diagrams.onprem.inmemory import Redis
from diagrams.onprem.mlops import Mlflow
from diagrams.onprem.monitoring import Grafana, Prometheus
from diagrams.onprem.network import Nginx
from diagrams.onprem.queue import Kafka
from diagrams.onprem.storage import Ceph

OUTPUT = Path(__file__).parent / "face_blur_architecture"

graph_attr = {
    "fontsize": "16",
    "bgcolor": "white",
    "splines": "ortho",
}

with Diagram(
    "Face Blur ML System",
    filename=str(OUTPUT),
    show=False,
    direction="LR",
    graph_attr=graph_attr,
):
    client = Users("Client App")

    with Cluster("Ingestion"):
        api = Nginx("API Gateway")
        raw_bucket = Ceph("S3: raw videos")
        upload_topic = Kafka("video.uploaded")

    with Cluster("Processing"):
        splitter = Server("Splitter")
        chunks_topic = Kafka("chunk.ready")

        with Cluster("GPU Workers"):
            workers = [Docker(f"Worker {i}") for i in range(1, 4)]

        processed_topic = Kafka("chunk.processed")
        merger = Server("Merger")
        done_topic = Kafka("video.processed")

    with Cluster("ML Platform"):
        registry = Mlflow("Model Registry")
        serving = Server("Triton Serving")
        metadata = PostgreSQL("Detections\nmetadata")

    with Cluster("Storage"):
        processed_bucket = Ceph("S3: processed")
        audit_bucket = Ceph("S3: audit logs")
        cache = Redis("Result cache")

    with Cluster("Result Serving"):
        result_api = Nginx("Result API")
        webhook = Server("Webhook\nNotifier")

    with Cluster("Observability"):
        prometheus = Prometheus("Metrics")
        grafana = Grafana("Dashboards")

    # Ingestion flow
    client >> Edge(label="upload") >> api
    api >> raw_bucket
    api >> upload_topic

    # Processing flow
    upload_topic >> splitter
    splitter >> raw_bucket
    splitter >> chunks_topic
    chunks_topic >> workers
    raw_bucket >> Edge(style="dashed", label="frames") >> workers
    registry >> serving
    serving >> Edge(style="dashed", label="model") >> workers
    workers >> processed_topic
    workers >> audit_bucket
    workers >> Edge(style="dashed") >> metadata
    processed_topic >> merger
    merger >> processed_bucket
    merger >> done_topic

    # Result flow
    done_topic >> webhook
    webhook >> Edge(label="callback") >> client
    client >> Edge(label="fetch") >> result_api
    result_api >> cache
    result_api >> processed_bucket

    # Observability
    workers >> Edge(style="dotted", color="gray") >> prometheus
    merger >> Edge(style="dotted", color="gray") >> prometheus
    api >> Edge(style="dotted", color="gray") >> prometheus
    prometheus >> grafana
