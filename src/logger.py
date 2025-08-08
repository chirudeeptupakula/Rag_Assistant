import logging

# Basic logger setup
logging.basicConfig(
    level=logging.INFO,  # Can change to DEBUG for more verbosity
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

rag_logger = logging.getLogger(__name__)
