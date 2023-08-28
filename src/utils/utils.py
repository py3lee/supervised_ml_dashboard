from datetime import datetime
from IPython.terminal.ipapp import launch_new_instance
import logging
import sys

def create_logger(log_path):
    """
    Creates logger

    Returns:
        logger
    """
    NOW = datetime.now()
    RUN_DATE = NOW.strftime('%Y%m%d')

    h1 = logging.StreamHandler()
    h1.setLevel(logging.DEBUG)

    h2 = logging.FileHandler(filename=f'{log_path}/vaers_{RUN_DATE}.log')
    h2.setLevel(logging.INFO)

    logging.basicConfig(level=logging.DEBUG, 
                        format='%(asctime)s : %(levelname)s : %(filename)s, line %(lineno)s : %(message)s',
                        handlers=[h1, h2]
                        )
    logger = logging.getLogger(__name__)
    return logger

def launch_dashboard():
    """launch Supervised Model Dashboard notebook"""

    sys.argv.append("notebook")
    sys.argv.append("--NotebookApp.open_browser=True")
    sys.argv.append("notebook/Supervised_Model_Dashboard.ipynb")

    launch_new_instance()