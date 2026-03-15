import logging
from datetime import datetime
import os
# create the logs folder
LOG_DIR = "logs"
os.makedirs(LOG_DIR,exist_ok=True)

# log file name with time-stamp
LOG_File = f"{datetime.now().strftime('%Y-%m-%d_%H-%M_%S')}.log"
LOG_File_Path = os.path.join(LOG_DIR,LOG_File)
# Logging configuration
logging.basicConfig(
    filename=LOG_File_Path,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Create logger object
logger = logging.getLogger()
