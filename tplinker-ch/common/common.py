# -*- coding: utf-8 -*-
import os
import logging.config


file_path = os.path.dirname(os.path.abspath(__file__)).split('common')
logging.config.fileConfig(os.path.join(file_path[0], 'config/logging.conf'))
logger = logging.getLogger()
