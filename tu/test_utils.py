import os
import unittest
from typing import List, Any

import numpy as np
import torch
from PIL import Image

from tu.loggers.html_helper import BaseHTMLHelper
from tu.loggers.html_table import HTMLTableVisualizer


class BaseTestCase(unittest.TestCase):
    name = 'default'
    save_dir = 'YOUR_SAVE_DIR'

    @classmethod
    def setUpClass(cls) -> None:

        cls.vis = HTMLTableVisualizer(os.path.join(cls.save_dir, cls.name), title='')

        cls.vis.begin_html()
        cls.vis_helper = BaseHTMLHelper()
        cls.vis_helper.print_url(cls.vis)

        cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @classmethod
    def tearDownClass(cls) -> None:
        cls.vis.end_html()

    @classmethod
    def dump(cls, layout: List[List[Any]], table_name=''):
        assert isinstance(layout, list)
        for row in range(len(layout)):
            assert isinstance(layout[row], list)
            for col in range(len(layout[row])):
                item = layout[row][col]
                if isinstance(item, np.ndarray):
                    layout[row][col] = Image.fromarray(item)
        cls.vis_helper.dump_table(cls.vis, layout=layout, table_name=table_name, col_type='auto')
