import datetime
import os
from typing import List, Any, Dict

import numpy as np

from tu.loggers.html_helper import BaseHTMLHelper
from tu.loggers.html_table import HTMLTableVisualizer


# interface adapted from cyclegan-pix2pix by Junyan Zhu


class HTMLVisualizer(BaseHTMLHelper):
    def __init__(self, opt):

        self.vis = HTMLTableVisualizer(visdir=os.path.join(opt.checkpoints_dir, opt.name, 'web'),
                                       title=f"{opt.name}_{datetime.datetime.now().strftime('%Y_%m%d_%H%M_%S')}",
                                       persist_row_counter=False)  # rewritten each time so not persisting row counts

        # create a logging file to store training losses
        os.makedirs(os.path.join(opt.checkpoints_dir, opt.name), exist_ok=True)
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        # with open(self.log_name, "a") as log_file:
        #     now = time.strftime("%c")
        #     log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        pass

    def display_current_results(self, layout: List[List[Dict[str, Any]]], epoch, iter):
        """

        Args:
            layout: a *2D* list, each element is a dictionary with keys 'info' and 'image'
            epoch:
            iter:

        Returns:

        """
        with self.vis.html():
            # rewrite html page
            # hard-code col_type for now
            self.dump_table(self.vis, layout=layout, table_name=f"ep_{epoch}_it_{iter}", col_type='image')
        self.print_url(self.vis)

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except VisdomExceptionBase:
            self.create_visdom_connections()

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.7f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
