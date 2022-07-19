from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options

        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')

        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--n_vis', type=int, default=32, help='Number of visualization images')
        parser.add_argument('--align_camera', action='store_true',
                            help='Same camera parameters for previous and after images')
        parser.add_argument('--dataset_alias', type=str, default='default', help='dataset alias')
        # rewrite devalue values
        parser.set_defaults(model='test')
        self.isTrain = False
        return parser
