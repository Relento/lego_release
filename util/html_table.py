import collections
import contextlib
import os
import os.path as osp
import shutil
import sys

import six

__all__ = ['HTMLTableColumnDesc', 'HTMLTableVisualizer']


class HTMLTableColumnDesc(collections.namedtuple(
    '_HTMLTableColumnDesc', ['identifier', 'name', 'type', 'css', 'td_css'],
)):
    pass


HTMLTableColumnDesc.__new__.__defaults__ = (None, None)


class HTMLTableVisualizer(object):
    """
    Example:
        >>> vis = HTMLTableVisualizer('<some_dir>', 'Visualization')
        >>> with vis.html():
        >>>     with vis.table('Table Name', [
        >>>         HTMLTableColumnDesc('column1', 'Image', 'image', {'width': '120px'}),
        >>>         HTMLTableColumnDesc('column2', 'Result', 'figure' {}),
        >>>         HTMLTableColumnDesc('column3', 'Supervision', 'text' {}),
        >>>         HTMLTableColumnDesc('column4', 'Prediction', 'code' {'font-size': '12px'})
        >>>     ]):
        >>>         vis.row(...)
    """

    def __init__(self, visdir, title):
        self.visdir = visdir
        self.title = title

        self._index_file = None
        self._table_counter = 0
        self._row_counter = 0

        self._current_columns = None

    @contextlib.contextmanager
    def html(self):
        self.begin_html()
        yield self
        self.end_html()

    def begin_html(self):
        if osp.isfile(self.visdir):
            raise FileExistsError('Visualization dir "{}" is a file.'.format(self.visdir))
        elif osp.isdir(self.visdir):
            if yes_or_no('Visualization dir "{}" is not empty. Do you want to overwrite?'.format(self.visdir)):
                shutil.rmtree(self.visdir)
            else:
                raise FileExistsError('Visualization dir "{}" already exists.'.format(self.visdir))

        os.makedirs(self.visdir, exist_ok=True)
        os.makedirs(osp.join(self.visdir, 'assets'), exist_ok=True)
        self._index_file = open(self.get_index_filename(), 'w')
        self._print('<html>')
        self._print('<head>')
        self._print('<title>{}</title>'.format(self.title))
        self._print('<style>')
        self._print('td {vertical-align:top;padding:5px}')
        self._print('</style>')
        self._print('</head>')
        self._print('<body>')
        self._print('<h1>{}</h1>'.format(self.title))

    def end_html(self):
        self._print('</body>')
        self._print('</html>')
        self._index_file.close()
        self._index_file = None

    @contextlib.contextmanager
    def table(self, name, columns):
        self.begin_table(name, columns)
        yield self
        self.end_table()

    def begin_table(self, name, columns):
        self._current_columns = columns

        self._print('<style>')
        for c in columns:
            css = {} if c.css is None else c.css
            self._print('.table{}_column_{}'.format(self._table_counter, c.identifier), '{',
                        ';'.join([k + ':' + v for k, v in css.items()]), '}')
            css = {} if c.td_css is None else c.td_css
            self._print('.table{}_td_{}'.format(self._table_counter, c.identifier), '{',
                        ';'.join([k + ':' + v for k, v in css.items()]), '}')

        self._print('</style>')
        self._print('<h3>{}</h3>'.format(name))
        self._print('<table>')
        self._print('<tr>')
        for c in columns:
            self._print('  <td><b>{}</b></td>'.format(c.name))
        self._print('</tr>')

    def end_table(self):
        self._print('</table>')
        self._current_columns = None
        self._table_counter += 1
        self._row_counter = 0

    def row(self, *args, **kwargs):
        assert self._current_columns is not None

        if len(args) > 0:
            assert len(kwargs) == 0 and len(args) == len(self._current_columns)
            for c, a in zip(self._current_columns, args):
                kwargs[c.identifier] = a

        row_identifier = kwargs.pop('row_identifier', 'row{:06d}'.format(self._row_counter))

        self._print('<tr>')
        for c in self._current_columns:
            obj = kwargs[c.identifier]
            classname = 'table{}_td_{}'.format(self._table_counter, c.identifier)
            self._print('  <td class="{}">'.format(classname))
            classname = 'table{}_column_{}'.format(self._table_counter, c.identifier)
            if c.type == 'file':
                link, alt = self.canonize_link('file', obj)
                self._print('    <a class="{}" href="{}">{}</a>'.format(classname, link, alt))
            elif c.type == 'image' or c.type == 'figure':
                link, alt = self.canonize_link(c.type, obj, row_identifier, c.identifier)
                self._print('    <img class="{}" src="{}" alt="{}" />'.format(classname, link, alt))
            elif c.type == 'text' or c.type == 'code':
                tag = 'pre' if c.type == 'code' else 'div'
                self._print('    <{} class="{}">{}</{}>'.format(tag, classname, obj, tag))
            elif c.type == 'raw':
                self._print('    {}'.format(obj))
            else:
                raise ValueError('Unknown column type: {}.'.format(c.type))
            self._print('  </td>')
        self._print('</tr>')
        self._flush()

        self._row_counter += 1

    def _print(self, *args, **kwargs):
        assert self._index_file is not None
        print(*args, file=self._index_file, **kwargs)

    def _flush(self):
        self._index_file.flush()

    def get_index_filename(self):
        return osp.join(self.visdir, 'index.html')

    def get_asset_filename(self, row_identifier, col_identifier, ext):
        table_dir = osp.join(self.visdir, 'assets', 'table{}'.format(self._table_counter))
        os.makedirs(table_dir, exist_ok=True)
        return osp.join(table_dir, '{}_{}.{}'.format(row_identifier, col_identifier, ext))

    def save_image(self, image, row_identifier, col_identifier, ext='png'):
        filename = self.get_asset_filename(row_identifier, col_identifier, ext)
        image.save(filename)
        return filename

    def save_figure(self, figure, row_identifier, col_identifier, ext='png'):
        filename = self.get_asset_filename(row_identifier, col_identifier, ext)
        figure.savefig(filename, bbox_inches='tight', pad_inches=0)
        return filename

    def canonize_link(self, filetype, obj, row_identifier=None, col_identifier=None):
        if filetype == 'file':
            assert isinstance(obj, six.string_types)
            return osp.relpath(obj, self.visdir), osp.basename(obj)
        elif filetype == 'image':
            if not isinstance(obj, six.string_types):
                obj = self.save_image(obj, row_identifier, col_identifier)
            return osp.relpath(obj, self.visdir), osp.basename(obj)
        elif filetype == 'figure':
            if not isinstance(obj, six.string_types):
                obj = self.save_figure(obj, row_identifier, col_identifier)
            return osp.relpath(obj, self.visdir), osp.basename(obj)
        else:
            raise ValueError('Unknown file type: {}.'.format(filetype))


def yes_or_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.
    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
    It must be "yes" (the default), "no" or None (meaning that
    an answer is required from the user).
    The "answer" return value is True for "yes" or False for "no".
    """

    valid = {
        "yes": True, "y": True, "ye": True,
        "no": False, "n": False,
        "default": None, "def": None, "d": None
    }

    quiet = os.getenv('JAC_QUIET', '')
    if quiet != '':
        quiet = quiet.lower()
        assert quiet in valid, 'Invalid JAC_QUIET environ: {}.'.format(quiet)
        choice = valid[quiet]
        sys.stdout.write('Jacinle Quiet run:\n\tQuestion: {}\n\tChoice: {}\n'.format(question,
                                                                                     'Default' if choice is None else 'Yes' if choice else 'No'))
        return choice if choice is not None else default

    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("Invalid default answer: '%s'." % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")
