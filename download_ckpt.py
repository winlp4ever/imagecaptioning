from urllib.request import urlretrieve
from urllib.parse import urlparse
import os, sys
import zipfile

def error_handle():
    """
    print out error line and filename
    (from stack overflow)
    """
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print(exc_type, fname, exc_tb.tb_lineno)


def retri_fn_url(url: str) -> str:
    """return filename from an url
    Parameters
    ----------
    url : str
        link url
    Returns
    -------
    str
        url file name
    """
    addr = urlparse(url).path
    return os.path.basename(addr)


def down_fr_url(urls: list, save_dir: str=''):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for url in urls:
        try:
            fn = retri_fn_url(url)
            save_path = os.path.join(save_dir, fn)
            print('Downloading ... {}'.format(fn))
            urlretrieve(url, save_path)

        except Exception as e:
            error_handle()
            print(e)
    print('Done.')


if __name__ == '__main__':
    urls = ['https://www.dropbox.com/s/h4pypk9s2mxzzme/checkpoint-3.pth.tar?dl=1']
    down_fr_url(urls, save_dir='checkpoints')
    vocab = ['https://www.dropbox.com/s/4wdkvr9pqep9maq/vocab.txt?dl=1']
    down_fr_url(vocab, save_dir='data')
