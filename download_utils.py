from urllib.request import urlretrieve
from urllib.parse import urlparse
import os, sys
import zipfile
import time
import urllib
from math import ceil

BYTES_PER_MB = 1024 * 1024

def error_handle():
    """Print out error line and filename
    (from stack overflow)
    """
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print(exc_type, fname, exc_tb.tb_lineno)


def retri_file_size(url: str):
    """Return to-be-downloaded file size
    """
    meta = urllib.request.urlopen(url).info()
    return int(meta['Content-Length'])


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


def time_format(secs: int):
    """Transform nb of secs to time format
    """
    s = '{}s'.format(secs % 60).zfill(3)
    if secs < 60:
        return s.rjust(6);
    secs = secs // 60
    m = '{}\''.format(secs % 60).zfill(3)
    if secs < 60:
        return (m + s).rjust(6)
    secs = secs // 60
    h = '{}h'.format(secs % 24).zfill(3)
    if secs < 24:
        return (h + m).rjust(6)
    if secs > 2 * 24:
        return 'uknown'
    d = '{}d'.format(secs // 24)
    return (d + h).rjust(6)


def down_fr_url(urls: list, save_dir: str='', unzip: bool=False):
    """Downloading urls to a chosen dir, unzip if necessary

    Parameters
    ----------
    urls : list
        list of urls of files to be downloaded
    save_dir : str
        where to save downloaded files
    unzip : bool
        if True, unzip files after downloaded. default False

    Returns
    -------

    """
    def indicator(quantity, width=10):
        """Change unit between KB and MB
        """
        if quantity > 1024:
            return '{:.0f} MB/s'.format(quantity / 1024).rjust(width)
        return '{:.0f} KB/s'.format(quantity).rjust(width)

    def progress(count, block_size, total_size):
        """Show progress bar when downloading
        """
        global start_time
        if count == 0:
            start_time = time.time()
        if count % 100 == 99 or (count + 1) * block_size >= total_size:
            percent = (count * block_size) / total_size
            down_size_in_mb = count * block_size // BYTES_PER_MB
            total_size_in_mb = total_size // BYTES_PER_MB
            pos = int(ceil(percent * 20))
            down_bar = '[' + '=' * max(pos - 1, 0) + '>' + (20 - pos) * '-' + ']'
            if (count + 1) * block_size >= total_size:
                down_bar = '[' + '=' * 20 +']'
                down_size_in_mb = total_size_in_mb

            speed = (count * block_size) / (time.time() - start_time + 1e-3) / 1024
            time_left = int((total_size_in_mb - down_size_in_mb) * 1024 / (speed + 1e-3))
            print('{} {}/{} MB {} {}\testim. time left: {}'.format(down_bar,
                    str(down_size_in_mb).rjust(len(str(total_size_in_mb))), # right align text
                    total_size_in_mb, ('(%2.1f%%)'%(percent * 100)).rjust(8),
                    indicator(speed), time_format(time_left)),
                flush=True, end='\r')
    for url in urls:
        try:
            fn = retri_fn_url(url)
            save_path = os.path.join(save_dir, fn)
            if os.path.exists(save_path) and os.path.getsize(save_path) >= retri_file_size(url):
                print('{} already exists.'.format(save_path))
                continue
            print('Downloading {} ...'.format(fn))
            urlretrieve(url, save_path, reporthook=progress)
            print('\n')
            if unzip:
                print('Extracting file ...')
                zip = zipfile.ZipFile(save_path)
                zip.extractall('.')
                zip.close()

        except Exception as e:
            error_handle()
            print(e)
    print('Done.')


if __name__ == '__main__':
    urls = ['https://www.dropbox.com/s/bwnenur90ocp1gz/checkpoint-70.pth.tar?dl=1']
    down_fr_url(urls, save_dir='checkpoints')
    vocab = ['https://www.dropbox.com/s/4wdkvr9pqep9maq/vocab.txt?dl=1']
    down_fr_url(vocab, save_dir='data')
