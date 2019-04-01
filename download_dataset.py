from download_utils import down_fr_url

urls = ['http://images.cocodataset.org/zips/train2014.zip',
        'http://images.cocodataset.org/zips/val2014.zip',
        'http://images.cocodataset.org/annotations/annotations_trainval2014.zip']

save_dir = 'data'

down_fr_url(urls, save_dir, unzip=True)
