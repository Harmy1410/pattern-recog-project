#!/usr/bin/env python3
from alive_progress import alive_bar
from google.cloud import storage

if __name__ == '__main__':
    path = 'full/numpy_bitmap/'
    # We choose the 5 best performing classes from the paper.
    classes = ['airplane', 'car', 'bird', 'sailboat', 'truck']
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name='quickdraw_dataset', user_project=None)
    with alive_bar(len(classes), dual_line = True, title = 'Downloading Data') as bar:
        for label in classes:
            temp_path = path + label +'.npy'
            blob = bucket.blob(temp_path)
            bar.title = '-> Downloading class %s' % label.capitalize()
            blob.download_to_filename('data/' + label + '.npy')
            bar()

