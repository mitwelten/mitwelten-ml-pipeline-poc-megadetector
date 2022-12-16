import urllib.request
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def main():
    WEIGHTS_A = 'https://github.com/microsoft/CameraTraps/releases/download/v5.0/md_v5a.0.0.pt'
    OUTPUT_A = 'model_weights/md_v5a.0.0.pt' 
    WEIGHTS_B = 'https://github.com/microsoft/CameraTraps/releases/download/v5.0/md_v5b.0.0.pt'
    OUTPUT_B = 'model_weights/md_v5b.0.0.pt'

    download_url(url=WEIGHTS_A, output_path=OUTPUT_A)
    download_url(url=WEIGHTS_B, output_path=OUTPUT_B)


if __name__ == '__main__':
    main()
