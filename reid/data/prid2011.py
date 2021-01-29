import glob
import os.path as osp
import json
from torch.utils.data import Dataset
from PIL import Image
import torch


def read_json(fpath):
    """Reads json file from a path."""
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


class PRID2011(Dataset):
    """PRID2011.

    Reference:
        Hirzer et al. Person Re-Identification by Descriptive and
        Discriminative Classification. SCIA 2011.

    URL: `<https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/PRID11/>`_

    Dataset statistics:
        - identities: 200.
        - tracklets: 400.
        - cameras: 2.
    """
    dataset_dir = 'prid2011'
    dataset_url = None

    def __init__(self, transform, root='', split_id=0, mode="query", **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.mode = mode
        self.transform = transform

        self.split_path = osp.join(self.dataset_dir, 'splits_prid2011.json')
        self.cam_a_dir = osp.join(
            self.dataset_dir, 'prid_2011', 'multi_shot', 'cam_a'
        )
        self.cam_b_dir = osp.join(
            self.dataset_dir, 'prid_2011', 'multi_shot', 'cam_b'
        )

        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError(
                'split_id exceeds range, received {}, but expected between 0 and {}'
                    .format(split_id,
                            len(splits) - 1)
            )
        split = splits[split_id]
        train_dirs, test_dirs = split['train'], split['test']

        # self.train = self.process_dir(train_dirs, cam1=True, cam2=True)
        query = self.process_dir(test_dirs, cam1=True, cam2=False)
        gallery = self.process_dir(test_dirs, cam1=False, cam2=True)
        self.data = {"query": query, "gallery": gallery}

    def __getitem__(self, index):
        data = self.data[self.mode]
        person_tracklet, id_, cam = data[index]
        imgs = torch.tensor([])
        for path in person_tracklet[:32]:
            img_data = Image.open(path)
            img = torch.unsqueeze(self.transform(img_data), 0)
            imgs = torch.cat((imgs, img), 0)
        return imgs, id_, cam

    def __len__(self):
        return len(self.data[self.mode])

    def process_dir(self, dirnames, cam1=True, cam2=True):
        tracklets = []
        dirname2pid = {dirname: i for i, dirname in enumerate(dirnames)}

        for dirname in dirnames:
            if cam1:
                person_dir = osp.join(self.cam_a_dir, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 0))

            if cam2:
                person_dir = osp.join(self.cam_b_dir, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 1))

        return tracklets
