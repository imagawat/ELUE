import csv
import pprint
import os.path as osp
_header = None
_snapshot_dir = None
def set_dir(dirname):
    global _snapshot_dir
    _snapshot_dir = dirname
def csv_write(inps, init):
    if _snapshot_dir:
        file_name = osp.join(_snapshot_dir, 'additional.csv')
        if init:
            with open(file_name, 'w') as f:
                global _header
                _header = list(inps.keys())
                writer = csv.DictWriter(f, fieldnames=_header)
                writer.writeheader()
                writer.writerow(inps)
        else:
            with open(file_name, 'a') as f:
                writer = csv.DictWriter(f, fieldnames=_header)
                writer.writerow(inps)


    else:        
        print("set dirname")
        raise RuntimeError
def trajectory_write(infos, epinum):
    if _snapshot_dir:
        file_name = osp.join(_snapshot_dir, 'traj_{}.csv'.format(epinum))
        keys = ['fingerX', 'fingerY', 'objX', 'objY', 'goalX', 'goalY']
        info_keys = ['fingerXY', 'objXY', 'goal']
        with open(file_name, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=list(keys))
            writer.writeheader()
        #with open(file_name, 'a') as f:
            for info in infos:
                _dict = {}
                for i, info_key in enumerate(info_keys):
                    assert(info_key in info)
                    vals = info[info_key]
                    _dict[keys[2*i]] = vals[0]
                    _dict[keys[2*i+1]] = vals[1]
                writer.writerow(_dict)


    else:        
        print("set dirname")
        raise RuntimeError
