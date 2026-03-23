import logging
import numpy as np


logger = logging.getLogger('main')


def merge_dataset(*args):

    res = args[0]

    [f.update({"test": []}) for f in res["fold"] if "test" not in f]

    for a in args[1:]:

        [f.update({"test": []}) for f in a["fold"] if "test" not in f]
        res["data"].update(a["data"])
        [
            [n[k].extend(o[k]) for k in n.keys()]
            for n, o in zip(res["fold"], a["fold"])
        ]

    for fl in res["fold"]:

        ld = len(res["data"])
        ls = np.sum([len(v) for v in fl.values()])

        if ld != ls:
            logger.warning(f"Warning, the data keys duplicated: {ld} != {ls}")
            
    return res