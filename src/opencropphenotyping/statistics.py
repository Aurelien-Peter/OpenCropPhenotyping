import numpy as np


def compute_statistics(image: np.ndarray) -> dict[str, float]:
    results = {}
    results["min"]=np.nanmin(image)
    results["max"]=np.nanmax(image)
    results["mean"]=np.nanmean(image)
    results["median"]=np.nanmedian(image)
    results["var"]=np.nanvar(image)
    results["std"]=np.nanstd(image)
    results["p5"]=np.nanpercentile(image,5)
    results["p25"]=np.nanpercentile(image,25)
    results["p75"]=np.nanpercentile(image,75)
    results["p95"]=np.nanpercentile(image,95)

    return results