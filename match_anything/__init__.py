from deep_image_matching import logger as logger
from deep_image_matching.config import confs, opt_zoo

confs["match_anything"] = {
    "extractor": {"name": "no_extractor"},
    "matcher": {"name": "match_anything"},
}
confs["minima_roma"] = {
    "extractor": {"name": "no_extractor"},
    "matcher": {"name": "minima_roma"},
}

opt_zoo["matchers"].append("match_anything")
opt_zoo["matchers"].append("minima_roma")
