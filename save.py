import pyexr
import numpy
import os
from utils import tone_mapping

non_ta_irr = pyexr.read(os.path.join('dataset', 'sponza', 'inputs', 'color59.exr'))[:,:,:3]
albedo = pyexr.read(os.path.join('dataset', 'sponza', 'inputs', 'albedo59.exr'))

one_spp_full = non_ta_irr * albedo
one_spp_full_tone_mapped = tone_mapping(one_spp_full)

pyexr.write(os.path.join('1spp_non_tone_mapped.exr'), one_spp_full)