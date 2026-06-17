from dataclasses import dataclass


# TODO: We don't use this, but might be nice if we do? Or maybe not and the mechanism list is good
# I like the mechanism list so maybe remove this
@dataclass
class HarnessConfig:
    m1_info_seeking: bool = False
    m2_memory: bool = False
    m3_state: bool = False
    m4_compute: bool = False
    m5_templating: bool = False
    m6_planning: bool = False
