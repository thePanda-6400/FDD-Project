import pandas as pd, numpy as np
from scipy.stats import spearmanr, pearsonr
PP = {"CV1":.90,"CV2":.83,"CV3":.81,"CV4":.85,"CV5":.90,"CV6":.86,"CV7":.90,"CV8":.86,"CV9":.83,
      "CCV1":.86,"CCV2":.82,"CCV3":.80,"CCV4":.91,"CCV5":.86}
c = pd.read_csv("stage2_coeffs.csv"); c = c[c.regime=="forward"]
c["pp"] = c.uid.map(PP); c = c.dropna(subset=["pp"])
for col in ["K_T_rel","KT_over_KQ"]:
    print(col, "pearson=%.3f spearman=%.3f" % (pearsonr(c[col],c.pp)[0], spearmanr(c[col],c.pp)[0]))