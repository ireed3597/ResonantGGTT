import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
plt.rcParams["figure.figsize"] = (12.5,10)

vars = ["LeadPhoton_pt", "SubleadPhoton_pt", "Diphoton_pt", "Diphoton_mass"]
columns = ["weight_central", "process_id"] + vars

nominal = pd.read_parquet("Inputs/merged_nominal.parquet", columns=columns)
up = pd.read_parquet("Inputs/merged_smear_up.parquet", columns=columns)
down = pd.read_parquet("Inputs/merged_smear_down.parquet", columns=columns)

nominal = nominal[nominal.process_id==33]
up = up[up.process_id==33]
down = down[down.process_id==33]


nbins = 50
for norm in [True, False]:
  for var in vars:
    f, axs = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    l = nominal[var].quantile(0.01)
    h = nominal[var].quantile(0.99)
    n_nominal, bins, patches = axs[0].hist(nominal[var], bins=nbins, range=(l,h), weights=nominal.weight_central, label="Central", histtype='step', density=norm)
    n_up, bins, patches = axs[0].hist(up[var], bins=nbins, range=(l,h), weights=up.weight_central, label="Up", histtype='step', density=norm)
    n_down, bins, patches = axs[0].hist(down[var], bins=nbins, range=(l,h), weights=down.weight_central, label="Down", histtype='step', density=norm)
    axs[0].legend()

    bin_centers = (bins[:-1]+bins[1:])/2
    axs[1].hist(bin_centers, bins=bins, weights=1 - n_up/n_nominal, label="Up", histtype='step')
    axs[1].hist(bin_centers, bins=bins, weights=1 - n_down/n_nominal, label="Down", histtype='step')
    axs[1].legend()
    axs[1].set_ylabel("1 - Ratio")
    axs[1].set_xlabel(var)

    print(n_up/n_nominal)
    print(n_down/n_nominal)

    if not norm: name = "plots/smear/%s.png"%var
    else:        name = "plots/smear/%s_normed.png"%var
    plt.savefig(name)
    plt.clf()