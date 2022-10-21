import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
plt.rcParams["figure.figsize"] = (12.5,10)


import numpy as np
from scipy.integrate import quad
import copy

def dcb(x, N, mean, sigma, beta1, m1, beta2, m2):
  beta1, m1, beta2, m2 = np.abs(beta1), np.abs(m1), np.abs(beta2), np.abs(m2)

  with np.errstate(all='ignore'):
    A1 = np.power(m1/beta1, m1) * np.exp(-beta1**2/2)
    B1 = m1/beta1 - beta1
    A2 = np.power(m2/beta2, m2) * np.exp(-beta2**2/2)
    B2 = m2/beta2 - beta2

    xs = (x-mean)/sigma

    middle = N*np.exp(-xs**2/2)*(-beta1 < xs)*(xs < beta2)
    left = np.nan_to_num(N*A1*np.power(B1-xs, -m1)*(xs<=-beta1), nan=0.0)
    right = np.nan_to_num(N*A2*np.power(B2+xs, -m2)*(xs>=beta2), nan=0.0)

  return left + middle + right

def getVariationsFinalFits(finalfits_sys, sys_name):
  mean, sigma, rate = None, None, None
  
  for line in finalfits_sys.split("\n"):
    if sys_name+"_" in line:
      if "mean" in line: mean = float(line.split("mean")[1])
      if "sigma" in line: sigma = float(line.split("sigma")[1])
      if "rate" in line: rate = float(line.split("rate")[1])

  return mean, sigma, rate

def getVariationsMyCode(my_code_sys, sys_name):
  mean = my_code_sys["const_mean_%s"%sys_name]
  sigma = my_code_sys["const_sigma_%s"%sys_name]
  rate = my_code_sys["const_rate_%s"%sys_name]

  return mean, sigma, rate

def getParams(nuisance, my_code_params, variations):
  my_code_params = copy.deepcopy(my_code_params)
  
  my_code_params[0] *= (1+variations[2]*nuisance)
  my_code_params[1] *= (1+variations[0]*nuisance)
  my_code_params[2] *= (1+variations[1]*nuisance)

  return my_code_params

#100->180 with 800 bins using final fits effSigma
finalfits_sys = """scale_13TeVscale_mean                                                  0.0012548
scale_13TeVscale_sigma                                                0.00321716
scale_13TeVscale_rate                                                 0.00166564
fnuf_13TeVscaleCorr_mean                                              0.00203787
fnuf_13TeVscaleCorr_sigma                                           -0.000332399
fnuf_13TeVscaleCorr_rate                                             0.000287382
material_13TeVscaleCorr_mean                                         0.000638552
material_13TeVscaleCorr_sigma                                        0.000692735
material_13TeVscaleCorr_rate                                          0.00137207
smear_13TeVsmear_mean                                                5.52798e-05
smear_13TeVsmear_sigma                                               -0.00168793
smear_13TeVsmear_rate                                                 0.00299956"""

my_code_sys = {
  "const_mean_fnuf": 0.00195739115588367,
  "const_mean_material": 0.0006697592907585204,
  "const_mean_scale": 0.0013024606741964817,
  "const_mean_smear": -5.788198177469894e-05,
  "const_rate_fnuf": 0.00028739048866555095,
  "const_rate_material": 0.0013721204595640302,
  "const_rate_scale": -0.0016656030202284455,
  "const_rate_smear": -0.0029995758086442947,
  "const_sigma_fnuf": -0.0005951332967958035,
  "const_sigma_material": 0.0006931891240896248,
  "const_sigma_scale": 0.003089882822496621,
  "const_sigma_smear": -0.001693073055512591
  #"const_sigma_smear": -0.1
}
my_code_params = [
  1.414748191833496,
  -0.15025648561301352,
  1.2067525355959041,
  1.3461983259189332,
  4.186224845288327,
  1.6430080778532261,
  19.999999999210292
]
my_code_params[1] += 125
norm = quad(dcb, 100, 180, args=tuple(my_code_params))[0]
my_code_params[0] /= norm

sys_names = ["fnuf", "material", "scale", "smear"]
for sys_name in sys_names:
  variations = getVariationsFinalFits(finalfits_sys, sys_name)

  x=np.linspace(-1, 1, 100)
  params = np.array([getParams(xi, my_code_params, variations) for xi in x])
  
  plt.plot(x, params[:, 0])
  plt.xlabel(r"$\theta$")
  plt.ylabel("Rate")
  plt.savefig("finalfits_%s_rate.png"%sys_name)
  plt.clf()

  plt.plot(x, params[:, 1])
  plt.xlabel(r"$\theta$")
  plt.ylabel("Mean")
  plt.savefig("finalfits_%s_mean.png"%sys_name)
  plt.clf()

  plt.plot(x, params[:, 2])
  plt.xlabel(r"$\theta$")
  plt.ylabel("Sigma")
  plt.savefig("finalfits_%s_sigma.png"%sys_name)
  plt.clf()

  x = np.linspace(115, 135, 1000)
  plt.plot(x, dcb(x, *getParams(0, my_code_params, variations)), label="Nominal")
  plt.plot(x, dcb(x, *getParams(1, my_code_params, variations)), label="Up")
  plt.plot(x, dcb(x, *getParams(-1, my_code_params, variations)), label="Down")
  plt.xlabel(r"$m_{\gamma\gamma}$")
  plt.legend()
  plt.savefig("finalfits_%s_pdf.png"%sys_name)
  plt.clf()

for sys_name in sys_names:
  variations = getVariationsMyCode(my_code_sys, sys_name)

  x=np.linspace(-1, 1, 100)
  params = np.array([getParams(xi, my_code_params, variations) for xi in x])
  
  plt.plot(x, params[:, 0])
  plt.xlabel(r"$\theta$")
  plt.ylabel("Rate")
  plt.savefig("my_code_%s_rate.png"%sys_name)
  plt.clf()

  plt.plot(x, params[:, 1])
  plt.xlabel(r"$\theta$")
  plt.ylabel("Mean")
  plt.savefig("my_code_%s_mean.png"%sys_name)
  plt.clf()

  plt.plot(x, params[:, 2])
  plt.xlabel(r"$\theta$")
  plt.ylabel("Sigma")
  plt.savefig("my_code_%s_sigma.png"%sys_name)
  plt.clf()

  x = np.linspace(115, 135, 1000)
  plt.plot(x, dcb(x, *getParams(0, my_code_params, variations)), label="Nominal")
  plt.plot(x, dcb(x, *getParams(1, my_code_params, variations)), label="Up")
  plt.plot(x, dcb(x, *getParams(-1, my_code_params, variations)), label="Down")
  plt.xlabel(r"$m_{\gamma\gamma}$")
  plt.legend()
  plt.savefig("my_code_%s_pdf.png"%sys_name)
  plt.clf()