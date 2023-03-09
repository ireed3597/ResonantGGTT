import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import bisect
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from scipy.stats import chi2
from scipy.stats import expon
from scipy.stats import norm
import scipy.interpolate as spi

from tqdm import tqdm

import warnings

from numba import jit

"""---Limit Setting------------------------------------------------------------------------------------------------"""

class Chi2CdfLookup:
  def __init__(self, granularity=0.001):
    l = 1
    h = 10
    x = np.arange(l, h, granularity)
    self.cdf = spi.interp1d(x, chi2.cdf(x, 1), kind='linear', fill_value=tuple(chi2.cdf([l,h], 1)), bounds_error=False)

  def __call__(self, x):
    return self.cdf(x)

chi2_cdf = Chi2CdfLookup()

def calculateExpectedCLs(mu, s, b):
  """
  Calculates CLs for a given mu and expected number of signal and background events, s and b.
  Will calculate for a single category where s and b are input as just numbers.
  Will also calculate for multiple categories where s and b are arrays (entry per category).
  """
  s, b = np.array(s), np.array(b)
  assert s.shape == b.shape

  qmu = -2 * np.sum( b*(np.log(mu*s+b) - np.log(b)) - mu*s )

  CLs = 1 - chi2_cdf(qmu)
  return CLs

def calculateExpectedLimit(s, b, rlow=0, rhigh=100.0):
  if calculateExpectedCLs(rhigh, s, b) < 0.05:
    return bisect(lambda x: calculateExpectedCLs(x, s, b)-0.05, rlow, rhigh, rtol=0.001)
  else:
    warnings.warn("Limit above rhigh = %f"%rhigh)
    return rhigh

  return limit

def calculateObservedCLs(mu, s, b, k):
  s, b, k = np.array(s), np.array(b), np.array(k)
  assert s.shape == b.shape == k.shape

  qmu = lambda mui, ki: -2 * np.sum( ki*(np.log(b+mui*s) - np.log(ki)) - b - mui*s + ki )
  
  def get_t(mu, k):
    a = np.ones_like(k)
    mu_hat = np.min( [ np.max([0*a, (k-b)/s], axis=0), mu*a ], axis=0 )
    return qmu(mu, k) - qmu(mu_hat, k)

  t = get_t(mu, k)
  sqrt_t = np.sqrt(t)
  t_asi = get_t(mu, b)
  sqrt_t_asi = np.sqrt(t_asi)

  if 0 <= t <= t_asi:
    p_spb = 1 - norm.cdf(sqrt_t)
    p_b = norm.cdf(sqrt_t - sqrt_t_asi)
  elif t > t_asi:
    p_spb = 1 - norm.cdf( (t + t_asi) / (2 * sqrt_t_asi))
    p_b = norm.cdf( (t - t_asi) / (2 * sqrt_t_asi) )
  else:
    raise Exception("Expect t to be >= 0")
  
  CLs = p_spb / (1-p_b)
  return CLs

def calculateObservedLimit(s, b, k, rlow=0, rhigh=100.0):
  if calculateObservedCLs(rhigh, s, b, k) < 0.05:
    return bisect(lambda x: calculateObservedCLs(x, s, b, k)-0.05, rlow, rhigh, rtol=0.001)
  else:
    warnings.warn("Limit above rhigh = %f"%rhigh)
    return rhigh

def calculateExpectedCLs_v2(mu, s, b, CLb=0.5):
  s, b = np.array(s), np.array(b)
  assert s.shape == b.shape

  qmu = lambda mui: -2 * np.sum( b*(np.log(b+mui*s) - np.log(b)) - mui*s)
  
  def get_t(mu):
    return qmu(mu) - qmu(0)

  t = get_t(mu)
  sqrt_t = np.sqrt(t)

  p_spb = 1 - norm.cdf(sqrt_t - norm.ppf(CLb))
      
  CLs = p_spb / CLb
  return CLs

def calculateExpectedLimit_v2(s, b, rlow=0, rhigh=100.0, CLb=0.5):
  if calculateExpectedCLs_v2(rhigh, s, b, CLb) < 0.05:
    return bisect(lambda x: calculateExpectedCLs_v2(x, s, b, CLb)-0.05, rlow, rhigh, rtol=0.001)
  else:
    warnings.warn("Limit above rhigh = %f"%rhigh)
    return rhigh

"""----------------------------------------------------------------------------------------------------------------"""
"""---Fitting------------------------------------------------------------------------------------------------------"""

class ExpFunc():
  def __init__(self, N, norm, l, l_up=0, l_down=0):
    self.l = l
    #self.l_up = l_up
    #self.l_down = l_down
    self.l_up = l
    self.l_down = l

    self.N = N
    self.N_up = N + np.sqrt(N)
    self.N_down = N - np.sqrt(N)
    #self.N_up = N
    #self.N_down = N

    self.norm = norm
  
  def __call__(self, m):
    return (self.N / self.norm(self.l)) * np.exp(-self.l*m)

  def getLowerUpper(self, m):
    fluctuations = []
    for i, l in enumerate([self.l, self.l_up, self.l_down]):
      for j, N in enumerate([self.N, self.N_up, self.N_down]):
        fluctuations.append((N / self.norm(l)) * np.exp(-l*m))
    fluctuations = np.array(fluctuations)
     
    return np.min(fluctuations, axis=0), np.max(fluctuations, axis=0)

  # def getNEventsInSR(self, sr):
  #   nominal = intExp(sr[0], sr[1], self.l, self.N/self.norm(self.l))
  #   return nominal

  def getNEventsInSR(self, sr):
    nominal = intExp(sr[0], sr[1], self.l, self.N/self.norm(self.l))
    return nominal, nominal*(self.N_up/self.N - 1)
    #nonminal, up, down = self.getNEventsInSR_CI(sr)
    #return nonminal, (up-down)/2

  def getNEventsInSR_CI(self, sr):
    nominal = intExp(sr[0], sr[1], self.l, self.N/self.norm(self.l))

    fluctuations = []
    for i, l in enumerate([self.l, self.l_up, self.l_down]):
      for j, N in enumerate([self.N, self.N_up, self.N_down]):
        fluctuations.append(intExp(sr[0], sr[1], l, N/self.norm(l)))

    return nominal, min(fluctuations), max(fluctuations)

def intExp(a, b, l, N=1):
  return (N/l) * (np.exp(-l*a) - np.exp(-l*b))

def bkgNLL(l, mass, pres, sr):
  """
  Negative log likelihood from an unbinned fit of the background to an exponential.
  The exponential probability distribution: P(m, l) = [1/N(l)] * exp(-l*m)
  where m is the diphoton mass, l is a free parameter and N(l) normalises the distribution.

  bkg: a DataFrame of background events with two columns: "mass" and "weight"
  norm: a function that normalises P. This has to be specific to the range of masses
        considered, e.g. the sidebands but not the signal.

  WARNING: weighting not implemented yet
  """
  l = np.abs(l)
  a, b, c, d = pres[0], sr[0], sr[1], pres[1]
  N = -(1/l)*(np.exp(-l*b)-np.exp(-l*a)+np.exp(-l*d)-np.exp(-l*c))
  return np.mean(l*mass+np.log(N))

def gradNLL(l, mass, pres, sr):
  l = np.abs(l)
  a, b, c, d = pres[0], sr[0], sr[1], pres[1]

  N = -(1/l)*(np.exp(-l*b)-np.exp(-l*a)+np.exp(-l*d)-np.exp(-l*c))
  dN_dl = -(1/l)*N + (1/l)*(b*np.exp(-l*b)-a*np.exp(-l*a)+d*np.exp(-l*d)-c*np.exp(-l*c))
  return np.mean(mass + dN_dl/N)

def grad2NLL(l, mass, pres, sr):
  l = np.abs(l)
  a, b, c, d = pres[0], sr[0], sr[1], pres[1]

  N = -(1/l)*(np.exp(-l*b)-np.exp(-l*a)+np.exp(-l*d)-np.exp(-l*c))
  dN_dl = -(1/l)*N + (1/l)*(b*np.exp(-l*b)-a*np.exp(-l*a)+d*np.exp(-l*d)-c*np.exp(-l*c))
  dN2_dl2 = -(2/l)*dN_dl - (1/l)*(b*b*np.exp(-l*b)-a*a*np.exp(-l*a)+d*d*np.exp(-l*d)-c*c*np.exp(-l*c))
  return (1/len(mass)) * ((1/N)*dN2_dl2 - (1/N**2)*dN_dl**2)

def failedBkgFit(bkg, res, pres):
  m_min, m_max = min(bkg.mass), max(bkg.mass)
  print("Nbkg = %f, l_fit = %f \n"%(bkg.weight.sum(), res.x[0]) + str(res))

  n, edges = np.histogram(bkg.mass, range=(pres[0], pres[1]), bins=4*int(pres[1]-pres[0]), weights=bkg.weight)
  plt.errorbar(edges[:-1], n, np.sqrt(n), fmt='o')
  x = np.linspace(min(bkg.mass),max(bkg.mass),100)
  N = n[0] / np.exp(-x[0]*abs(res.x[0]))
  plt.plot(x, N*np.exp(-x*abs(res.x[0])))
  plt.savefig("failed_bkg_fit.png")
  plt.clf()

def fitBkg(bkg, pres, sr, l_guess, counting_sr=None):
  if counting_sr == None: 
    counting_sr = sr
  if pres[0] > sr[0]:
    print(pres, sr)
    sr[0] = pres[0]
  if pres[1] < sr[1]: 
    print(pres, sr)
    sr[1] = pres[1]
  assert sr[1]-sr[0] > 0

  norm = lambda l: (intExp(pres[0], sr[0], l) + intExp(sr[1], pres[1], l))

  if len(bkg) == 0:
    print("WARNING: bkg had no events")
    return ExpFunc(0, norm, l_guess, l_guess, l_guess), 0.0, 0.0

  bounds = [(0.0001, 1)]
  res = minimize(bkgNLL, l_guess, bounds=[(0.0001, 1)], args=(bkg.mass.to_numpy(), pres, sr), jac=gradNLL, tol=1e-4)
  l_fit = res.x[0]

  hess_inv = res.hess_inv.todense()[0][0]
  #estimate error on l from hessian inverse
  l_up = l_fit + np.sqrt(hess_inv/len(bkg))
  l_down = l_fit - np.sqrt(hess_inv/len(bkg))

  #bkg_func(m) = No. bkg events / GeV at a given value of m
  N = float(sum(bkg.weight))
  bkg_func = ExpFunc(N, norm, l_fit, l_up, l_down)

  #assert (l_fit != bounds[0][0]) and (l_fit != bounds[0][1]), plotBkgFit(bkg, bkg_func, pres, sr, saveas="failed_bkg_fit.png")

  #plotBkgFit(bkg, bkg_func, pres, sr, saveas="bkg_fits/bkg_fit_%d.png"%np.random.randint(0, 1000))

  #Number of bkg events in signal region found from fit
  nbkg_sr, nbkg_sr_err = bkg_func.getNEventsInSR(counting_sr)

  return bkg_func, nbkg_sr, nbkg_sr_err

def performFit(sig, bkg, pres=(100,180), sr=(120,130), l_guess=0.01):
  """
  Return the number of signal and background events in the signal region.
  Number of signal events found by simply summing the number found in signal region.
  Number of background events found from an exponential fit to the side bands.

  sig, bkg: DataFrames of signal and background events with two columns: "mass" and "weight"
  pres: The preselection window on the diphoton mass
  sr: The signal region
  l_guess: An initial guess for the l parameter in the exponential fit

  IMPORTANT: bkg should only include events in the sidebands
  """

  if len(sig) > 100:
    sig_sort = sig.sort_values("mass")
    cdf = np.cumsum(sig_sort.weight) / sig_sort.weight.sum()
    l, h = sig_sort.mass.iloc[np.argmin(abs(cdf-0.16))], sig_sort.mass.iloc[np.argmin(abs(cdf-0.84))]
  
    l = max([sr[0], l])
    h = min([sr[1], h])
    counting_sr = (l, h)
    
    #counting_sr = (sig.mass.quantile(0.16), sig.mass.quantile(1-0.16))
  else:
    # this does not affect the limits (because the signal yield is so small at this point) but is nice to have a reasonable guess
    width = 1.5 * (sr[1]-sr[0])/20
    sr_middle = (sr[1]+sr[0]) / 2
    counting_sr = (sr_middle-width, sr_middle+width)
  
  #background fit
  bkg_func, nbkg_sr, nbkg_sr_err = fitBkg(bkg, pres, sr, l_guess, counting_sr)

  #just sum signal events in signal region
  nsig_sr = sig.loc[(sig.mass>counting_sr[0])&(sig.mass<counting_sr[1]), "weight"].sum()

  return nsig_sr, nbkg_sr, nbkg_sr_err, bkg_func

def plotBkgFit(bkg, bkg_func, pres, sr, saveas="bkg_fit.png"):
  bkg = bkg[((bkg.mass>pres[0])&(bkg.mass<sr[0])) | ((bkg.mass>sr[1])&(bkg.mass<pres[1]))]
    
  m = np.linspace(pres[0], pres[1], 100)
  n, bin_edges = np.histogram(bkg.mass, bins=int(pres[1]-pres[0]), range=(pres[0], pres[1]))
  bin_centers = np.array( (bin_edges[:-1] + bin_edges[1:]) / 2 )
  err = np.sqrt(n)
  err[err==0] = 1

  side_bands = (bin_centers<sr[0])|(bin_centers>sr[1])
  plt.errorbar(bin_centers[side_bands], n[side_bands], err[side_bands], fmt='o')
  plt.plot(m, bkg_func(m), label=r"$N = %.1f \cdot $exp$(-(%.3f^{+%.3f}_{-%.3f})*m_{\gamma\gamma})$"%((bkg_func.N/bkg_func.norm(bkg_func.l)), bkg_func.l, bkg_func.l_up-bkg_func.l, bkg_func.l-bkg_func.l_down))
  lower, upper = bkg_func.getLowerUpper(m)
  plt.fill_between(m, lower, upper, color="yellow", alpha=0.5)
  plt.title("Background fit")
  plt.xlabel(r"$m_{\gamma\gamma}$")
  plt.ylabel("No. events")
  plt.ylim(bottom=0)
  plt.legend()
  plt.savefig(saveas)
  #plt.show()
  plt.clf()

"""---Category Optimisation----------------------------------------------------------------------------------------"""

def AMS(s, b):
  """
  Calculate AMS for expected number of signal and background events, s and b.
  Will calculate for a single category where s and b are input as just numbers.
  Will also calculate for multiple categories where s and b are arrays (entry per category)
  """
  s, b = np.array(s), np.array(b)
  AMS2 = 2 * ( (s+b)*np.log(1+s/b) -s ) #calculate squared AMS for each category
  AMS = np.sqrt(np.sum(AMS2)) #square root the sum (summing in quadrature)
  return AMS

# def formBoundariesGrid(bkg, low, high, step, nbounds, include_lower):
#   """
#   Create a list of all possible sets of boundaries given the lowest and highest allowed score and number of boundaries.
#   If include_lower = True, then an additional category in included which always starts from score = 0.
#   """
#   poss_bounds = np.arange(low, high+step, step)
#   # bkg_select = bkg[bkg.score >= low]
#   # bkg_select.sort_values("score", inplace=True)
#   # poss_bounds = np.array(bkg_select.score - 1e-6)
#   #print(len(poss_bounds))

#   grid = np.array(np.meshgrid(*[poss_bounds for i in range(nbounds)])).T.reshape(-1, nbounds)
#   grid = grid[np.all(np.diff(grid) > 0, axis=1)] #only consider ordered sets of boundaries
  
#   grid = np.concatenate((grid, np.ones(len(grid)).reshape(-1, 1)), axis=1) #add 1 to end of every boundary set
#   if include_lower: grid = np.concatenate((np.zeros(len(grid)).reshape(-1, 1), grid), axis=1) #add 0 to end of every boundary set

#   return grid

def formBoundariesGrid(bkg, low, high, step, nbounds, include_lower):
  """
  Create a list of all possible sets of boundaries given the lowest and highest allowed score and number of boundaries.
  If include_lower = True, then an additional category in included which always starts from score = 0.
  """
  bkg_select = bkg[bkg.score >= low]
  bkg_select.sort_values("score", ascending=False, inplace=True)
  #pd.set_option('precision',10)
  #print(bkg_select.iloc[:12]["score"])
  
  poss_bounds = np.arange(1, len(bkg_select), 1)
  # if len(bkg_select) < 160:
  #   poss_bounds = np.concatenate([np.arange(1, 30, 1), np.arange(30, 40, 2), np.arange(40, len(bkg_select), 4)])
  # else:
  #   poss_bounds = np.concatenate([np.arange(1, 30, 1), np.arange(30, 40, 2), np.arange(40, 160, 4), np.arange(160, len(bkg_select), 8)])
  #poss_bounds = np.concatenate([np.arange(1, 20, 1), np.arange(20, 40, 2), np.arange(40, 80, 4), np.arange(80, len(bkg_select), 8)])
  #poss_bounds = np.concatenate([np.arange(1, 40, 1), np.arange(40, 80, 4), np.arange(80, len(bkg_select), 8)])
  print(poss_bounds)

  t = 10

  grid = [[i] for i in poss_bounds if i >= t]
  i=1
  while i < nbounds:
    extension = []
    for bound in grid:
      for j in poss_bounds:
        if j - bound[-1] >= t:
          extension.append(bound+[j])
    i += 1
    grid = extension
  grid = np.array(grid)[:,::-1]
  print(grid)

  #grid = bkg_select.score.to_numpy()[grid] - 1e-10
  grid = (bkg_select.score.to_numpy()[grid-1] + bkg_select.score.to_numpy()[grid]) / 2

  grid = np.concatenate((grid, np.ones(len(grid)).reshape(-1, 1)), axis=1) #add 1 to end of every boundary set
  if include_lower: grid = np.concatenate((np.zeros(len(grid)).reshape(-1, 1), grid), axis=1) #add 0 to end of every boundary set

  return grid

# known_invalid = []
# known_good = []
# def isValidBoundaries(bkg, sig, pres, sr, boundaries, threshold=10):
#   """
#   Checks whethere a given set of boundaries is valid.
#   A set is invalid if there is less than threshold sig events or bkg events in sidebands in any of the categories.
#   A list of good and bad categories (boundary pairs) are saved so that the same fit does not have to be redone.
#   """
#   ncats = len(boundaries) - 1

#   boundary_pairs = [[boundaries[i], boundaries[i+1]] for i in range(len(boundaries)-1)]
#   for pair in boundary_pairs:
#     if pair in known_invalid:
#       return False

#   #helper function to grab part of dataframe belonging to a pair of boundaries (a category)
#   select = lambda df, pair: (df.score > pair[0]) & (df.score <= pair[1])

#   for pair in boundary_pairs:
#     if pair in known_good: continue

#     bm = bkg.mass
#     sidebands = ((bm > pres[0]) & (bm < sr[0])) | ((bm > sr[1]) & (bm < pres[1]))

#     nbkg = (select(bkg, pair) & sidebands).sum() #nbkg events in sidebands
#     nsig = select(sig, pair).sum()

#     #if (nbkg < 10) | (nsig < 10):
#     if (nbkg < 10):
#       known_invalid.append(pair)
#       return False
#     else:
#       known_good.append(pair)

#   return True

def getBoundariesPerformance(bkg, sig, pres, sr, boundaries, i=None):
  nsigs = []
  nbkgs = []
  ncats = len(boundaries)-1

  #return dataframe with only events within category
  select = lambda df, i: df[(df.score > boundaries[i]) & (df.score <= boundaries[i+1])]

  for i in range(ncats):
    assert len(select(bkg, i)) > 0, print(i)
    nsig, nbkg, nbkg_err, bkg_func = performFit(select(sig, i), select(bkg, i), pres, sr)

    # these numbers contribute ~0 sensitivity but help with numerical stuff e.g. zeros in logs
    nbkg = max([nbkg, 0.0001]) 
    nsig = max([nsig, 0.02])

    nsigs.append(nsig)
    nbkgs.append(nbkg)

  limit = calculateExpectedLimit(nsigs, nbkgs)
  ams = AMS(nsigs, nbkgs)

  return limit, ams, nsigs, nbkgs

def parallel(bkg, sig, pres, sr, boundaries):
  """Function to be run in parallel mode"""
  # if isValidBoundaries(bkg, sig, pres, sr, boundaries):
  #  return getBoundariesPerformance(bkg, sig, pres, sr, boundaries)
  # else:
  #  return -1, -1
  return getBoundariesPerformance(bkg, sig, pres, sr, boundaries)

def optimiseBoundary(bkg, sig, pres=(100,180), sr=(115,135), low=0.05, high=1.0, step=0.01, nbounds=1, include_lower=False):
  bm = bkg.mass
  sidebands = ((bm>=pres[0])&(bm<=pres[1])) & ~((bm>=sr[0])&(bm<=sr[1]))
  bkg = bkg[sidebands]

  boundaries_grid = formBoundariesGrid(bkg, low, high, step, nbounds, include_lower)
  valid_boundaries = []
  limits = []
  amss = []

  n = len(boundaries_grid)
  print("Number of boundaries in grid: %d"%n)

  """Parallel approach"""
  from concurrent import futures
  import os
  #with futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
  with futures.ProcessPoolExecutor(max_workers=32) as executor:
    print("Making iterables")
    iterables = [[bkg]*n, [sig]*n, [pres]*n, [sr]*n, boundaries_grid]
    func = parallel
    chunksize = int(n / (os.cpu_count() * 4))
    if chunksize == 0: chunksize = 1
    if chunksize > 1000: chunksize = 1000
    for boundaries, result in tqdm(zip(boundaries_grid, executor.map(func, *iterables, chunksize=chunksize)), total=n):
      #print("boo")
      if result[0] != -1:
        valid_boundaries.append(boundaries)
        limits.append(result[0])
        amss.append(result[1])
  """-----------------"""

  """Single core approach"""
  # for boundaries in tqdm(boundaries_grid):
  #   #if isValidBoundaries(bkg, sig, pres, sr, boundaries):
  #   valid_boundaries.append(boundaries)
  #   limit, ams, nsigs, nbkgs = getBoundariesPerformance(bkg, sig, pres, sr, boundaries)
  #   limits.append(limit)
  #   amss.append(ams)
  """--------------------"""

  limits = np.array(limits)
  amss = np.array(amss)
  optimal_boundaries = valid_boundaries[limits.argmin()]
  optimal_limit = limits.min()
  #optimal_boundaries = valid_boundaries[amss.argmax()]
  #optimal_limit = amss.max()

  #select = lambda df, i: df[(df.score > optimal_boundaries[i]) & (df.score <= optimal_boundaries[i+1])]
  #bkg_func, nbkg_sr = fitBkg(select(bkg, 1), pres, sr, 0.1)
  #plotBkgFit(select(bkg, 1), bkg_func, (100,180), (120,130), "bkg_fits/bkg_fit.png")
  
  return optimal_limit, optimal_boundaries, valid_boundaries, limits, amss

"""----------------------------------------------------------------------------------------------------------------"""
"""---Testing------------------------------------------------------------------------------------------------------"""

def generateToyScores(nbkg=100, nsig=100, lbkg=1, lsig=10):
  nbkg_to_sample = int(1.5 * nbkg * 1/intExp(0, 1, lbkg))
  nsig_to_sample = int(1.5 * nsig * 1/intExp(0, 1, lsig))

  bkg_scores = expon.rvs(size=nbkg_to_sample, scale=1/lbkg)
  sig_scores = -1*expon.rvs(size=nsig_to_sample, scale=1/lsig) + 1

  bkg_scores = bkg_scores[(bkg_scores>0)&(bkg_scores<1)]
  sig_scores = sig_scores[(sig_scores>0)&(sig_scores<1)]

  return bkg_scores[:nbkg], sig_scores[:nsig]

def generateToyData(nbkg=100, nsig=100, l=0.05, mh=125, sig=1, pres=(100,180)):
  #need to sample a greater number of background events since we cut away with preselection
  int_pres = (np.exp(-l*pres[0]) - np.exp(-l*pres[1])) #integral of exponential in preselection window
  nbkg_to_sample = int(1.5 * nbkg * 1/int_pres)
  nsig_to_sample = int(1.5 * nsig)

  #sample mass distributions
  bkg_mass = expon.rvs(size=nbkg_to_sample, scale=1/l)
  sig_mass = norm.rvs(size=nsig_to_sample, loc=mh, scale=sig)

  #apply preselection
  bkg_mass = bkg_mass[(bkg_mass>pres[0])&(bkg_mass<pres[1])][:nbkg]
  sig_mass = sig_mass[(sig_mass>pres[0])&(sig_mass<pres[1])][:nsig]

  bkg_scores, sig_scores = generateToyScores(nbkg, nsig)
  
  #make DataFrames with events given unity weight
  bkg = pd.DataFrame({"mass":bkg_mass, "weight":np.ones(nbkg), "score":bkg_scores})
  sig = pd.DataFrame({"mass":sig_mass, "weight":np.ones(nsig), "score":sig_scores})

  return bkg, sig

def plotSigPlusBkg(bkg, sig, pres, saveas="bkg_sig.png"):
  plt.hist([bkg.mass, sig.mass], bins=50, range=pres, stacked=True, histtype='step', label=["background", "signal"])
  plt.title("Toy experiment")
  plt.xlabel(r"$m_{\gamma\gamma}$")
  plt.ylabel("No. events")
  plt.legend()
  plt.savefig(saveas)
  #plt.show()
  plt.clf()

def plotScores(bkg, sig, optimal_boundaries=None, labels=None, saveas="output_scores.png"):
  plt.hist(bkg.score, bins=50, range=(0,1), weights=bkg.weight, histtype='step', label="bkg", density=True)
  plt.hist(sig.score, bins=50, range=(0,1), weights=sig.weight, histtype='step', label="sig", density=True)

  if optimal_boundaries != None:
    for i, bound in enumerate(optimal_boundaries):
      plt.plot([bound, bound], [0, plt.ylim()[1]], '--', label=labels[i])

  plt.xlabel("Output score")
  plt.ylabel("No. Events (normalised)")
  plt.yscale("log")
  #plt.ylim(top=plt.ylim()[1]*10)
  plt.legend(loc='upper left')
  plt.savefig(saveas)
  #plt.show()
  plt.clf()

def testFit(nbkg=100, nsig=100, l=0.05, pres=(100,180), sr=(120,130)):
  bkg, sig = generateToyData(nbkg, nsig, l=l, pres=pres)
  plotSigPlusBkg(bkg, sig, pres, saveas="test_fit_sig_bkg.png")

  true_nsig = sum(sig[(sig.mass>sr[0])&(sig.mass<sr[1])].weight)
  true_nbkg = sum(bkg[(bkg.mass>sr[0])&(bkg.mass<sr[1])].weight)

  fit_nsig, fit_nbkg, fit_nbkg_err, bkg_func = performFit(sig, bkg, pres, sr)
  print("True (fit) nsig: %d (%f)"%(true_nsig, fit_nsig))
  print("True (fit) nbkg: %d (%f)"%(true_nbkg, fit_nbkg))

  plotBkgFit(bkg, bkg_func, pres, sr, saveas="test_fit_bkg_fit.png")

  limit = calculateExpectedLimit(fit_nsig, fit_nbkg, fit_nbkg_err)
  print("95%% CL limit on mu: %f"%limit)

def testOptimisation(nbkg=100, nsig=100, l=0.05, pres=(100,180), sr=(120,130)):
  bkg, sig = generateToyData(nbkg, nsig, l=l, pres=pres)
  plotSigPlusBkg(bkg, sig, pres, saveas="test_optimisation_sig_bkg.png")

  plotScores(bkg, sig, saveas="test_optimisation_scores_no_boundaries.png")
  optimal_limit, optimal_boundary, boundaries, limits, ams = optimiseBoundary(bkg, sig, low=0.5, high=1.0, nbounds=1)
  boundaries = [b[0] for b in boundaries]
  plotScores(bkg, sig, optimal_boundaries=[boundaries[limits.argmin()], boundaries[ams.argmax()]], labels=["CLs optimal boundary", "AMS optimal boundary"], saveas="test_optimisation_scores_w_boundaries.png")
  
  line = plt.plot(boundaries, (limits-min(limits))/max(limits), label="CLs")
  plt.plot([boundaries[limits.argmin()],boundaries[limits.argmin()]], [0, 1.1], '--', color=line[0]._color)

  for sf in [0.001, 0.01, 0.1, 1, 10,100]:
    sig_scaled = sig.copy()
    sig_scaled.loc[:,"weight"] *= sf
    optimal_limit, optimal_boundary, boundaries, limits, ams = optimiseBoundary(bkg, sig_scaled, low=0.5, high=0.95, step=0.005, nbounds=1)
    boundaries = [b[0] for b in boundaries]

    line = plt.plot(boundaries, (ams-max(ams))/max(-ams), label="AMS sf=%.2f"%sf)
    plt.plot([boundaries[ams.argmax()],boundaries[ams.argmax()]], [0, 1], '--', color=line[0]._color)

    #line = plt.plot(boundaries, (limits-min(limits))/max(limits), label="CLs sf=%.2f"%sf)
    #line = plt.plot(boundaries, limits*sf, label="95%% CL Limit sf=%f"%sf)
    #plt.plot([boundaries[limits.argmin()],boundaries[limits.argmin()]], [0, plt.ylim()[1]], '--', color=line[0]._color)
  
  plt.legend()
  plt.xlabel("Output Score")
  plt.ylabel("Normalised performance metric")
  plt.savefig("test_optimisation_scores_norm_check.png")
  #plt.show()
  plt.clf()

def transformScores(sig, bkg):
  sig.sort_values("score", inplace=True)
  bkg.sort_values("score", inplace=True)
  
  sig_score = sig.score.to_numpy()
  bkg_score = bkg.score.to_numpy()

  sig_cdf = (np.cumsum(sig.weight)/np.sum(sig.weight)).to_numpy()
  
  idx = np.searchsorted(sig_score, bkg_score, side="right")
  
  sig_score = sig_cdf
  bkg_score = sig_cdf[idx]

  sig.loc[:, "score"] = sig_score
  bkg.loc[:, "score"] = bkg_score

if __name__=="__main__":
  testFit(1000, 100, l=0.01)
  testOptimisation(10000, 10000)
