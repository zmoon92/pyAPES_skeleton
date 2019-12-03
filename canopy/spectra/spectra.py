"""
- load spectra
- re-bin spectra
- correct binned spectra for APES

"""

import os
_this_dir = os.path.dirname(os.path.realpath(__file__))

import numpy as np
from scipy import interpolate, integrate
import xarray as xr

def load_default_sp2():
    """
    wl_sp2 : assume actual values in the spectrum (wavelength band edges)
    SI_dr, SI_df : spectral direct and diffuse irradiance at those wl

    if we assume that the loaded spectrum traces out the true spectrum (which it is meant to),
    then we need to calculate values at wavelength bin centers
    since this is what we want to use in our calculations.

    if resolution is high, using the left wl and value there won't make big difference
    but if is not...
    """

    wl_sp2, SI_dr, SI_df = np.loadtxt(f'{_this_dir}/SPCTRAL2_xls_default-spectrum.csv',
        unpack=True, delimiter=',', skiprows=1)
    
    wl = wl_sp2
    dwl = np.diff(wl)

    wl1 = wl[:-1]
    # wl2 = wl[1:]
    wlc = wl1 + 0.5*dwl

    SI_dr_c = (SI_dr[:-1]+SI_dr[1:])/2.  # spectral irradiance for bin center as average of edge values
    I_dr = SI_dr_c*dwl  # integration: spectral irradiance -> bin-level irradiance
    
    SI_df_c = (SI_df[:-1]+SI_df[1:])/2.
    I_df = SI_df_c*dwl
    
    dimsx12 = ('wl')  # band edges / actual spectrum tracing
    dimsx = ('wlc')  # band centers
    Sunits = 'W m^-2 um^-1'  # spectral units
    units = 'W m^-2'  # non-spectral

    attrs = {}
    coords = {
        'wl': (dimsx12, wl, {'units': 'um', 'long_name': 'wavelength'}),
        'wlc': (dimsx, wlc, {'unit': 'um', 'long_name': 'wavelength of waveband center'})
        }
    dset = xr.Dataset(\
        coords=coords,
        data_vars={\
            'SI_dr': (dimsx12, SI_dr,  {'units': Sunits, 'long_name': 'spectral direct irradiance'}),
            'SI_df': (dimsx12, SI_df,  {'units': Sunits, 'long_name': 'spectral diffuse irradiance'}),
            'I_dr':  (dimsx, I_dr,  {'units': units, 'long_name': 'direct irradiance'}),
            'I_df':  (dimsx, I_df,  {'units': units, 'long_name': 'diffuse irradiance'}),
            'dwl':   (dimsx, dwl, {'units': 'um', 'long_name': 'waveband width'})            
            },
        attrs=attrs,
    )
    return dset


def load_default_ps5():
    """
    
    """
    wl_nm, r, t = np.loadtxt(f'{_this_dir}/sample_PROSPECT.txt', unpack=True)
    wl = wl_nm/1000.  # nm->um
    # wl_pro_min = wl[0]
    # wl_pro_max = wl[-1]
    
    # dwl = np.diff(wl)
    
    # wl1 = wl[:-1]
    # wl2 = wl[1:]
    # wlc = wl1 + 0.5*dwl
    
    attrs = {}
    dims = ('wl')
    coords = {
        'wl': (dims, wl, {'units': 'um', 'long_name': 'wavelength'}),
        }
    dset = xr.Dataset(\
        coords=coords,
        data_vars={\
            'r':  (dims, r,  {'units': '', 'long_name': f'leaf reflectance'}),
            't':  (dims, t,  {'units': '', 'long_name': f'leaf transmittance'})
            },
        attrs=attrs,
    )
    return dset


def load_default_soil():
    """From PS5
    https://github.com/jgomezdans/prosail/blob/master/prosail/soil_reflectance.txt
    """
    rho_soil_dry, rho_soil_wet = np.loadtxt(f'{_this_dir}/sample_soil.txt', unpack=True)
    # ^ in um
    
    wl = load_default_ps5()['wl']
    
    attrs = {}
    dims = ('wl')
    coords = {
        'wl': (dims, wl, {'units': 'um', 'long_name': 'wavelength'}),
        }
    dset = xr.Dataset(\
        coords=coords,
        data_vars={\
            'rho_soil':  (dims, rho_soil_dry,  {'units': '', 'long_name': f'soil reflectance'}),
            },
        attrs=attrs,
    )
    return dset


def smear_trapz_interp(x, y, xnew):
    """Here we assume that we have y(x) from the true spectrum
    i.e., x = band edges and we know y(x), not y for the bands/bins 
    
    As with smear_TUV,
    this method conserves the band integral,
    i.e., xnew*dwl_xnew will give the same integral
    that the spectral integral over the same region did in the higher resolution

    Returns
    -------
    ynew : 
        values of y in the bands defined by edges xnew
    """
    
    # dx = np.diff(x)
    # x1 = x[:-1]
    x2 = x[1:]
    # xm = x1 + 0.5*dx  # band midpoints in the original resolution
    
    xnew1 = xnew[:-1]  # left edges of new bands
    xnew2 = xnew[1:]   # right edges of new bands
    dxnew = np.diff(xnew)  # new band widths
    # xmnew = xnew1 + 0.5*dxnew  # band midpoints in the new resolution

    y_int = integrate.cumtrapz(y, x)  # cumulative discrete integral in original resolution
    
    #> fit an interpolator to the integral
    #  so that we can be more accurate smearing when the change in res is not much
    #y_int_interp = interpolate.CubicSpline(xm, y_int)
    y_int_interp = interpolate.CubicSpline(x2, y_int)
    # ^ here we use right band edge
    
    # y_interp = y_int_interp.derivative(1)  # first derivative

    return (y_int_interp(xnew2)-y_int_interp(xnew1))/dxnew  # int(y) -> y


def distribute_crt_quantities(nbands, banddefn, 
    Idr0, Idf0, albL0, albS0
    ):
    """ 
    Using the sample spectra, distribute initial values for a given band
    (most likely PAR or NIR) among sub bands while conserving the integral.

    Args
    ----
    nbands : int
        Number of (equally spaced) bands to use
    banddefn : array_like, size 2
        band left and right bounds (um)
    Idr0, Idf0 : float
        direct, diffuse irradiance (W/m^2)
    albL0, albS0 : float
        leaf, soil albedo

    Returns
    -------
    Idr, Idf, albL, albS : array_like, shape (nbands, )

    Notes
    -----
    This would be improved by using solar, leaf, and soil base spectra more 
    applicable to the site. These would more likely be calculated in advance,  
    e.g., using PROSPECT for leaf and an atmos RT model for solar spectra,
    and given as model inputs. This method is an attempt to approximate some of 
    the influences of those spectra while keeping things simple. 

    """
    ds_solar = load_default_sp2()
    ds_leaf = load_default_ps5()
    ds_soil = load_default_soil()

    wl_max = min((ds_solar.wl.max().values, 
        ds_leaf.wl.max().values, ds_soil.wl.max().values))
    # can't go beyond this or they won't overlap 
    # 2.5 um is the max for ps5 and the soil
    # 4 um for sp2

    bdf = np.asarray(banddefn)
    assert( bdf.size == 2 )
    wla, wlb = bdf[0], bdf[1]
    if wlb > wl_max:
        wlb = wl_max
    #print(wla, wlb)

    #> set up new grid (constant dwl)
    x12_new = np.linspace(wla, wlb, nbands+1)
    # x1_new = x12_new[:-1]
    # x2_new = x12_new[1:]
    dx_new = np.diff(x12_new)
    # x_new = x1_new + 0.5*dx_new

    #> first light
    wl0 = ds_solar['wl']
    wl_in_bound = (wl0 >= wla) & (wl0 <= wlb)
    #ds = ds_solar.sel(dict(wl=wl_in_bound))
    ds = ds_solar.sel(dict(wl=wl0[wl_in_bound]))  # could also use .where(wl_in_bound, drop=True)
    x = ds['wl']
    SI_dr = ds['SI_dr']
    SI_df = ds['SI_df']
    SI_dr_new = smear_trapz_interp(x, SI_dr, x12_new)
    SI_df_new = smear_trapz_interp(x, SI_df, x12_new)
    I_dr_new = SI_dr_new * dx_new
    I_df_new = SI_df_new * dx_new

    #> now leaf properties
    wl0 = ds_leaf['wl']
    wl_in_bound = (wl0 >= wla) & (wl0 <= wlb)
    #ds = ds_leaf.sel(dict(wl=wl_in_bound))
    ds = ds_leaf.sel(dict(wl=wl0[wl_in_bound]))
    x = ds['wl']
    r = ds['r']
    t = ds['t']
    r_new = smear_trapz_interp(x, r, x12_new) / dx_new  # for the albedos, want band avg not sum!
    t_new = smear_trapz_interp(x, t, x12_new) / dx_new
    albL_new = (r_new + t_new)/2  # APES wants leaf albedo, not leaf-level refl and tran

    #> now soil
    wl0 = ds_soil['wl']
    wl_in_bound = (wl0 >= wla) & (wl0 <= wlb)
    #ds = ds_soil.sel(dict(wl=wl_in_bound))
    ds = ds_soil.sel(dict(wl=wl0[wl_in_bound]))
    x = ds['wl']
    rho_soil = ds['rho_soil']
    # rho_soil_new = smear_trapz_interp(x, rho_soil, x12_new)
    albS_new = smear_trapz_interp(x, rho_soil, x12_new) / dx_new

    #> correct the re-binned spectra for consistency with the input values

    def _correct_for_sum(new_array, orig_scalar):
        c = orig_scalar/new_array.sum()  # correction factor
        corrected = c * new_array
        assert( np.isclose(corrected.sum(), orig_scalar) )
        return corrected

    def _mean_over_x(array):
        """even with variable grid"""
        return (array*dx_new).sum() / dx_new.sum()
        # return (array*dx_new).mean() / dx_new.sum()

    def _correct_for_mean(new_array, orig_scalar):
        c = orig_scalar / _mean_over_x(new_array)
        corrected = c * new_array
        if np.any(np.isnan(corrected)):
            print('nan in albedo!')
            print('values:', corrected)
            print('uncorr:', new_array)
            print('ref   :', orig_scalar)
        if np.any(corrected > 1.):
            print('> 1 in albedo!')
            print('values:', corrected)
            print('dx_new:', dx_new)
            print('uncorr:', new_array)
            print('ref   :', orig_scalar)
        assert( np.isclose(_mean_over_x(corrected), orig_scalar) )
        return corrected

    input_values = [Idr0, Idf0, albL0, albS0]
    rebinned_spectra = [I_dr_new, I_df_new, albL_new, albS_new]
    rebinned_spectra_corrected = []
    for input_value, rebinned_spectrum in zip(input_values, rebinned_spectra):
        if input_value in [Idr0, Idf0]:
            #print('correcting for sum')
            corrected = _correct_for_sum(rebinned_spectrum, input_value)
        else:
            #print('correcting for mean')
            corrected = _correct_for_mean(rebinned_spectrum, input_value)
            #print('corrected:', corrected)
        rebinned_spectra_corrected.append(corrected)

    # return Idr, Idf, albL, albS
    return rebinned_spectra_corrected



