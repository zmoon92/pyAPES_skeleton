

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

    wl_sp2, SI_dr, SI_df = np.loadtxt('SPCTRAL2_xls_default-spectrum.csv',
        unpack=True, delimiter=',', skiprows=1)


#    x12 = wl_sp2
#    x1 = x12[:-1]
#    x2 = x12[1:]
#    dx = x2 - x1
#    x = x1 + 0.5*dx
    
    
    wl = wl_sp2
    dwl = np.diff(wl)

    wl1 = wl[:-1]
    wl2 = wl[1:]
    wlc = wl1 + 0.5*dwl
    
#    I_dr = integrate.cumtrapz(SI_dr, wl_sp2)#/dwl
#    I_df = integrate.cumtrapz(SI_df, wl_sp2)#/dwl
    
    SI_dr_c = (SI_dr[:-1]+SI_dr[1:])/2.
    I_dr = SI_dr_c*dwl
    
    SI_df_c = (SI_df[:-1]+SI_df[1:])/2.
    I_df = SI_df_c*dwl    
    
    dimsx12 = ('wl')  # band edges / actual spectrum tracing
    dimsx = ('wlc')  # band centers
    Sunits = 'W m^-2 um^-1'
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
    wl_nm, r, t = np.loadtxt('sample_PROSPECT.txt', unpack=True)
    wl = wl_nm/1000.  # nm->um
    wl_pro_min = wl[0]
    wl_pro_max = wl[-1]
    
    dwl = np.diff(wl)
    
    wl1 = wl[:-1]
    wl2 = wl[1:]
    wlc = wl1 + 0.5*dwl
    
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
    rho_soil_dry, rho_soil_wet = np.loadtxt('sample_soil.txt', unpack=True)
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
    i.e., x = band edges, but we know what y(x), not y for the band 
    
    As with smear_TUV,
    this method conserves the band integral,
    i.e., xnew*dwl_xnew will give the same integral
    that the spectral integral over the same region did in the higher resolution

    Returns
    -------
    ynew : 
        values of y in the bands defined by edges xnew
    """
    
    dx = np.diff(x)
    x1 = x[:-1]
    x2 = x[1:]
    xm = x1 + 0.5*dx  # band midpoints in the original resolution
    
    xnew1 = xnew[:-1]  # left edges of new bands
    xnew2 = xnew[1:]   # right edges of new bands
    dxnew = np.diff(xnew)  # new band widths
    xmnew = xnew1 + 0.5*dxnew  # band midpoints in the new resolution

    y_int = integrate.cumtrapz(y, x)  # cumulative discrete integral in original resolution
    
    #> fit an interpolator to the integral
    #  so that we can be more accurate smearing when the change in res is not much
#    y_int_interp = interpolate.CubicSpline(xm, y_int)
    y_int_interp = interpolate.CubicSpline(x2, y_int)
    # ^ we use right band edge
    
#    y_interp = y_int_interp.derivative(1)  # first derivative

    return (y_int_interp(xnew2)-y_int_interp(xnew1))/dxnew





def distribute_crt_quantities(nbands, banddefn, ):
    """ """
    return



