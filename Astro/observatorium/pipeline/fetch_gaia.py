"""Download every Gaia DR3 sample the Observatorium uses. Results cached in cache/*.csv.
Run:  python3 fetch_gaia.py [name ...]"""
import sys, concurrent.futures as cf
from gaia import query

# Quality cuts of Gaia Collaboration, Babusiaux et al. 2018 (A&A 616, A10), used for the Gaia HR diagram.
HRD_CUTS = """parallax_over_error > 10
 AND phot_g_mean_flux_over_error > 50 AND phot_rp_mean_flux_over_error > 20 AND phot_bp_mean_flux_over_error > 20
 AND phot_bp_rp_excess_factor < 1.3 + 0.06*POWER(bp_rp, 2) AND phot_bp_rp_excess_factor > 1.0 + 0.015*POWER(bp_rp, 2)
 AND visibility_periods_used > 8
 AND astrometric_chi2_al/(astrometric_n_good_obs_al - 5) < 1.44*GREATEST(1, EXP(-0.4*(phot_g_mean_mag - 19.5)))"""

COLS = 'source_id, ra, dec, parallax, parallax_error, pmra, pmdec, phot_g_mean_mag, bp_rp, radial_velocity'

def cluster(ra, dec, rad, plx_lo, plx_hi, pmra, pmdec, dpm, gmax=19):
    return f"""SELECT {COLS}, teff_gspphot FROM gaiadr3.gaia_source
 WHERE 1 = CONTAINS(POINT(ra, dec), CIRCLE({ra}, {dec}, {rad}))
 AND parallax BETWEEN {plx_lo} AND {plx_hi}
 AND SQRT(POWER(pmra - ({pmra}), 2) + POWER(pmdec - ({pmdec}), 2)) < {dpm}
 AND phot_g_mean_mag < {gmax} AND bp_rp IS NOT NULL"""

Q = {
  # The solar neighbourhood: everything within 100 pc with clean photometry and astrometry.
  # Gaia Catalogue of Nearby Stars (Gaia Collaboration, Smart et al. 2021): EDR3 = DR3 astrometry and photometry.
  'nearby100': """SELECT source_id, ra, dec, parallax, parallax_error, pmra, pmdec, phot_g_mean_mag,
 phot_bp_mean_mag - phot_rp_mean_mag AS bp_rp, adoptedrv AS radial_velocity, wd_prob, xcoord_50, ycoord_50, zcoord_50, ruwe
 FROM external.gaiaedr3_gcns_main_1
 WHERE parallax > 10 AND parallax > 10*parallax_error AND gcns_prob > 0.38 AND ruwe < 1.4
 AND phot_g_mean_flux_over_error > 50 AND phot_rp_mean_flux_over_error > 20 AND phot_bp_mean_flux_over_error > 20
 AND phot_bp_rp_excess_factor < 1.3 + 0.06*POWER(phot_bp_mean_mag - phot_rp_mean_mag, 2)
 AND phot_bp_rp_excess_factor > 1.0 + 0.015*POWER(phot_bp_mean_mag - phot_rp_mean_mag, 2)""",
  # Clusters (centres, parallaxes and mean proper motions from Gaia Collaboration 2018 / Cantat-Gaudin 2018).
  # Generous cuts on purpose: the player separates members from the field in proper-motion space.
  'pleiades_field': f"""SELECT {COLS} FROM gaiadr3.gaia_source
 WHERE 1 = CONTAINS(POINT(ra, dec), CIRCLE(56.75, 24.12, 3)) AND parallax > 2 AND parallax_over_error > 5 AND phot_g_mean_mag < 18 AND bp_rp IS NOT NULL""",
  'hyades': cluster(66.75, 15.87, 20, 14, 32, 101.0, -28.0, 40, 18),
  'praesepe': cluster(130.10, 19.67, 3, 4.8, 6.0, -35.9, -12.9, 2.0),
  'm67': cluster(132.85, 11.81, 0.6, 0.95, 1.35, -10.97, -2.94, 0.8),
  'ngc104': cluster(6.024, -72.081, 0.35, -0.5, 1.0, 5.25, -2.53, 1.2, 19.5),
  'ngc6397': cluster(265.175, -53.674, 0.4, 0.2, 0.7, 3.30, -17.60, 1.5, 19.5),
  # Classical Cepheids (fundamental mode) from the Gaia DR3 variability pipeline (Ripepi et al. 2023).
  'cepheids': """SELECT c.source_id, g.ra, g.dec, g.parallax, g.parallax_error, c.pf, c.int_average_g, c.int_average_bp, c.int_average_rp,
 c.peak_to_peak_g, g.phot_g_mean_mag, g.bp_rp, g.ruwe
 FROM gaiadr3.vari_cepheid AS c JOIN gaiadr3.gaia_source AS g USING (source_id)
 WHERE c.type_best_classification = 'DCEP' AND c.mode_best_classification = 'FUNDAMENTAL' AND c.pf IS NOT NULL""",
  # Double-lined eclipsing spectroscopic binaries: both velocity amplitudes and the inclination are fitted.
  'eb_sb2': """SELECT n.source_id, n.nss_solution_type, n.period, n.period_error, n.semi_amplitude_primary, n.semi_amplitude_primary_error,
 n.semi_amplitude_secondary, n.semi_amplitude_secondary_error, n.eccentricity, n.arg_periastron, n.inclination, n.inclination_error,
 n.center_of_mass_velocity, n.t_periastron, n.goodness_of_fit, n.significance, g.ra, g.dec, g.parallax, g.phot_g_mean_mag, g.bp_rp
 FROM gaiadr3.nss_two_body_orbit AS n JOIN gaiadr3.gaia_source AS g USING (source_id)
 WHERE n.nss_solution_type IN ('EclipsingSpectro', 'SB2') AND n.semi_amplitude_secondary IS NOT NULL""",
  # Whole-sky star counts on HEALPix level 6 (49152 pixels) from a 2% random subsample of Gaia DR3: the Milky Way.
  'skymap': """SELECT GAIA_HEALPIX_INDEX(6, source_id) AS hpx6, COUNT(*) AS n FROM gaiadr3.gaia_source
 WHERE random_index < 36000000 GROUP BY hpx6""",
}

def run(name):
    df = query(Q[name], cache=f'cache/gaia_{name}.csv', timeout=1800)
    return name, len(df)

if __name__ == '__main__':
    names = sys.argv[1:] or list(Q)
    with cf.ThreadPoolExecutor(1) as ex:
        for fut in cf.as_completed([ex.submit(run, n) for n in names]):
            try:
                print(*fut.result(), flush=True)
            except Exception as e:
                print('FAILED', e, flush=True)
