The testing inputs and labels goes into this folder
the input csv file contains features given below from gaia data. We use all features except source_id as the input of the model
`source_id,ra,dec,parallax,pmra,pmdec,radial_velocity,parallax_error,pmra_error,pmdec_error,radial_velocity_error,parallax_pmra_corr,parallax_pmdec_corr,pmra_pmdec_corr`
The output features are,
`source_id,dph_med,dph_std,vph_med,vph_std,tph_med,tph_std,tph_dph_corr,tph_vph_corr,dph_vph_corr`
calculated using numerical integration
