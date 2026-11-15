# Firechron — Harmonized active-fire calendar (MODIS + VIIRS)

Project by team **BurnInHell** for the **NASA Space Apps Challenge 2026** (Mendoza local event), challenge
*Harmonization of MODIS and VIIRS Hot Spots*. Not an official NASA tool.

Firechron harmonizes 25+ years of satellite active-fire hotspots over South America into one
consistent series, so the sensor of origin no longer matters, and shows it as an interactive
burning-activity calendar: pick an area and a period, see which months and years burn most,
compare the current month with the historical median, and download the data.

## Run the app

```bash
pip install -r app/requirements.txt
streamlit run app/app.py
```

## Method (summary)

1. **Data**: NASA FIRMS active-fire hotspots, South America, Nov 2000 – present:
   MODIS Collection 6.1 (Terra + Aqua, 1 km) and VIIRS 375 m (Suomi NPP, NOAA-20); archive (standard) + NRT.
2. **Cleaning** (`scripts/limpieza_firms.py`): common schema, common confidence scale, UTC timestamps,
   1-km grid cell per hotspot, archive/NRT de-duplication.
3. **Harmonization** (`scripts/armonizar_firms.py`): nominal/high confidence, vegetation fires only,
   static sources removed; hotspots counted as **1-km cell-days** (removes the 375 m vs 1 km pixel effect);
   region-specific (1°×1°) factors calibrated on the 2012–2017 MODIS/VIIRS overlap and validated on
   2018–2020 (annual total error < 2 %); NOAA-20 bridged to S-NPP after S-NPP's last data (Nov 2026).
4. **App data** (`app/preparar_datos_app.py`): aggregated to 0.1° × day for the web app.

## Data sources and references

- NASA FIRMS — https://earthdata.nasa.gov/firms
  *We acknowledge the use of data and/or imagery from NASA's Fire Information for Resource Management
  System (FIRMS) (https://earthdata.nasa.gov/firms), part of NASA's Land, Atmosphere Near real-time
  Capability for Earth observations (LANCE) (https://earthdata.nasa.gov/lance) and NASA's Earth Science
  Data and Information System (ESDIS).*
- MODIS Collection 6.1 Active Fire product (MCD14) and VIIRS 375 m Active Fire product (VNP14IMG / VJ114IMG).
- Basemap: © OpenStreetMap contributors (ODbL).
- News: Google News public RSS search feed (headlines and links only), GDELT Project DOC 2.0 API (fallback), Wikipedia (CC BY-SA); place names from Natural Earth admin-1 (public domain).
- Place search: OpenStreetMap Nominatim (© OpenStreetMap contributors, ODbL).
- Forecast inputs: ERA5-Land monthly (Copernicus C3S, CC-BY 4.0) and NOAA CPC ONI index.
- Design: NASA Space Apps brand colors; Fira Sans typeface (SIL OFL).
- Software (open source): Python, pandas, NumPy, PyArrow, DuckDB, matplotlib, Streamlit, streamlit-folium,
  Folium/Leaflet, Plotly, Shapely.
- AI assistance: Claude (Anthropic) was used to help write code and documentation; all results were run
  and reviewed by the team.

## License

Code: MIT (see `LICENSE`). FIRMS data are NASA open data.
