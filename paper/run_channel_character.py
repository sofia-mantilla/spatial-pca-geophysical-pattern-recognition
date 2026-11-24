"""Is TMI an edge channel and radiometric U a composition channel?

Case 1 selects ground near lithological contacts and Case 2 does not. Before
reading either as a statement about prospectivity, it is worth knowing what
each variable can respond to.

A magnetic anomaly exists because susceptibility changes from one rock to
another, so a magnetic signal lives at the edge of a body. Radiometric uranium
reports the uranium in the top few tens of centimetres of rock, which is a
property of the rock itself. If that is right, a ranking built on TMI geometry
is drawn to boundaries by physics, and one that adds uranium is drawn to
bodies.

Two measurements test it, on the ground the ARIM 1:250,000 map covers:

  gradient against distance to a mapped contact, normalised by each grid's
  own median, which says whether the signal concentrates at boundaries

  the share of each grid's variance explained by which mapped unit a cell
  falls in, which says whether the signal is a property of the unit

Run from inside paper/:

    python run_channel_character.py
"""
import sys
sys.path.insert(0,"/Users/sofiamantillasalas/Documents/GitHub_Sofia/EroCopper_Xcalibur_Project/Code/Spatial_PCA_Multivariate_Repo/paper")
from pathlib import Path
import numpy as np, geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt
from shapely.geometry import box
import numpy_repro_concat as R

ROOT=Path("/Users/sofiamantillasalas/Documents/GitHub_Sofia/EroCopper_Xcalibur_Project/Code/Spatial_PCA_Multivariate_Repo")
LITO=ROOT/"data/Carajas_Brazil_Structures/sig_arim_carajas_versao_preliminar_vr/Arim_Carajas/Litologia/arim_carajas_lito.shp"

tmi,u,ox,oy=R.load_cropped(); H,W=tmi.shape
tr=from_origin(ox,oy,R.PX,R.PX); area=box(ox,oy-H*R.PX,ox+W*R.PX,oy)
g=gpd.clip(gpd.read_file(LITO).to_crs("EPSG:32722"),area); g=g[~g.geometry.is_empty]
names=sorted(set(g["SIGLA_UNID"].fillna("?"))); code={s:i+1 for i,s in enumerate(names)}
grid=rasterize([(gm,code[s]) for gm,s in zip(g.geometry,g["SIGLA_UNID"].fillna("?"))],
               out_shape=(H,W),transform=tr,fill=0,dtype="int32")
diss=g.dissolve("SIGLA_UNID").reset_index()
contacts=rasterize([gm.boundary for gm in diss.geometry if not gm.is_empty],
                   out_shape=(H,W),transform=tr,fill=0,default_value=1,all_touched=True).astype(bool)
dist=distance_transform_edt(~contacts)*R.PX
inside=grid>0

def grad(a):
    a=np.where(np.isfinite(a),a,np.nan)
    gy,gx=np.gradient(a)
    return np.hypot(gy,gx)

print("cells inside the ARIM footprint: %d" % inside.sum())
for nm,a in (("TMI",tmi),("radiometric U",u)):
    gm=grad(a.astype(float))
    ok=inside & np.isfinite(gm)
    med=np.nanmedian(gm[ok])
    print("\n%s: median |gradient| per cell = %.4g" % (nm,med))
    print("   distance to contact      mean |grad| / median")
    for lo,hi in ((0,250),(250,500),(500,1000),(1000,2000),(2000,100000)):
        sel=ok & (dist>=lo) & (dist<hi)
        if sel.sum()>500:
            print("     %5d-%-6d m   n=%7d   %.2f" % (lo,hi,sel.sum(), np.nanmean(gm[sel])/med))
    # how much of the variance is between units vs within
    vals=a.astype(float)[ok]; labs=grid[ok]
    gmean=np.nanmean(vals); tot=np.nansum((vals-gmean)**2)
    btw=0.0
    for uq in np.unique(labs):
        m=labs==uq
        if m.sum()>50:
            btw += m.sum()*(np.nanmean(vals[m])-gmean)**2
    print("   share of variance explained by which unit you are in: %.1f%%" % (100*btw/tot))
