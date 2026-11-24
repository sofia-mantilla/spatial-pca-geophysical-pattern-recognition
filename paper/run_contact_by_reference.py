"""What makes a selection sit on contacts: the variable, the component count,
or the reference deposit?

Case 1 selects ground near lithological contacts and Case 2 does not, and
Case 2 is half magnetic geometry, so the second variable alone cannot be the
reason. This separates the three candidates by ranking on each half of Case 2
on its own, on Alemao's magnetic geometry at four retained-component counts,
and on Paulo Afonso's at two.

The answer is the reference deposit. Alemao's magnetic geometry gives
34.2 to 37.4 per cent of its selection within 500 m of a contact at every k
tried, inside the 30.0 to 40.6 per cent range of 2,000 random draws. Paulo
Afonso's gives 49.8 per cent at k=2 and 46.6 per cent at k=17. Neither the
variable nor the component count moves it; the reference does.

Run from inside paper/:

    python run_contact_by_reference.py
"""
import sys
sys.path.insert(0,"/Users/sofiamantillasalas/Documents/GitHub_Sofia/EroCopper_Xcalibur_Project/Code/Spatial_PCA_Multivariate_Repo/paper")
from pathlib import Path
import numpy as np, geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import Point, box
from shapely.strtree import STRtree
import case1_uni_repro as C1, numpy_repro_concat as R
from run_era_shares import clear_of_deposits
from run_geology_vs_baselines import deposit_boxes

ROOT=Path("/Users/sofiamantillasalas/Documents/GitHub_Sofia/EroCopper_Xcalibur_Project/Code/Spatial_PCA_Multivariate_Repo")
LITO=ROOT/"data/Carajas_Brazil_Structures/sig_arim_carajas_versao_preliminar_vr/Arim_Carajas/Litologia/arim_carajas_lito.shp"
tmi,_u,ox,oy=R.load_cropped(); H,W=tmi.shape
tr=from_origin(ox,oy,R.PX,R.PX); area=box(ox,oy-H*R.PX,ox+W*R.PX,oy)
g=gpd.clip(gpd.read_file(LITO).to_crs("EPSG:32722"),area); g=g[~g.geometry.is_empty]
diss=g.dissolve("SIGLA_UNID").reset_index()
lines=[gm.boundary for gm in diss.geometry if not gm.is_empty]
tree=STRtree(lines)
def dist_at(x,y):
    """Exact point-to-boundary distance. A rasterised transform at 200 m with
    all_touched places every boundary up to one cell too close."""
    p=Point(x,y); return p.distance(lines[tree.nearest(p)])
foot=rasterize(list(g.geometry),out_shape=(H,W),transform=tr,fill=0,default_value=1).astype(bool)
deposits=deposit_boxes()

def stats(order,idx,wshape,origin,dep):
    oxx,oyy=origin; wh,ww=wshape; v=[]
    for r,c in clear_of_deposits(order,idx,wshape,origin,dep,deposits):
        rr=min(r+wh//2,H-1); cc=min(c+ww//2,W-1)
        if foot[rr,cc]:
            v.append(dist_at(oxx+(c+ww/2)*R.PX, oyy-(r+wh/2)*R.PX))
    v=np.array(v,float)
    return len(v), np.median(v), (v<=500).mean()

print("%-42s %5s %8s %8s" % ("ranking","n","median","<=500m"))
# Case 2 and its halves, all reference Alemao, k_TMI=2, k_U=8
for lab,a in (("Case 2 combined, alpha=0.503",0.503),
              ("Case 2, TMI alone (k=2)",1.0),
              ("Case 2, U alone (k=8)",0.0)):
    o,i,ws,og,d=R.rank_concat(3,2,8,a)
    n,m,p=stats(o,i,ws,og,d); print("%-42s %5d %7.0fm %7.1f%%" % (lab,n,m,100*p))
# does the TMI component count explain it? same reference, same window
for k in (2,8,17,25):
    o,i,ws,og,d=R.rank_concat(3,k,8,1.0)
    n,m,p=stats(o,i,ws,og,d); print("%-42s %5d %7.0fm %7.1f%%" % ("Alemao TMI alone, k=%d"%k,n,m,100*p))
# Case 1 for reference, and Case 1 at low k
for k in (2,17):
    o,i,ws,og,d=C1.rank_spca(k)
    n,m,p=stats(o,i,ws,og,d); print("%-42s %5d %7.0fm %7.1f%%" % ("Paulo Afonso TMI alone, k=%d"%k,n,m,100*p))
print("\n(random, exact geometry: mean 919m / 32.5%, 5-95%: 27.6-37.9%)")
