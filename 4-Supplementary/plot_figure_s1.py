
from pathlib import Path
from hashlib import sha256
import os,csv,json,argparse

parser=argparse.ArgumentParser(description="Draw Figure S1 from saved applicability diagnostics.")
parser.add_argument("--diagnostics",type=Path,help="Path to the diagnostics CSV.")
args=parser.parse_args()

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
QA=ROOT/'generated/Figure_S1'
QA.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(QA/'matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import numpy as np

SOURCE=args.diagnostics.expanduser().resolve() if args.diagnostics else Path(os.environ.get('QSPR_AD_RESULTS_DIR', str(ROOT/'3-Conditioning Runtime and Applicability'/'generated/results_csv')))/'full_fit_applicability_diagnostics.csv'
PROPERTIES=[
 ('Boiling Point (BP)','Boiling point (BP)'),
 ('Flash Point (FP)','Flash point (FP)'),
 ('Molar Volume (MV)','Molar volume (MV)'),
 ('Polarizability (Pol)','Polarizability (Pol)'),
 ('Molar Refractivity (MR)','Molar refractivity (MR)')]
COLORS=['#20A43A','#FF9D00','#D62728']
MARKERS=['o','^','D']

def main():
    digest=sha256(SOURCE.read_bytes()).hexdigest()
    with SOURCE.open(encoding='utf-8-sig',newline='') as f:
        rows=[r for r in csv.DictReader(f) if r['Descriptor']=='DegEntropy' and r['Order']=='3']
    assert len(rows)==1105
    plt.rcParams.update({
        'font.family':'DejaVu Sans','font.size':9.5,'axes.titlesize':10.0,
        'axes.labelsize':10.0,'xtick.labelsize':8.8,'ytick.labelsize':8.8,
        'axes.edgecolor':'#666666','axes.linewidth':.65,
        'xtick.color':'#333333','ytick.color':'#333333',
        'xtick.major.size':3,'ytick.major.size':3,
        'xtick.major.width':.6,'ytick.major.width':.6,
        'pdf.fonttype':42,'ps.fonttype':42,'mathtext.fontset':'dejavusans',
        'savefig.facecolor':'white','savefig.dpi':300})
    fig=plt.figure(figsize=(6.27,8.10),facecolor='white')
    gs=fig.add_gridspec(3,2,left=.112,right=.985,bottom=.085,top=.952,
                        wspace=.42,hspace=.53)
    slots=[(0,0),(0,1),(1,0),(1,1),(2,0)]
    verification=[]
    annotations=[]

    offsets={
      ('BP',179):(30,-12),('BP',198):(30,4),('BP',203):(7,17),
      ('Pol',64):(27,15),('Pol',129):(27,1),
      ('MV',10):(8,7),('BP',165):(8,-7),('FP',183):(8,-7),
      ('Pol',200):(10,-1)}
    for i,((prop,title),(row,col)) in enumerate(zip(PROPERTIES,slots)):
        ax=fig.add_subplot(gs[row,col])
        part=[r for r in rows if r['Property']==prop]
        assert len(part)==221
        x=np.array([float(r['Leverage']) for r in part])
        y=np.array([float(r['Internally_studentized_residual']) for r in part])
        high=np.array([r['High_leverage']=='True' for r in part])
        large=np.array([r['Large_residual_diagnostic']=='True' for r in part])
        ids=np.array([int(r['Serial Number']) for r in part])
        cutoff=float(part[0]['Leverage_cutoff'])
        assert all(float(r['Leverage_cutoff'])==cutoff for r in part)
        groups=[~(high|large),high&~large,large]
        plotted=[]
        for category,mask in enumerate(groups):
            coll=ax.scatter(x[mask],y[mask],s=[18,34,30][category],
                            c=COLORS[category],marker=MARKERS[category],
                            alpha=[.86,1.0,1.0][category],
                            edgecolors='#1A1A1A',linewidths=[.32,.45,.45][category],
                            zorder=3+category)
            assert np.array_equal(np.asarray(coll.get_offsets()),np.column_stack([x[mask],y[mask]]))
            plotted.extend(ids[mask].tolist())
        assert sorted(plotted)==sorted(ids.tolist())
        ax.axvline(cutoff,color='#2455A4',linestyle=(0,(4,2.6)),linewidth=1.0,zorder=2)
        for level in [-3,3]:
            ax.axhline(level,color='#C82929',linestyle=(0,(4,2.6)),linewidth=1.0,zorder=2)
        ax.axhline(0,color='#A0A6AD',linewidth=.55,zorder=1)
        ax.set_xlim(-.018,.61);ax.set_ylim(-6.0,7.0)
        ax.set_xticks([0,.2,.4,.6])
        ax.set_xticklabels(['0.0','0.2','0.4','0.6'])
        ax.set_yticks([-6,-3,0,3,6])
        ax.grid(axis='both',color='#D8DDE3',linewidth=.55,zorder=0)
        ax.tick_params(pad=2.8)
        ax.set_title(f'({chr(97+i)})  {title}',loc='left',fontweight='bold',pad=7)
        abbr=prop.rsplit('(',1)[1].rstrip(')')
        labelmask=(x>.15)|large
        for j in np.flatnonzero(labelmask):
            serial=int(ids[j]);dx,dy=offsets.get((abbr,serial),(6,6))
            ha='left'
            if serial==106:dx,dy=-6,7;ha='right'
            label=ax.annotate(str(serial),(x[j],y[j]),xytext=(dx,dy),
                      textcoords='offset points',fontsize=8.0,color='#202020',
                      ha=ha,va='center',annotation_clip=False,zorder=8,
                      arrowprops={'arrowstyle':'-','color':'#90969D','linewidth':.45,
                                  'shrinkA':1.5,'shrinkB':2.5},
                      path_effects=[pe.withStroke(linewidth=1.8,foreground='white')])
            annotations.append((ax,label,serial))
        verification.append({
          'property':prop,'source_records':221,'plotted_records':len(plotted),
          'serial_order_in_source':ids.tolist(),
          'plot_coordinates_and_flags':[
            {'serial':int(s),'leverage':float(h),'studentized_residual':float(r),
             'high_leverage':bool(hi),'large_residual':bool(la)}
            for s,h,r,hi,la in zip(ids,x,y,high,large)],
          'high_leverage_count':int(high.sum()),'large_residual_count':int(large.sum()),
          'joint_count':int((high&large).sum()),'leverage_threshold':cutoff,
          'residual_thresholds':[-3,3],
          'annotated_serials':ids[labelmask].tolist(),
          'same_annotation_rule_as_original':'leverage > 0.15 OR saved large-residual flag'})
    fig.text(.55,.025,r'Leverage, $h_i$',ha='center',va='center',fontsize=10.5)
    fig.text(.018,.53,r'Internally studentized residual, $r_i$',ha='center',
             va='center',rotation='vertical',fontsize=10.5)
    marker_handles=[
        Line2D([],[],linestyle='',marker=marker,color=color,markeredgecolor='#1A1A1A',
               markeredgewidth=.45,markersize=size,label=label)
        for marker,color,size,label in zip(MARKERS,COLORS,[5.0,6.0,5.5],
            ['Within diagnostic limits','High leverage only',r'Residual warning ($|r_i|>3$)'])]
    threshold_handles=[
        Line2D([],[],color='#2455A4',linestyle=(0,(4,2.6)),linewidth=1.0,
               label=r'$h^*=12/221=0.0543$'),
        Line2D([],[],color='#C82929',linestyle=(0,(4,2.6)),linewidth=1.0,
               label=r'$r_i=\pm3$')]
    key=fig.add_subplot(gs[2,1]);key.set_axis_off()
    key.text(.02,.97,'Diagnostic key',transform=key.transAxes,
             fontsize=10.2,fontweight='bold',va='top')
    key.legend(handles=marker_handles+threshold_handles,loc='upper left',
               bbox_to_anchor=(.0,.85),frameon=False,fontsize=8.6,
               labelspacing=.9,handlelength=2.0,handletextpad=.6,borderaxespad=0)
    key.text(.02,.03,'221 compounds per panel',
             transform=key.transAxes,fontsize=8.8,color='#444444',va='bottom',
             linespacing=1.6)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    bounds=[]
    for ax,label,serial in annotations:
        box=label.get_window_extent(renderer)
        assert fig.bbox.contains(box.x0,box.y0) and fig.bbox.contains(box.x1,box.y1),(serial,'label outside canvas')
        bounds.append({'serial':serial,'bbox_px':list(box.bounds)})
    target=QA/'Figure_S1_Williams_Diagnostics'
    fig.savefig(target.with_suffix('.pdf'),metadata={'Title':'Applicability-domain plots for degree-entropy cubic models','Author':'Irfan Haider et al.'})
    fig.savefig(target.with_suffix('.png'),dpi=600)
    plt.close(fig)
    assert sha256(SOURCE.read_bytes()).hexdigest()==digest
    result={'source_CSV':SOURCE.name,'source_CSV_sha256':digest,
       'no_model_fitting_or_diagnostic_recalculation':True,
       'all_1105_plotted_coordinates_match_saved_CSV_exactly':True,
       'all_original_point_categories_and_annotation_selection_preserved':True,
       'common_axis_limits':{'leverage':[-.018,.61],'studentized_residual':[-6,7]},
       'models':verification,'annotation_bounds':bounds}
    (QA/'plot_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Plotted {len(rows)} stored observations in five panels; all values, flags and thresholds preserved.')

if __name__=='__main__':main()
