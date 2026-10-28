"""Visual treatment based on the author's original figures. Data stay fixed."""
import hashlib,json,re
from pathlib import Path
import numpy as np
from matplotlib.colors import to_rgba,PowerNorm
from matplotlib.patches import Ellipse,Rectangle,Polygon,FancyBboxPatch,Arc
from matplotlib.lines import Line2D
from matplotlib.text import Text

PINK='#f04aca'
BLUE='#2929ed'
ORANGE='#ec8826'
GRAY='#454545'
MODEL_COLORS={'random':'#111111','bias':'#8e22e8','simple':'#2525ec','simple_no_swap':'#8585f5','reward':'#df8453','reward_no_swap':'#edb08d','full':'#ee202b','full_no_swap':'#f1959b'}
QA=[]

def letter(ax):
    return next((t.get_text() for t in ax.texts if getattr(t,'_panel_letter',False)),'')

def axis_by_letter(fig,value):
    return next(ax for ax in fig.axes if letter(ax)==value)

def scientific_arrays(fig,name):
    records=[]
    for ax in fig.axes:
        if name=='Figure1' and letter(ax)=='A':continue
        for line in ax.lines:
            records.append((line,'line',np.array(line.get_xydata(),copy=True)))
        for collection in ax.collections:
            records.append((collection,'offsets',np.array(collection.get_offsets(),copy=True)))
        for image in ax.images:records.append((image,'image',np.array(image.get_array(),copy=True)))
    return records

def symbol(text):
    return re.sub(r'(?<![A-Za-z])F(?=[ ,/]|$)','♀',re.sub(r'(?<![A-Za-z])M(?=[ ,/]|$)','♂',text))

def recolor_text(text):
    value=text.get_text()
    if value=='♀':text.set_color(PINK)
    elif value=='♂':text.set_color('#111111')

def bold_symbol(text):
    text.set_fontfamily('DejaVu Sans');text.set_fontweight('bold')

def schematic(ax):
    """Small vector two-bottle assay illustration, with the same protocol."""
    for t in list(ax.texts):
        if not getattr(t,'_panel_letter',False):t.remove()
    for a in list(ax.artists):a.remove()
    ax.set_title('');ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    for x,label in [(.24,'1'),(.42,'2')]:
        ax.add_patch(FancyBboxPatch((x,.59),.11,.30,boxstyle='round,pad=.008,rounding_size=.018',fc='#edf1f4',ec='#9aa3ac',lw=.75))
        ax.add_patch(Rectangle((x+.008,.61),.094,.16,fc='#a8cce7',ec='none'))
        ax.add_patch(Polygon([[x+.04,.59],[x+.085,.59],[x+.11,.42],[x+.085,.39]],closed=True,fc='#b8c4cc',ec='#84929e',lw=.5))
        ax.text(x+.055,.82,label,ha='center',va='center',fontsize=8)
    ax.add_patch(Ellipse((.30,.32),.30,.19,angle=-12,fc='#76706d',ec='#4d4947',lw=.65))
    ax.add_patch(Ellipse((.43,.36),.14,.13,angle=10,fc='#76706d',ec='#4d4947',lw=.65))
    ax.add_patch(Ellipse((.41,.43),.07,.07,fc='#958b87',ec='#5a524e',lw=.55))
    ax.plot(.47,.38,'o',ms=1.5,c='k');ax.add_patch(Arc((.12,.28),.22,.18,theta1=70,theta2=290,color='#9e7778',lw=.9))
    ax.add_patch(Rectangle((.67,.32),.25,.055,fc='#a6adb4',ec='#606870',lw=.7))
    ax.add_patch(Rectangle((.70,.375),.20,.025,fc='#d9e0e5',ec='#7d8790',lw=.6))
    ax.add_patch(FancyBboxPatch((.765,.405),.065,.24,boxstyle='round,pad=.005,rounding_size=.01',fc='#bed9eb',ec='#6d879a',lw=.7))
    ax.plot([.795,.795],[.65,.72],c='#6d879a',lw=1)
    ax.text(.80,.84,'3 h',ha='center',va='center',fontsize=8)
    ax.annotate('',xy=(.76,.71),xytext=(.53,.71),arrowprops=dict(arrowstyle='->',lw=1.1,color='#4a4a4a'))
    ax.text(.31,-.01,'Habituation\n5 d',ha='center',va='center',fontsize=8)
    ax.text(.80,-.01,'Test\n24 h',ha='center',va='center',fontsize=8)

def figure1(fig):
    schematic(axis_by_letter(fig,'A'))
    for ax in fig.axes:
        if ax.name=='polar':
            ax.set_xticks(np.arange(4)*np.pi/2,['ZT0','6','12','18'])
            for collection in ax.collections:
                offsets=collection.get_offsets()
                if len(offsets)>100:collection.set_alpha(.18 if len(offsets)>500 else .28)
            continue
        title=ax.get_title()
        if title.startswith(('F,','M,')):
            age=title.split(', ',1)[1].split('\n')[0]
            age=age.replace('Early-mid','Early middle').replace('Mid','Middle')
            n=re.search(r'n=(\d+)',title)
            ax.set_title(age,fontsize=9,pad=5)
            if n:ax.text(.02,.96,'n = '+n.group(1),transform=ax.transAxes,ha='left',va='top',fontsize=8,color='.25')
            ax.set_ylim(0,1.12);ax.set_yticks([0,1])
            if letter(ax) in list('EFGH'):ax.set_ylabel('Preference')
        elif title=='Group cosinor p':
            ax.set_title('');ax.set_ylabel('Group cosinor p')
            ax.get_legend().set_bbox_to_anchor((.95,1.22));ax.get_legend()._loc=1
        elif title=='Phase emergence':ax.set_title('Emergence index',fontsize=9)
        if letter(ax)=='B':
            ax.set_ylim(0,1.25);ax.set_yticks([0,.5,1])
            for t in ax.texts:
                if getattr(t,'_panel_letter',False):continue
                if t.get_text().isdigit():t.set_y(1.20)
                elif t.get_text() in ['*','**','***','n.s.']:t.set_y(1.04)
    # Symbols are headers for the two complete sex columns.
    for x,value,color in [(.47,'♀',PINK),(.83,'♂','#111111')]:bold_symbol(fig.text(x,.972,value,ha='center',va='center',fontsize=18,color=color))

def figure2(fig):
    for t in fig.texts:
        if t.get_text() in ['Female','Male']:
            female=t.get_text()=='Female';t.set_text('♀' if female else '♂');t.set_rotation(0);t.set_fontsize(16);t.set_color(PINK if female else '#111111');t.set_position((.026,.967 if female else .785));bold_symbol(t)
    for value,sex in [('G','♀'),('H','♂')]:
        ax=axis_by_letter(fig,value);ax.set_title(sex,fontsize=17,color=PINK if sex=='♀' else '#111111',pad=9);bold_symbol(ax.title)
    for legend in fig.legends:
        for t in legend.get_texts():
            if t.get_text() in ['Female','Male']:
                female=t.get_text()=='Female';t.set_text('♀' if female else '♂');t.set_color(PINK if female else '#111111');t.set_fontsize(12);bold_symbol(t)
    for value in ['I','J']:
        ax=axis_by_letter(fig,value);ax.set_xticks(np.arange(4)*np.pi/2,['ZT0','6','12','18'])
    condition_colors={'Control':'#f09200','PA':'#ff3030','RF-morning':'#22c51a','RF-afternoon':'#13762a'}
    for ax in fig.axes:
        if any(t.get_text()=='Entrainment condition' for t in ax.texts):
            for t in ax.texts:
                for condition,color in condition_colors.items():
                    if t.get_text().startswith(condition+' '):t.set_color(color)

def figure3(fig):
    # More room for all eight model rows, while keeping the figure below A4 height.
    positions={'A':[.10,.855,.17,.095],'B':[.335,.855,.18,.095],'E':[.61,.855,.34,.095],
               'C':[.13,.69,.34,.095],'F':[.61,.69,.34,.095],
               'D':[.13,.525,.34,.095],'G':[.61,.525,.34,.095],
               'H':[.10,.345,.37,.105],'I':[.59,.345,.37,.105],
               'J':[.16,.075,.76,.15]}
    for value,box in positions.items():axis_by_letter(fig,value).set_position(box)
    a=axis_by_letter(fig,'A');a.set_title('');a.legend(frameon=False,loc='upper left',bbox_to_anchor=(.0,1.35),ncol=2,fontsize=8,handlelength=.9,columnspacing=.7,handletextpad=.3)
    for value,title in [('C','Random'),('D','Bias'),('E','Simple RL'),('F','Circadian reward RL'),('G','Full RL')]:
        ax=axis_by_letter(fig,value);ax.set_title(title,fontsize=9,pad=5);ax.set_xticks([0,6,12,18,24]);ax.set_yticks([0,1],['Water','Sweet']);ax.set_xlabel('ZT/CT (h)')
        if value in ['C','E','F']:ax.set_xlabel('')
        t=next(t for t in ax.texts if getattr(t,'_panel_letter',False));t.set_transform(fig.transFigure);t.set_position((.025 if value in ['C','D'] else .55,positions[value][1]+positions[value][3]+.005));t.set_va('bottom')
        if value in ['E','F','G']:ax.yaxis.tick_left()
        for line in ax.lines:
            if len(line.get_xdata())>100:line.set_linewidth(2.1)
    for value in ['H','I']:
        ax=axis_by_letter(fig,value);ax.set_title('')
        for label in ax.get_xticklabels():label.set_rotation(35);label.set_ha('right')
        ax.set_ylabel('Mean Δ score' if value=='H' else 'Best-fit count')
    for value,title,barlabel in [('J','Model score differences','Δ score')]:
        ax=axis_by_letter(fig,value);ax.set_title(title,fontsize=9,pad=6)
        image=ax.images[0];bar=image.colorbar
        if value=='J':
            image.set_norm(PowerNorm(gamma=.5,vmin=0,vmax=100));bar.update_normal(image);bar.set_ticks([0,5,25,100])
        else:bar.set_ticks([1,4,8])
        bar.ax.set_position([.934,positions[value][1],.012,positions[value][3]])
        bar.set_label('');bar.ax.set_title(barlabel,fontsize=8,pad=5)
        bar.ax.tick_params(labelsize=8,width=1.1,length=2)
        ax.set_xlabel('Animal ordered by best model')
        ax.tick_params(axis='y',length=2,pad=3)
        if value=='J':ax.tick_params(labelbottom=True)

def style(fig,name):
    before=scientific_arrays(fig,name)
    for ax in fig.axes:
        for spine in ax.spines.values():spine.set_linewidth(1.25)
        ax.tick_params(width=1.25,length=3.5,pad=2)
        for line in ax.lines:
            n=len(line.get_xdata());alpha=line.get_alpha()
            if n>=6:
                if alpha is not None and alpha<.4:
                    line.set_linewidth(.65);line.set_alpha(min(.36,max(alpha,.23)))
                else:line.set_linewidth(max(1.45,line.get_linewidth()))
        for collection in ax.collections:
            if hasattr(collection,'get_segments') and collection.get_linewidths().size:collection.set_linewidth(max(1.0,float(collection.get_linewidths()[0])))
        for t in ax.texts:
            if getattr(t,'_panel_letter',False):t.set_fontsize(12);t.set_fontweight('bold')
        title=ax.get_title()
        if title and title[0:2] in ['F,','M,']:ax.set_title(symbol(title),fontsize=9)
        if ax.get_legend():
            for t in ax.get_legend().get_texts():
                if t.get_text() in ['F','M']:t.set_text(symbol(t.get_text()));t.set_fontsize(10);recolor_text(t);bold_symbol(t)
        tick_text=[t.get_text() for t in ax.get_xticklabels()]
        if any(re.match(r'^[FM] ',s) for s in tick_text):
            ax.set_xticks(ax.get_xticks(),[symbol(s) for s in tick_text])
    if name=='Figure1':
        # Restore original strings for the compact age-only title extraction.
        for ax in fig.axes:
            title=ax.get_title()
            if title.startswith(('♀,','♂,')):ax.set_title(('F' if title[0]=='♀' else 'M')+title[1:])
        figure1(fig)
    elif name=='Figure2':figure2(fig)
    elif name=='Figure3':figure3(fig)
    for artist,kind,original in before:
        current=np.asarray(artist.get_xydata() if kind=='line' else artist.get_offsets() if kind=='offsets' else artist.get_array())
        assert np.array_equal(current,original,equal_nan=True),(name,kind,'numeric artist changed')
    # The original phase-marker geometry remains exactly on the unit circle.
    if name=='Figure1':
        for ax in fig.axes:
            if ax.name=='polar':assert ax.get_ylim()[1]==1
    QA.append({'figure':name,'numeric_artist_arrays_preserved':len(before),'sex_symbols':sum(t.get_text().count('♀')+t.get_text().count('♂') for t in fig.findobj(match=Text)),'figure_height_mm':fig.get_figheight()*25.4,'figure_width_mm':fig.get_figwidth()*25.4})
    fig.canvas.draw()

def install(f,renderer):
    f.BLUE=BLUE;f.ORANGE=ORANGE;f.PINK=PINK;f.GRAY=GRAY
    f.AGE_COLOR={'Young':'#ec8d2e','EarlyMid':'#9290bc','Mid':'#d95ca5','Old':'#78a23d'}
    f.MODEL_COLOR.update(MODEL_COLORS)
    renderer['replace_function'](f,'figure2',[
        ("'Control':'#dba03f','PA':'#c96565'", "'Control':'#f09200','PA':'#ff3030'"),
        ("'RF-morning':'#77aa50','RF-afternoon':'#397c4e'", "'RF-morning':'#22c51a','RF-afternoon':'#13762a'")])
    renderer['HEIGHT_MM'].update({'Figure1':150,'Figure2':195,'Figure3':180,'Figure4':180})
    original_prepare=renderer['prepare']
    def prepare(fig,name):
        original_prepare(fig,name);style(fig,name)
        dest=renderer['DEST']/'audit';dest.mkdir(exist_ok=True,parents=True)
        (dest/'reference_style_data_preservation.json').write_text(json.dumps(QA,indent=2))
    renderer['prepare']=prepare
