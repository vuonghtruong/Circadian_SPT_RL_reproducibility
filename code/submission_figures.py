"""Preserve submitted S3 and rename the phase supplements after omitting audit S9."""
from pathlib import Path
import hashlib,json
from PIL import Image

MAP={'FigureS10_model_phases':'FigureS9_model_phases',
     'FigureS11_model_phase_examples':'FigureS10_model_phase_examples'}

def install(f,renderer):
    original_save=f.save
    for old,new in MAP.items():renderer['HEIGHT_MM'][new]=renderer['HEIGHT_MM'][old]
    def save(fig,name):
        return original_save(fig,MAP.get(name,name))
    f.save=save
    asset=Path(__file__).resolve().parent/'assets/Original_FigureS3.png'
    with Image.open(asset) as im:
        height=175*im.height/im.width
    renderer['HEIGHT_MM']['FigureS3']=height
    def submitted_s3():
        with Image.open(asset) as source:art=source.convert('RGB').copy()
        fig=f.plt.figure(figsize=(175/25.4,height/25.4))
        ax=fig.add_axes([0,0,1,1]);ax.imshow(art,aspect='auto',interpolation='none');ax.axis('off')
        renderer['INPUTS'].add(str(asset))
        save(fig,'FigureS3')
        dest=renderer['DEST']/'audit';dest.mkdir(exist_ok=True,parents=True)
        (dest/'original_s3_preservation.json').write_text(json.dumps({'source':'User-provided original S3 reference codex-clipboard-fae07c40-e7a8-4781-9877-2e0a61d5cdd9.png; matching submitted word/media/image7.emf composition','rendered_asset_sha256':hashlib.sha256(asset.read_bytes()).hexdigest(),'width_mm':175,'height_mm':height,'artwork_content_changed':False,'manual_visual_review_required':True},indent=2))
    f.supplement3=submitted_s3
