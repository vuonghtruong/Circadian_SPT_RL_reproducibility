"""Keep one vector figure per panel set and journal TIFFs; omit preview duplicates."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    folder=ROOT/'figures'
    paths=sorted(folder.glob('Figure*.pdf'),key=lambda p:('S' in p.stem,int(''.join(x for x in p.stem.split('_')[0] if x.isdigit()))))
    assert len(paths)==14 and not any('FigureS12' in p.name for p in paths)
    lines=['# Publication figures','All figures are 175 mm wide. Generated plots retain vector graphics; S3 preserves the original submitted raster artwork. The four main TIFFs are 600 dpi RGB with LZW compression.','The former S9 light-dark audit and S12 are omitted. Original S3 is restored; phase figures are renumbered S9 and S10.']
    for p in paths:
        label=p.stem.split('_')[0].replace('FigureS','Supplement S').replace('Figure','Figure ')
        line=f'- [{label}]({p.name})'
        if p.stem in ['Figure1','Figure2','Figure3','Figure4']:line+=f' · [TIFF]({p.stem.replace("Figure","Fig")}.tif)'
        lines.append(line)
    (folder/'README.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    # Exact known generated preview extensions, confined to this output folder.
    for pattern in ['Figure*.png','Figure*.svg']:
        for path in folder.glob(pattern):path.unlink()
    print(f'Kept {len(paths)} PDFs and {len(list(folder.glob("Fig*.tif")))} TIFFs')

if __name__=='__main__':main()
