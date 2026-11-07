#!/usr/bin/env python3
"""Publish local measured evidence as Markdown, figures and a self-contained HTML report."""
from __future__ import annotations
import base64
import csv
import hashlib
import html
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'
LABELS={'every_sort':'Corrected SORT','every_two_stage':'Two-stage', 'fixed2':'Fixed 1/2','fixed4':'Fixed 1/4','fixed8':'Fixed 1/8',
        'covariance50':'Covariance, cap50%','risk50':'Risk, cap50%','adaptive25':'Adaptive, cap25%','adaptive50':'Adaptive, cap50%',
        'matched_adaptive25':'Uniform matched25','matched_adaptive50':'Uniform matched50','ocsort':'Official OC-SORT','bytetrack':'ByteTrack + adapters',
        'active_outputs':'Two-stage active output','all_estimates':'Two-stage all estimates','active_sort':'SORT active output'}
SCOPES=[('synthetic_test','Synthetic diagnostics: 6 clips, 1,080 frames','Generated frames and noisy simulated detections. Different seeds within the same six scenario families.'),
        ('gmot40_test','Natural insect footage: 2 clips, 348 frames','Author-provided one-shot detector outputs on untouched GMOT-40 ant/bee frames. This is cached association/scheduling evaluation, not accuracy from our ONNX model.'),
        ('native_bee_oracle_test','Processed bee recordings: oracle association diagnostic','Ground-truth boxes are used as perfect detections. ROI masks/counters and sampling gaps prevent natural scene/new-arrival deployment claims.'),
        ('gmot40_onnx_test','Original pest ONNX model: transfer probe','Actual OpenCV DNN detections on natural held-out ant/bee clips. Unknown original training provenance; this is a transfer probe, not a newly trained detector.')]

def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
def rows_for(name):
    report=load(ROOT/'runs'/name/'aggregate.json');rows=dict(report['aggregate'])
    ref={'synthetic_test':'reference_synthetic_test_provided','gmot40_test':'reference_gmot40_test_provided',
         'native_bee_oracle_test':'reference_native_bee_test_provided'}.get(name)
    if ref and (ROOT/'runs'/ref/'aggregate.json').exists():rows.update(load(ROOT/'runs'/ref/'aggregate.json')['aggregate'])
    return rows

def table(rows,markdown=False):
    cols=['Method','Detector calls','HOTA','IDF1','ID switches','FP','FN']
    ordered=[k for k in LABELS if k in rows]
    values=[[LABELS.get(k,k),f"{rows[k]['detector_rate']*100:.1f}%",f"{rows[k]['HOTA']*100:.2f}",f"{rows[k]['IDF1']*100:.2f}",
             str(rows[k]['IDSW']),str(rows[k]['FP']),str(rows[k]['FN'])] for k in ordered]
    if markdown:return '| '+' | '.join(cols)+' |\n|'+ '|'.join(['---']*len(cols))+'|\n'+'\n'.join('| '+' | '.join(row)+' |' for row in values)
    return '<div class="scroll"><table><thead><tr>'+''.join(f'<th>{html.escape(c)}</th>' for c in cols)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{html.escape(c)}</td>' for c in row)+'</tr>' for row in values)+'</tbody></table></div>'

def figures():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(14,5.5),layout='constrained')
    for ax,(name,title) in zip(axes,[('synthetic_test','Synthetic test: simulated detector'),('gmot40_test','Natural insect clips: author detector cache')]):
        rows=rows_for(name)
        fixed=['fixed8','fixed4','fixed2','every_two_stage']
        ax.plot([rows[k]['detector_rate']*100 for k in fixed],[rows[k]['HOTA']*100 for k in fixed],color='#8995a6',marker='o',label='Fixed intervals / every frame')
        for key,color,marker,label in [('adaptive25','#007d8a','*','Adaptive cap25%'),('adaptive50','#165ac4','*','Adaptive cap50%'),
                                      ('matched_adaptive25','#007d8a','x','Uniform matched25'),('matched_adaptive50','#165ac4','x','Uniform matched50'),
                                      ('every_sort','#b36c00','s','Corrected SORT'),('ocsort','#7d3a9d','D','Official OC-SORT'),('bytetrack','#bd4366','D','ByteTrack + adapters')]:
            if key in rows:ax.scatter(rows[key]['detector_rate']*100,rows[key]['HOTA']*100,c=color,marker=marker,s=115 if marker=='*' else 48,label=label,zorder=4)
        ax.set(title=title,xlabel='Actual detector calls / source frames (%)',ylabel='Mean HOTA (%)',xlim=(0,105));ax.grid(alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False,fontsize=9)
    fig.savefig(OUT/'quality_compute.png',dpi=180);plt.close(fig)

    rows=rows_for('gmot40_test');keys=['every_sort','every_two_stage','ocsort','bytetrack']
    fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained');x=np.arange(len(keys))
    ax.bar(x-.18,[100*rows[k]['HOTA'] for k in keys],.36,label='HOTA',color='#136d83')
    ax.bar(x+.18,[100*rows[k]['IDF1'] for k in keys],.36,label='IDF1',color='#688cce')
    ax.set_xticks(x,[LABELS[k] for k in keys]);ax.set(ylabel='Unweighted sequence mean (%)',title='Natural ant/bee clips: identical author-provided detections');ax.legend(frameon=False);ax.grid(axis='y',alpha=.2)
    fig.savefig(OUT/'tracking_comparison.png',dpi=180);plt.close(fig)

def main():
    OUT.mkdir(exist_ok=True);tables=OUT/'tables';tables.mkdir(exist_ok=True)
    evidence={'source_date':'2026-11-07','branch':git('branch','--show-current'),'base_commit':git('rev-parse','main'),
              'study_sha256':digest(ROOT/'configs'/'study.json'),'split_sha256':digest(ROOT/'configs'/'splits.json'),
              'selection':load(ROOT/'configs'/'scheduler_selection.json'),'scopes':{}}
    sections=[];markdown=['# Kinetic-Tracker: measured implementation report','',
        'Native C++17 tracking, detector-budget scheduling, and reproducible MOT evaluation. Results are exploratory evidence on a small subset, not a state-of-the-art or field-deployment claim.','',
        '## Main findings','',
        '- The confidence-based two-stage association improves over the corrected SORT-style baseline on the tested natural ant/bee clips using the exact same author-provided detections.',
        '- Adaptive scheduling modestly improves over uniform selection at exactly matched detector-call counts on the synthetic test suite. On natural clips it collapses to periodic detection and does not beat the rate-matched baseline.',
        '- Original pest-model transfer is weak. Stronger association does not substitute for a well-trained detector.',
        '- Output policy matters: stale predictions can increase false positives. Both policies and their timing of introduction are disclosed.','',
        '## Protocol','',
        'Primary frozen scheduler configurations were selected on six synthetic VALIDATION clips only, then evaluated on separate synthetic seeds and real sequence splits. The original study deliberately retains all non-tentative estimates (`--output-policy all`). Scores below are unweighted means over sequences, not pooled official benchmark scores. HOTA/IDF1 are shown as percentages; error counts are summed. Exact matched baselines use each adaptive method\'s realized per-sequence call count, with content-independent evenly spaced positions.','',
        'Corrected SORT and two-stage are local implementations. ByteTrack and OC-SORT are pinned official tracking-only implementations; ByteTrack uses documented compatibility adapters. Upstream output rules, threshold details and fixed-step timing differences are retained and recorded.','']
    for name,title,note in SCOPES:
        if not (ROOT/'runs'/name/'aggregate.json').exists():continue
        rows=rows_for(name);evidence['scopes'][name]={'description':note,'rows':rows}
        markdown+=['## '+title,'',note,'',table(rows,True),'']
        sections.append(f'<section class="scope" data-name="{name}"><h2>{html.escape(title)}</h2><p>{html.escape(note)}</p>{table(rows)}</section>')
        shutil.copy2(ROOT/'runs'/name/'aggregate.json',tables/(name+'.json'))
        shutil.copy2(ROOT/'runs'/name/'aggregate.csv',tables/(name+'.csv'))
        with (tables/(name+'_combined.csv')).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=['method','detector_rate','HOTA','IDF1','AssA','IDSW','FP','FN']);w.writeheader()
            for k,r in rows.items():w.writerow({'method':k,**{p:r[p] for p in w.fieldnames if p!='method'}})
    diagnostics=[]
    for name,title in [('output_policy_gmot_test','Post-hoc output policy: natural clips'),('output_policy_native_test','Post-hoc output policy: oracle bee boxes')]:
        if (ROOT/'runs'/name/'aggregate.json').exists():
            r=load(ROOT/'runs'/name/'aggregate.json');evidence[name]=r['aggregate'];diagnostics.append(f'<h3>{title}</h3>{table(r["aggregate"])}')
            markdown+=['## '+title,'', '**Exploratory post-hoc diagnostic:** investigated after inspecting first-pass test false positives. It was then checked on validation, and the CLI default changed to `active`. These numbers are not an untouched confirmatory test. Primary frozen-study scores above retain `all`. Telemetry preserves all estimates in both modes; active suppresses detector-confirmed lost boxes from scored/rendered output but retains predictions during deliberate skips.','',table(r['aggregate'],True),'']
    runtime=None
    if (ROOT/'runs'/'runtime'/'aggregate.json').exists():
        runtime=load(ROOT/'runs'/'runtime'/'aggregate.json');evidence['runtime']=runtime
        shutil.copy2(ROOT/'runs'/'runtime'/'aggregate.json',tables/'runtime.json')
        markdown+=['## Actual uncached CPU timing','',f"Three counterbalanced repetitions on the original unlabelled sample, {runtime['threads']} OpenCV CPU threads. Baseline mean wall time: {runtime['aggregate']['every']['mean_wall_seconds']:.3f}s; adaptive mean wall time: {runtime['aggregate']['adaptive50']['mean_wall_seconds']:.3f}s. Wall-time speedup: {runtime['mean_wall_speedup']:.2f}x. This input has no identity labels; the timing result cannot establish accuracy. Initialisation and one warmup inference are separate. No GPU or energy claim.",'']
    markdown+=['## Limits and next experiments','',
        '- Natural scheduling evidence is only two GMOT-40 insect clips; the detector is the author\'s one-shot baseline and uses its target exemplar protocol. This is a custom subset, not an official full benchmark submission.',
        '- Real capture timing is not verified. Report discovery delays in frames; nominal millisecond conversions are not physical measurements.',
        '- Monash release images have annotation-related ROI masks and counters. Oracle detections isolate association and are never represented as detector performance.',
        '- Original `best.onnx` training data/code are absent; its evaluation is a transfer probe with unknown training provenance.',
        '- Adaptive foreground novelty is sensitive to vegetation, camera motion, missed tracks and small targets. Current evidence supports a prototype and a negative real-data scheduling result.',
        '- The active-output change is a post-hoc diagnostic. A new independent raw recording is needed to confirm that decision without development-set reuse.',
        '- Pinned TrackEval CLEAR Frag may undercount wholly empty prediction gaps; HOTA/IDF1 and FP/FN are the primary metrics.',
        '- Next: acquire independent raw labelled pest videos; establish detector recall with documented training provenance; compare a learned/size-aware motion gate against OC-SORT; test scene-motion rejection at matched budgets.','',
        '## Reproduction and attribution','',
        'See `docs/REPRODUCE.md`, `docs/METHOD.md`, `docs/ARCHITECTURE.md`, `docs/DATA.md`, `docs/RELATED_WORK.md` and pinned `third_party/PROVENANCE.json`. Raw inputs/caches are ignored by Git. Configurations, summary tables, report and licensed demo attribution are tracked.','',
        'GMOT-40: Bai et al., CVPR2021, https://github.com/Spritea/GMOT40 (dataset CC BY-NC-SA4.0). Monash native bees: DOI10.26180/32315067.v1 (CC BY4.0). Metrics: https://github.com/JonathonLuiten/TrackEval. References: https://github.com/noahcao/OC_SORT and https://github.com/FoundationVision/ByteTrack. Scheduling prior work: DriftSched, DOI10.1109/MECO70748.2026.11579050.','']
    (OUT/'REPORT.md').write_text('\n'.join(markdown),encoding='utf-8')
    (OUT/'evidence.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    figures()
    images=''.join('<img class="plot" alt="'+name+'" src="data:image/png;base64,'+base64.b64encode((OUT/name).read_bytes()).decode()+'">' for name in ['quality_compute.png','tracking_comparison.png'])
    raw=evidence['scopes']['gmot40_test']['rows'];syn=evidence['scopes']['synthetic_test']['rows']
    middle=(f'<b>{runtime["mean_wall_speedup"]:.2f}×</b><span>Uncached CPU wall-time speedup<br>3 runs on the unlabelled original clip</span>' if runtime else f'<b>{100*syn["adaptive50"]["detector_rate"]:.1f}%</b><span>Detector calls on synthetic test<br>adaptive cap 50%</span>')
    intro=f'<div class="cards"><article><b>{100*raw["every_two_stage"]["HOTA"]:.2f}</b><span>Mean HOTA, natural cached detections<br>corrected SORT {100*raw["every_sort"]["HOTA"]:.2f}</span></article><article>{middle}</article><article><b>No real-data scheduling gain</b><span>Adaptive50 equals fixed2 on these natural clips</span></article></div>'
    runtime_html=''
    if runtime:
        runtime_html='<section><h2>Actual uncached CPU timing</h2><p>'+html.escape(markdown[markdown.index('## Actual uncached CPU timing')+2])+'</p><pre>'+html.escape(json.dumps(runtime['aggregate'],indent=2))+'</pre></section>'
    video_html=''
    for name,title in [('gmot_comparison.mp4','Natural insect tracking: same author detector cache'),('synthetic_births.mp4','Synthetic new arrivals: exactly matched call counts'),('background_failure.mp4','Synthetic background motion: adverse case')]:
        if (OUT/name).exists():video_html+=f'<article><h3>{title}</h3><video controls preload="metadata" src="{name}"></video><p><a href="{name}">Open video</a> · cached replay, not live inference speed</p></article>'
    page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Kinetic-Tracker | Measured results</title><style>
:root{color-scheme:light;font-family:system-ui,Segoe UI,sans-serif;color:#14283d;background:#f1f5f8}body{margin:0}main{max-width:1180px;margin:auto;padding:42px 22px}header{background:#102b42;color:white;padding:32px;border-radius:18px}h1{font-size:clamp(28px,4vw,44px);margin:8px 0}h2{margin-top:30px}p,li{line-height:1.65}.tag{color:#a9e4e4;font-weight:650;letter-spacing:.08em}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:20px 0}.cards article,section{background:white;border:1px solid #dbe5ed;border-radius:14px;padding:22px;margin:18px 0}.cards b{font-size:27px;display:block;color:#076d78}.cards span{font-size:14px;display:block;margin-top:10px}.notice{border-left:5px solid #bf7c18;background:#fff6df;padding:18px}.plot{display:block;width:100%;background:white;margin:22px 0;border-radius:12px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px}th,td{text-align:right;padding:11px 10px;border-bottom:1px solid #e5ebef;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:#edf5f7}select{font:inherit;padding:10px;border:1px solid #b4c7d4;border-radius:8px}video{max-width:100%;border-radius:10px}a{color:#075b92}pre{overflow:auto;font-size:12px}.small{font-size:13px;color:#496075}footer{margin:25px 0;font-size:13px}@media(max-width:750px){.cards{grid-template-columns:1fr}.cards article{margin:0}main{padding:18px 12px}header{padding:22px}}
*{box-sizing:border-box}main{width:100%}header{overflow-wrap:anywhere}.cards>*{min-width:0}select{max-width:100%}pre{max-width:100%}
</style></head><body><main><header><div class="tag">NATIVE C++17 · REPRODUCIBLE EVALUATION</div><h1>Kinetic-Tracker</h1><p>Tracking quality under a limited detector budget. Working code, measured tradeoffs and visible failure cases.</p><p>Local research branch: <code>research/compute-aware-tracking</code> · main preserved · 7 November 2026</p></header>'''
    page+=intro+'<p class="notice"><strong>Scope:</strong> exploratory small-sample results. Natural-video tables use author-provided cached detections unless labelled ONNX. Synthetic, oracle and actual inference measurements are kept separate. No new state-of-the-art or field-deployment claim.</p>'
    page+='<section><h2>What was built</h2><p>A modular C++ detector, timestamp-aware Kalman tracker, class-aware gated assignment, track lifecycle, confidence-based two-stage association, token-budget scheduler, causal raw-frame novelty trigger, official reference adapters and pinned TrackEval metrics. Input/output collision safeguards and automated semantic tests are included.</p><p>Primary study: thresholds selected on synthetic validation; exact per-sequence detector-rate matching; all-estimate output policy retained. Subsequent active-output diagnostics are labelled post-hoc.</p></section>'+images
    page+='<label for="scope">Show dataset: </label><select id="scope"><option value="all">All sources</option>'+''.join(f'<option value="{n}">{html.escape(t)}</option>' for n,t,_ in SCOPES)+'</select>'+''.join(sections)
    page+='<section><h2>Output-policy diagnostic</h2><p>This analysis followed initial held-out error inspection, and is exploratory. The CLI now defaults to active output after validation checks; primary study tables above retain all estimates. No failure frames were removed.</p>'+''.join(diagnostics)+'</section>'+runtime_html
    page+='<section><h2>Comparison videos</h2>'+video_html+'</section>'
    page+='<section><h2>Limits and next work</h2><ul>'+''.join('<li>'+html.escape(x[2:])+'</li>' for x in markdown[markdown.index('## Limits and next experiments')+2:markdown.index('## Reproduction and attribution')] if x.startswith('- '))+'</ul><p>Inspect <a href="evidence.json">machine-readable evidence</a>, <a href="REPORT.md">full Markdown report</a> and the <a href="tables/gmot40_test_combined.csv">natural-video result CSV</a>.</p></section>'
    page+='<footer>Data: GMOT-40, Bai et al., CVPR2021 (CC BY-NC-SA4.0); Monash native bees, DOI10.26180/32315067.v1 (CC BY4.0). Derived videos include attribution files. Source code retains its own licence. This report is local and has no external scripts or network requests.</footer><script>document.getElementById("scope").addEventListener("change",function(){document.querySelectorAll(".scope").forEach(s=>s.hidden=this.value!=="all"&&s.dataset.name!==this.value)});</script></main></body></html>'
    (OUT/'report.html').write_text(page,encoding='utf-8')
    print(OUT/'report.html')

if __name__=='__main__':main()
