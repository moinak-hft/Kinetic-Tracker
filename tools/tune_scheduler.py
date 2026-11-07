#!/usr/bin/env python3
"""Select scheduler thresholds using synthetic VALIDATION only, then freeze them."""
import argparse
import json
from pathlib import Path
from benchmark import ROOT,aggregate,run_one

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'runs'/'tuning');a=p.parse_args()
    split=json.loads((ROOT/'configs'/'splits.json').read_text())['synthetic']['validation']
    candidates=[]
    for budget in [.25,.5]:
        for risk in [.12,.18,.28]:
            for novelty in [.00008,.0003]:
                config={'name':f'adaptive_b{budget}_r{risk}_n{novelty}', 'schedule':'adaptive','association':'two_stage',
                        'target_rate':budget,'risk_threshold':risk,'novelty_threshold':novelty,'max_gap':8}
                records=[]
                for s in split:records.append(run_one(ROOT/s,config,a.output/Path(s).name/config['name']))
                result=aggregate(records)[config['name']]
                candidates.append({'config':config,'validation':result})
                print(config['name'],'HOTA',round(result['HOTA'],4),'IDF1',round(result['IDF1'],4),'rate',round(result['detector_rate'],4),flush=True)
    selected=[]
    for budget in [.25,.5]:
        best=max((r for r in candidates if r['config']['target_rate']==budget),key=lambda r:(r['validation']['HOTA'],r['validation']['IDF1'],-r['validation']['detector_rate']))
        config=dict(best['config']);config['name']='adaptive25' if budget==.25 else 'adaptive50';selected.append(config)
    frozen={'selection':'Maximize unweighted mean HOTA over six synthetic validation sequences, separately at target rates0.25/0.5; tie-break IDF1 then fewer calls. No test results used.',
            'validation_sequences':split,'selected':selected,'candidates':candidates}
    (ROOT/'configs'/'scheduler_selection.json').write_text(json.dumps(frozen,indent=2)+'\n')
    configs=[{'name':'every_sort','schedule':'every','association':'sort'},
             {'name':'every_two_stage','schedule':'every','association':'two_stage'},
             {'name':'fixed2','schedule':'fixed','interval':2,'association':'two_stage'},
             {'name':'fixed4','schedule':'fixed','interval':4,'association':'two_stage'},
             {'name':'fixed8','schedule':'fixed','interval':8,'association':'two_stage'},
             {'name':'covariance50','schedule':'covariance','association':'two_stage','target_rate':.5,'risk_threshold':selected[1]['risk_threshold'],'max_gap':8},
             {'name':'risk50','schedule':'risk','association':'two_stage','target_rate':.5,'risk_threshold':selected[1]['risk_threshold'],'max_gap':8},*selected]
    (ROOT/'configs'/'study.json').write_text(json.dumps(configs,indent=2)+'\n')
    print(json.dumps({'selected':selected},indent=2))

if __name__=='__main__':main()
