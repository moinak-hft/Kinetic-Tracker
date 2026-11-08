#!/usr/bin/env python3
"""Validate predeclared schedules using the already validation-selected association."""
import argparse
import json
from pathlib import Path
import numpy as np
from evaluate import guard_output_paths
from refinement_study import ROOT, case_key, run_one, summarize, write_json, select_validation

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,default=ROOT/'runs/refinement')
    p.add_argument('--protocol',type=Path,default=ROOT/'configs/refinement_protocol.json')
    a=p.parse_args()
    if (a.study/'freeze.json').exists():raise ValueError('No validation selection after confirmation freeze')
    protocol=json.loads(a.protocol.read_text());selection=json.loads((a.study/'validation/selection.json').read_text())
    quality_profile=selection['selected_profile'];fixed_records=[]
    candidates=[name for name,preset in protocol['profiles'].items() if preset.get('selection_eligible') or name=='baseline_v2']
    for group in protocol['validation_groups'].values():
        for case in group['cases']:
            for candidate in candidates:
                fixed_records.append(run_one(protocol,case,candidate,'fixed2',protocol['schedules']['fixed2'],'active',
                    a.study/'scheduling_validation'/case_key(case)/candidate/'fixed2'/'active','validation'))
    budget_selection=select_validation(protocol,fixed_records,schedule_name='fixed2')
    profile=budget_selection['selected_profile'];records=[]
    for group in protocol['validation_groups'].values():
        for case in group['cases']:
            for name,schedule in protocol['schedules'].items():
                directory=a.study/'scheduling_validation'/case_key(case)/profile/name/'active'
                row=run_one(protocol,case,profile,name,schedule,'active',directory,'validation')
                records.append(row)
                if schedule['schedule']=='adaptive':
                    matched={**schedule,'schedule':'uniform','uniform_calls':row['summary']['detector_calls']}
                    control=run_one(protocol,case,profile,'matched__'+name,matched,'active',
                                    a.study/'scheduling_validation'/case_key(case)/profile/('matched__'+name)/'active','validation')
                    if row['summary']['detector_calls']!=control['summary']['detector_calls']:raise ValueError('Rate mismatch')
                    records.append(control)
    outcomes={}
    for schedule in protocol['schedules']:
        per_group={}
        for group_name,group in protocol['validation_groups'].items():
            ids={case_key(c) for c in group['cases']}
            group_rows=[r for r in records if r['schedule_name']==schedule and case_key(r['case']) in ids]
            per_group[group_name]={k:float(np.mean([r['metrics'][k] for r in group_rows])) for k in ['HOTA','IDF1']}
        chosen=[r for r in records if r['schedule_name']==schedule]
        outcomes[schedule]={'groups':per_group,
                            'balanced_HOTA':sum(group['weight']*per_group[name]['HOTA'] for name,group in protocol['validation_groups'].items()),
                            'balanced_IDF1':sum(group['weight']*per_group[name]['IDF1'] for name,group in protocol['validation_groups'].items()),
                            'detector_rate':sum(r['summary']['detector_calls'] for r in chosen)/sum(r['summary']['frames'] for r in chosen)}
    minimum=outcomes['fixed2']['groups']['raw_insect']['HOTA']-.005
    allowed=[k for k,v in outcomes.items() if k!='every' and v['detector_rate']<=.51 and v['groups']['raw_insect']['HOTA']>=minimum-1e-12]
    winner=max(allowed,key=lambda k:(outcomes[k]['balanced_HOTA'],outcomes[k]['balanced_IDF1'],-outcomes[k]['detector_rate']))
    base={**protocol['shared_tracker'],**protocol['profiles'][quality_profile]['args'],'output_policy':'active'}
    budget_base={**protocol['shared_tracker'],**protocol['profiles'][profile]['args'],'output_policy':'active'}
    recommendation={'schema_version':1,'selected_profile':quality_profile,'selected_association':base['association'],
                    'budget_profile':profile,'budget_association':budget_base['association'],
                    'selected_schedule':'every','budget_schedule':winner,
                    'selection_basis':'Validation only. Quality and budget association selected separately under every/fixed2 using identical source-balanced HOTA and raw-insect non-regression guard. Budget schedule: maximize same balanced HOTA among declared <=51% realized-call options, raw HOTA within0.5points of fixed2; tie IDF1 then fewer calls. No confirmation results used.',
                    'args':{**base,'schedule':'every'},'budget_args':{**budget_base,**protocol['schedules'][winner]},
                    'budget_association_validation':budget_selection,
                    'schedule_validation':outcomes,'budget_raw_hota_minimum':minimum,
                    'limitations':['Detector weights are frozen. Cached detection quality is distinct from real inference throughput.',
                                   'Original test clips were already inspected; additional confirmation is scored only after this recommendation is frozen.']}
    unique={r['output']:r for r in fixed_records+records}
    report={'quality_profile':quality_profile,'budget_profile':profile,'profile':profile,
            'aggregate':summarize(list(unique.values())),'selection':recommendation,
            'experiments':[str(Path(r['output'])/'experiment.json') for r in unique.values()]}
    guard_output_paths([a.protocol,a.study/'validation/selection.json'],[ROOT/'configs/recommended_profile.json',a.study/'scheduling_validation/aggregate.json'])
    write_json(ROOT/'configs/recommended_profile.json',recommendation)
    write_json(a.study/'scheduling_validation/aggregate.json',report)
    print(json.dumps({'selected_profile':profile,'budget_schedule':winner,'outcomes':outcomes},indent=2))

if __name__=='__main__':main()
