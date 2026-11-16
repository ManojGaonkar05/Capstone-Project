import pandas as pd

def get(o,k,default=None):
    return o.get(k,default) if isinstance(o,dict) else getattr(o,k,default)
def duration_ms(start,end):
    try: return max(0,(end-start).total_seconds()*1000)
    except Exception: return None
def runs_to_frame(runs):
    rows=[]
    for r in runs:
        extra=get(r,'extra',{}) or {}; tags=get(r,'tags',[]) or []; usage=extra.get('metadata',{}).get('usage_metadata',{}) if isinstance(extra,dict) else {}
        total=get(r,'total_tokens',None); prompt=get(r,'prompt_tokens',None); completion=get(r,'completion_tokens',None)
        if total is None: total=usage.get('total_tokens')
        if prompt is None: prompt=usage.get('input_tokens',usage.get('prompt_tokens'))
        if completion is None: completion=usage.get('output_tokens',usage.get('completion_tokens'))
        start=get(r,'start_time'); end=get(r,'end_time'); error=get(r,'error')
        rows.append({'run_id':str(get(r,'id','')),'trace_id':str(get(r,'trace_id') or get(r,'id','')),'parent_run_id':str(get(r,'parent_run_id') or ''),'run_name':get(r,'name',''),'run_type':str(get(r,'run_type','')),'is_llm':str(get(r,'run_type','')).lower()=='llm','is_tool':str(get(r,'run_type','')).lower()=='tool','has_error':bool(error),'error_text':str(error)[:500] if error else '','start_time':str(start or ''),'end_time':str(end or ''),'duration_ms':duration_ms(start,end),'total_tokens':total,'prompt_tokens':prompt,'completion_tokens':completion,'estimated_cost':get(r,'total_cost',None),'tags':','.join(map(str,tags)),'input_chars':len(str(get(r,'inputs',{}) or {})),'output_chars':len(str(get(r,'outputs',{}) or {}))})
    return pd.DataFrame(rows)
def extract_features(raw):
    if raw.empty: return pd.DataFrame()
    out=[]
    for trace, g in raw.groupby('trace_id',dropna=False):
        roots=g[g.parent_run_id.fillna('')=='']; root=roots.iloc[0] if len(roots) else g.iloc[0]
        dur=pd.to_numeric(g.duration_ms,errors='coerce'); rootdur=pd.to_numeric(pd.Series([root.duration_ms]),errors='coerce').iloc[0]
        if pd.isna(rootdur): rootdur=dur.max() if dur.notna().any() else 0
        errors=int(g.has_error.sum()); costs=pd.to_numeric(g.estimated_cost,errors='coerce')
        parents=dict(zip(g.run_id.astype(str),g.parent_run_id.fillna('').astype(str))); depthmax=0
        for rid in parents:
            cur=rid; seen=set(); depth=0
            while cur in parents and parents[cur] and cur not in seen:
                seen.add(cur); cur=parents[cur]; depth+=1
            depthmax=max(depthmax,depth)
        out.append({'trace_id':str(trace),'root_run_name':str(root.run_name),'root_status':'failed' if root.has_error else 'success','run_count':len(g),'llm_call_count':int(g.is_llm.sum()),'tool_call_count':int(g.is_tool.sum()),'error_count':errors,'failure_rate':round(errors/max(len(g),1),4),'retry_tag_count':int(g.tags.fillna('').str.contains('retry',case=False).sum()),'workflow_depth':depthmax,'total_duration_ms':round(float(rootdur or 0),2),'mean_span_duration_ms':round(float(dur.mean()) if dur.notna().any() else 0,2),'total_tokens':int(pd.to_numeric(g.total_tokens,errors='coerce').fillna(0).sum()),'prompt_tokens':int(pd.to_numeric(g.prompt_tokens,errors='coerce').fillna(0).sum()),'completion_tokens':int(pd.to_numeric(g.completion_tokens,errors='coerce').fillna(0).sum()),'total_cost':round(float(costs.sum()),8) if costs.notna().any() else None,'input_chars':int(g.input_chars.sum()),'output_chars':int(g.output_chars.sum())})
    return pd.DataFrame(out)
