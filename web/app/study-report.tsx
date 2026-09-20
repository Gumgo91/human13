"use client";
import {useEffect,useRef,useState} from "react";
import {ArrowUpRight,Download,FlaskConical,LoaderCircle} from "lucide-react";
import {LineChart,Line,XAxis,YAxis,Tooltip,CartesianGrid,ResponsiveContainer,Legend} from "recharts";
import {Button} from "@/components/ui/button";
import {API,fmt} from "@/lib/simulation";
import type {AnalysisReport,StudyRun} from "@/lib/study";

const statuses:Record<string,string>={passed:"수치 검사 통과",partial:"일부 검사",not_recorded:"기록 없음",not_applied:"적용 없음",not_declared:"장부 미선언",not_evaluated:"미평가",failed:"실패"};
function save(value:unknown,name:string){const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:"application/json"}));const a=document.createElement("a");a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}

export function StudyReportPanel({run,disabled}:{run:StudyRun|null;disabled:boolean}){
 const report=run?.study_report;
 const [selected,setSelected]=useState<string[]>(()=>(report?.sensitivity_parameters.some(p=>p.id.startsWith("model:"))?report.sensitivity_parameters.filter(p=>p.id.startsWith("model:")):report?.sensitivity_parameters??[]).slice(0,6).map(p=>p.id));
 const [fraction,setFraction]=useState(.2);const [structural,setStructural]=useState(true);const [busy,setBusy]=useState(false);const [error,setError]=useState("");
 const [analysis,setAnalysis]=useState<AnalysisReport|null>(null);const [metricId,setMetricId]=useState("");const [scenarioId,setScenarioId]=useState("");const controller=useRef<AbortController|null>(null);
 useEffect(()=>()=>controller.current?.abort(),[]);
 useEffect(()=>{const id=new URLSearchParams(window.location.search).get("analysis");if(!id||!/^[a-f0-9]{16}$/.test(id)||!run)return;const abort=new AbortController();fetch(`${API}/api/analyses/${id}`,{signal:abort.signal}).then(r=>{if(!r.ok)throw Error("저장한 분석을 찾지 못했습니다.");return r.json() as Promise<AnalysisReport>}).then(r=>{if(r.run_id!==run.run_id)return;setAnalysis(r);setMetricId(r.baseline.metrics.find(m=>m.id.startsWith("model:")&&m.values)?.id??r.baseline.metrics.find(m=>m.values)?.id??"");setScenarioId(r.scenarios.find(s=>s.status==="computed")?.id??"");}).catch(e=>{if(e.name!=="AbortError")setError(e.message);});return()=>abort.abort();},[run?.run_id]);
 async function analyze(){if(!run)return;setBusy(true);setError("");const abort=new AbortController();controller.current=abort;
  try{const response=await fetch(`${API}/api/runs/${run.run_id}/analysis`,{method:"POST",headers:{"Content-Type":"application/json"},signal:abort.signal,body:JSON.stringify({parameter_ids:selected,fraction,include_structural:structural})});
   const data=await response.json();if(!response.ok)throw Error(data&&typeof data==="object"&&"detail" in data&&typeof data.detail==="string"?data.detail:"분석 조건을 확인해 주세요.");
   const result=data as AnalysisReport;setAnalysis(result);setMetricId(result.baseline.metrics.find(m=>m.id.startsWith("model:")&&m.values)?.id??result.baseline.metrics.find(m=>m.values)?.id??"");setScenarioId(result.scenarios.find(s=>s.status==="computed")?.id??"");
  }catch(e){if((e as Error).name!=="AbortError")setError((e as Error).message);}finally{setBusy(false);}
 }
 const observed=analysis?.baseline.metrics.find(m=>m.id===metricId);
 const scenario=analysis?.scenarios.find(s=>s.id===scenarioId);
 const variant=scenario?.metrics?.find(m=>m.id===metricId);
 // Scenario grids share intervention boundaries; interpolate only for removed extensions.
 const rows=observed?.values&&analysis?analysis.baseline.time.map((time,i)=>{
  let alternative:number|null=null;
  if(variant?.values&&scenario?.time){const times=scenario.time;let j=times.findIndex(t=>t>=time);if(j<0)j=times.length-1;const a=Math.max(0,j-1);alternative=times[j]===times[a]?variant.values[j]:variant.values[a]+(variant.values[j]-variant.values[a])*(time-times[a])/(times[j]-times[a]);}
  return{time,baseline:observed.values?.[i],alternative};
 }):[];
 return <section className="study-panel"><div className="study-heading"><div><div className="kicker">MODEL EVIDENCE</div><h3>모델의 근거와 가정의 영향</h3></div>{report&&<Button variant="outline" size="sm" onClick={()=>save(analysis??report,`human13-${run?.run_id}-${analysis?"sensitivity":"evidence"}.json`)}><Download size={13}/>보고서 저장</Button>}</div>
 <p>FDA의 2026년 QSP 드래프트와 ICH M15를 참고해 계산 검사, 실측 검증, 사용 범위를 구분합니다.</p>
 <div className="guidance-links"><a href="https://www.fda.gov/media/193230/download" target="_blank" rel="noreferrer">QSP · MABEL/FIH · 2026.06 초안<ArrowUpRight size={13}/></a><a href="https://www.fda.gov/media/184747/download" target="_blank" rel="noreferrer">ICH M15 · 2026.06 최종<ArrowUpRight size={13}/></a></div>
 {!report?<p className="study-empty">사건을 계산하면 실행별 근거 보고서와 가정 영향 분석이 여기에 나타납니다.</p>:<>
 <div className="study-statuses"><div><b>계산 검사</b><span>실행별 결과 아래 표시</span></div><div><b>실측 보정</b><span>미실시</span></div><div><b>독립 데이터 검증</b><span>미실시</span></div></div>
 <p className="study-question">{report.question_of_interest}</p>
 <details><summary>사용 목적 · 검증 범위 · 남은 근거</summary><dl><dt>사용 목적</dt><dd>{report.context_of_use.purpose}</dd><dt>대상</dt><dd>{report.context_of_use.population}</dd><dt>결과의 역할</dt><dd>{report.context_of_use.decision_role}</dd><dt>적용 범위 밖</dt><dd>{report.context_of_use.outside_scope}</dd><dt>모델 위험 평가</dt><dd>{report.model_risk.rationale}</dd></dl>
 <div className="study-checks">{report.verification.checks.map(c=><div key={c.id}><b>{c.label}<span className={c.status==="passed"?"passed":""}>{statuses[c.status]??c.status}</span></b><p>{c.detail}</p></div>)}</div>
 <p>{report.calibration.detail}</p><p>{report.independent_validation.detail}</p><ul>{report.gaps.map(g=><li key={g}>{g}</li>)}</ul><p>{report.refinement_plan}</p></details>
 <details><summary>파라미터 근거 장부 · {report.parameters.rows.length}개 항목</summary><p>{report.parameters.coverage}</p><p>문헌 링크가 있다는 것만으로 이 수치의 실측 근거가 확인되지는 않습니다. 대안 수치·종·실험 조건은 아직 수집되지 않았습니다.</p><div className="study-table-scroll"><table><thead><tr><th>계수</th><th>사용값</th><th>선택 근거</th></tr></thead><tbody>{report.parameters.rows.map(p=><tr key={p.id}><td>{p.label}<small>{p.origin}</small></td><td>{fmt(p.selected_value,5)} {p.unit}</td><td>{p.selection_rationale}</td></tr>)}</tbody></table></div></details>
 <div className="sensitivity-panel"><div className="study-heading"><h4><FlaskConical size={16}/>가정을 바꾸면 얼마나 달라질까?</h4><span>로컬 재계산 · LLM 호출 없음</span></div><p>저장된 실행의 조건을 기준으로 비교합니다. 왼쪽에서 바꾼 설정은 먼저 재계산해 저장하세요.</p>
 <details><summary>비교할 계수 선택 · {selected.length}/12개</summary><p>생성 계수와 공개 설정을 선택할 수 있습니다. 입력에서 사용되지 않는 설정은 차이가 없을 수 있습니다.</p><div className="sensitivity-options">{report.sensitivity_parameters.map(p=><label key={p.id}><input type="checkbox" checked={selected.includes(p.id)} disabled={busy||(!selected.includes(p.id)&&selected.length>=12)} onChange={e=>setSelected(a=>e.target.checked?[...a,p.id]:a.filter(id=>id!==p.id))}/><span>{p.label}<small>{fmt(p.value,4)} {p.unit}</small></span></label>)}</div></details>
 <div className="sensitivity-actions"><label>계수 변화 <select aria-label="민감도 분석 계수 변화 비율" value={fraction} disabled={busy} onChange={e=>setFraction(Number(e.target.value))}><option value={.1}>±10%</option><option value={.2}>±20%</option><option value={.5}>±50%</option></select></label><label><input type="checkbox" checked={structural} disabled={busy} onChange={e=>setStructural(e.target.checked)}/>LLM 연결 제거도 비교</label><Button disabled={disabled||busy||!!run?.is_preview||(!selected.length&&!structural)} onClick={()=>void analyze()}>{busy?<LoaderCircle size={14} className="spin"/>:<FlaskConical size={14}/>}가정 영향 계산</Button></div>
 <p className="sensitivity-note">한 번에 한 계수만 변경합니다. 기준이 0이면 허용 범위의 비율을 쓰며, 경계에서는 변화가 제한됩니다. 결과는 신뢰구간이 아닙니다.</p>
 {error&&<p role="alert" className="error-message">{error}</p>}
 {analysis&&<div className="sensitivity-results">{analysis.engine_changed&&<p className="source-warning">저장 이후 엔진이 바뀌어 현재 코드로 기준 모델부터 다시 계산했습니다. 아래 비교는 모두 그 기준을 사용합니다.</p>}
 <div className="sensitivity-selectors"><label>관측값<select value={metricId} onChange={e=>setMetricId(e.target.value)}>{analysis.baseline.metrics.filter(m=>m.values).map(m=><option key={m.id} value={m.id}>{m.label} · {m.unit}</option>)}</select></label><label>비교 시나리오<select value={scenarioId} onChange={e=>setScenarioId(e.target.value)}>{analysis.scenarios.map(s=><option key={s.id} value={s.id}>{s.label}{s.status==="failed"?" · 실패":s.status==="unchanged"?" · 변화 없음":""}</option>)}</select></label></div>
 {!!rows.length&&<div className="sensitivity-chart"><ResponsiveContainer width="100%" height={235}><LineChart data={rows} margin={{top:10,right:15,bottom:5,left:0}}><CartesianGrid stroke="#263a48" strokeDasharray="3 4"/><XAxis dataKey="time" tick={{fill:"#8fa8b9",fontSize:11}} minTickGap={40} tickFormatter={t=>`${fmt(t)}분`}/><YAxis tick={{fill:"#8fa8b9",fontSize:11}} domain={["auto","auto"]} tickFormatter={v=>fmt(v,3)} width={60}/><Tooltip contentStyle={{background:"#14212c",border:"1px solid #425366",fontSize:12}}/><Legend/><Line dataKey="baseline" name="기준 모델" stroke="#80dbcf" dot={false} isAnimationActive={false}/><Line dataKey="alternative" name="변경 모델" stroke="#dab37c" dot={false} connectNulls={false} isAnimationActive={false}/></LineChart></ResponsiveContainer></div>}
 {scenario&&(scenario.status!=="computed"||!variant?.values)&&<p className="source-warning">{scenario.detail??"이 시나리오에는 해당 관측값이 없습니다. 값을 0으로 채우지 않았습니다."}</p>}
 <div className="study-table-scroll"><table><thead><tr><th>시나리오</th><th>최대 절대 차이</th><th>발생 시각</th></tr></thead><tbody>{analysis.scenarios.map(s=>{const c=s.comparisons.find(c=>c.id===metricId);return<tr key={s.id} className={s.id===scenarioId?"selected":""}><td><button onClick={()=>setScenarioId(s.id)}>{s.label}</button>{s.kind==="parameter"&&<small>{fmt(s.baseline_value,4)} → {fmt(s.value,4)} {s.unit}</small>}</td><td>{s.status==="failed"?"계산 실패":s.status==="unchanged"?"변경되지 않음":c?.status==="compared"?`${fmt(c.max_absolute_delta,5)} ${observed?.unit??""}`:"비교 불가"}</td><td>{c?.time_at_max_min!=null?`${fmt(c.time_at_max_min)}분`:"—"}</td></tr>})}</tbody></table></div>
 <p>{analysis.scenarios.filter(s=>s.status==="computed").length}개 계산 · {analysis.scenarios.filter(s=>s.status==="failed").length}개 실패 · 미선택 계수 {analysis.unselected_parameters.length}개</p>{analysis.structural_status==="not_available"&&<p>이 실행에는 제거해서 비교할 LLM 수치 확장이 없습니다.</p>}<details><summary>분석 방법과 한계</summary><p>{analysis.analysis_plan.method}</p><p>{analysis.analysis_plan.comparison}</p><ul>{analysis.limitations.map(l=><li key={l}>{l}</li>)}</ul><code>ANALYSIS {analysis.analysis_id}<br/>PLAN {analysis.baseline.plan_hash}</code></details>
 </div>}</div>
 </>}
 </section>;
}
