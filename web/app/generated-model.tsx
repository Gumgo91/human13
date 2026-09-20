"use client";
import {ChevronRight,Network,Check,SlidersHorizontal} from "lucide-react";
import {Run,Settings,fmt} from "@/lib/simulation";
import {Input} from "@/components/ui/input";
import {Slider} from "@/components/ui/slider";

export function GeneratedModelSummary({run,onSelect,onRebuild,busy}:{run:Run;onSelect:(id:string)=>void;onRebuild:()=>void;busy:boolean}){
 const model=run.generated;
 if(run.is_preview)return <div className="model-composition pending"><Network size={18}/><div><b>입력에 필요한 모델을 구성하고 있습니다</b><p>기본 수식은 먼저 탐색할 수 있습니다. 새 상태·반응·관측식은 검사 후 연결됩니다.</p></div></div>;
 if(!model)return null;
 const active=model.status==="compiled";const states=run.nodes.filter(n=>n.generated);
 return <section className="model-composition"><div className="composition-heading"><div><div className="kicker">COMPOSED MODEL</div><b>{active?"이 사건에서 구성된 생리 모델":model.status==="disabled"?"생성 수식의 실행이 꺼져 있습니다":"기존 수식의 실행 결과"}</b></div><span>{run.coverage.events_modeled??0}/{run.coverage.events_total??0} 사건 연결</span></div>
 <div className="event-coverage">{run.event_coverage?.map(e=><button key={e.event_index} className={e.status==="uncovered"?"uncovered":""} onClick={()=>onSelect("event:"+e.event_index)}>{e.status==="modeled"&&<Check size={12}/>}<span>{e.label}</span><small>{e.generated?e.native?"기존 + 생성 수식":"생성 수식":e.native?"기존 수식":"연결 없음"}</small></button>)}</div>
 {active&&<><div className="composition-counts"><span>새 상태 <b>{model.states}</b></span><span>반응·조절식 <b>{model.processes}</b></span><span>추가 관측식 <b>{model.observations.length}</b></span><span>조절 계수 <b>{model.parameters.length}</b></span></div><details><summary>구성된 상태와 관측식 살펴보기<ChevronRight size={14}/></summary><div className="composed-states">{states.map(n=><button key={n.id} onClick={()=>onSelect(n.id)}><span>{n.group}</span>{n.label}<ChevronRight size={12}/></button>)}</div>{model.observations.map(o=><button className="composed-observation" key={o.id} onClick={()=>onSelect("output:"+o.key)}><b>{o.label}</b><code>{o.expression}</code></button>)}<div className="composition-checks">{model.checks.map(c=><span key={c}><Check size={11}/>{c}</span>)}</div><p>검사는 수식의 실행 가능성을 확인합니다. 생성 계수와 상대 지수는 실측값으로 보정되지 않은 탐색 가정입니다.</p></details></>}
 {model.reason&&<p>{model.reason}</p>}
 {run.generation?.planning_status==="failed"&&<p className="amber-text">기본 수식은 계산했지만 추가 LLM 기전 구성은 완료되지 않았습니다. 실행 기록에서 이유를 확인하거나 경로를 다시 구성하세요.</p>}
 {!run.generation?.complete&&run.event_coverage?.some(e=>e.status==="uncovered")&&<p className="amber-text">아직 경로를 구성하지 못한 사건이 있습니다. 기본 수치가 보이더라도 해당 사건의 영향을 계산한 것은 아닙니다.</p>}
 <div className="composition-footer"><small>{run.generation?.cached?"로컬 모델 계획 재사용":run.llm.replayed?"같은 모델 계획으로 재계산":"수식·계획 저장됨"}{run.generation?.repaired?` · 자동 수정 ${run.generation.repaired}회`:""}</small><button disabled={busy} onClick={onRebuild}>경로부터 다시 구성<ChevronRight size={12}/></button></div>
 </section>
}

export function GeneratedParameters({run,settings,onChange,disabled}:{run:Run|null;settings:Settings;onChange:(id:string,value:number)=>void;disabled:boolean}){
 const parameters=run?.generated?.parameters??[];
 if(!parameters.length)return null;
 return <details className="parameter-settings generated-parameters"><summary><span><SlidersHorizontal size={13}/>이 모델의 계수 {parameters.length}개</span><ChevronRight size={14}/></summary><p>생성된 계수의 범위 안에서 바꾸고 아래 재계산을 누르세요. 같은 수식을 유지한 채 하류 변화를 비교합니다.</p>{parameters.map(p=>{const value=settings.model_parameters[p.id]??p.value;return <div className="parameter-input" key={p.id}><label htmlFor={"generated-"+p.id}>{p.label}<small>{p.unit==="dimensionless"?"상대 계수":p.unit}</small></label><Input id={"generated-"+p.id} aria-label={p.label+" 생성 계수"} type="number" min={p.low} max={p.high} step="any" value={value} disabled={disabled||p.low===p.high} onChange={e=>{const v=Number(e.target.value);if(Number.isFinite(v)&&v>=p.low&&v<=p.high)onChange(p.id,v)}}/><Slider aria-label={p.label+" 범위 조절"} min={p.low} max={p.high} step={(p.high-p.low)/100||1} value={[value]} disabled={disabled||p.low===p.high} onValueChange={v=>onChange(p.id,v[0])}/><small className="generated-range">{fmt(p.low,3)} ~ {fmt(p.high,3)}</small>{p.rationale&&<details><summary>계수의 가정</summary><p>{p.rationale}</p></details>}</div>})}</details>
}
