"use client";
import {ChevronRight,Network,ArrowUpRight} from "lucide-react";
import {Run,fmt} from "@/lib/simulation";

export type BodyModule={id:string;label:string;cells:string[];target:string;tau_min:number;node_id?:string;max_deviation?:number};
export type BodySystem={id:string;label:string;scope:string;modules:BodyModule[];changed?:number};
export type BodyModel={status?:string;system_count:number;module_count:number;state_count?:number;systems:BodySystem[];gaps:string[];scope:string;checks?:string[];events?:number[];input_profiles?:{event_index:number;channel:string;scale:number;rationale:string}[]};

export function BodyAtlas({run,catalog,timeIndex,selected,onSystem,onNode}:{run:Run|null;catalog:BodyModel|null;timeIndex:number;selected:string;onSystem:(id:string)=>void;onNode:(id:string)=>void}){
 const model=run?.body??catalog;if(!model)return null;const active=run?.body?.status==="active";
 const system=model.systems.find(s=>s.id===selected);
 return <section className="body-atlas"><div className="body-atlas-heading"><div><div className="kicker">CONNECTED BODY</div><h3>전신 계통 지도 <span>{model.system_count}개 계통 · {model.module_count}개 기본 과정</span></h3></div><button onClick={()=>onSystem("all")} className={selected==="all"?"active":""}><Network size={14}/>전체 연결</button></div>
 <p>{active?"기존 순환·당대사에 연결된 기본 생리 네트워크입니다. 계통을 선택하면 아래 그래프와 세포 과정이 좁혀집니다.":"재사용하는 기본 모델의 전체 구성입니다. 사건을 실행하면 각 과정의 상대 반응을 계산합니다."}</p>
 <div className="body-system-grid">{model.systems.map(s=><button key={s.id} onClick={()=>onSystem(s.id)} className={`${selected===s.id?"selected":""} ${active&&(s.changed??0)>0?"responding":""}`}><span className="body-system-dot"/><strong>{s.label}</strong><small>{s.modules.length}개 과정{active?` · 궤적 변화 ${s.changed??0}`:""}</small><ChevronRight size={13}/></button>)}</div>
 {system&&<div className="body-system-detail"><div><b>{system.label}</b><span>{system.scope}</span></div><div className="body-module-grid">{system.modules.map(m=>{const value=run?.traces["body:"+m.id]?.[timeIndex];return<button key={m.id} disabled={!active} onClick={()=>onNode("body:"+m.id)}><strong>{m.label}</strong><span>{m.cells.join(" · ")}</span><b>{value!=null?`${fmt(value,4)} 상대 편차`:"기본 모델"}<ArrowUpRight size={12}/></b></button>})}</div></div>}
 <div className="body-atlas-foot"><span>{active?"상대 상태는 안정 기준 0 · 연결 계수 미보정":"실행할 상태·수식·피드백이 정의된 기본 모델 라이브러리입니다."}</span><details><summary>모델 해상도와 빠진 영역</summary><p>{model.scope}</p><ul>{model.gaps.map(g=><li key={g}>{g}</li>)}</ul><p>상대 편차는 혈중 농도·세포 수·결합률이 아닙니다. 기존 BioGears/Pulse 엔진의 원본 이식이나 임상 검증을 의미하지 않습니다.</p></details></div>
 </section>;
}
