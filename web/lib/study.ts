import type {Metric,Run,Source} from "./simulation";

export type SensitivityParameter={id:string;label:string;value:number;unit:string;low:number;high:number;origin:string};
export type StudyReport={
 schema_version:string;guidance:(Source&{date:string;status:string;scope:string})[];interpretation:string;question_of_interest:string;
 context_of_use:{purpose:string;population:string;decision_role:string;outside_scope:string};
 model_risk:{status:string;rationale:string};verification:{checks:{id:string;label:string;status:string;detail:string}[]};
 calibration:{status:string;detail:string};independent_validation:{status:string;detail:string};
 applicability:{id:string;label:string;status:string;detail:string}[];
 parameters:{coverage:string;complete:boolean;rows:{id:string;label:string;unit:string;selected_value:number;origin:string;selection_rationale:string;alternative_evidence_status:string}[]};
 sensitivity_parameters:SensitivityParameter[];gaps:string[];refinement_plan:string;
 reproducibility:{plan_hash:string;engine_hash?:string;software_at_run?:Record<string,string>};
};
export type Comparison={id:string;label:string;unit:string;status:string;max_absolute_delta:number|null;signed_delta_at_max?:number;time_at_max_min?:number;final_delta?:number;detail?:string};
export type AnalysisScenario={id:string;label:string;kind:string;status:string;detail?:string;value?:number;baseline_value?:number;unit?:string;time?:number[];metrics?:Metric[];comparisons:Comparison[]};
export type AnalysisReport={analysis_id:string;run_id:string;engine_changed:boolean;llm_calls:number;analysis_plan:{method:string;scope:string;comparison:string};baseline:{time:number[];metrics:Metric[];plan_hash:string};study_report:StudyReport;scenarios:AnalysisScenario[];unselected_parameters:string[];structural_status:string;limitations:string[]};
export type StudyRun=Run&{study_report?:StudyReport};
