import type {Run,Progress} from "./simulation";

export function progressMessage(progress:Progress|null):string {
  if(progress?.message&&!/[가-힣]/.test(progress.message))return progress.message;
  return ["Interpreting the event.","Retrieving molecular evidence.","Calculating the initial model.","Checking the final model.","Finalizing the response."][progress?.stage??0]??"Working on the simulation.";
}

export function reviewMessage(run:Run):string|null {
  const review=run.generation;
  if(!review)return null;
  if(review.planning_status==="not_needed"||review.planning_status==="disabled")return null;
  if(review.planning_status==="failed"){
    const why=review.failure_reason==="output_limit"?"reached its output limit":review.failure_reason==="timeout"?"exceeded its time limit":"could not be completed";
    return `Pathway construction ${why}. The initial model is available; no additional pathways were applied.`;
  }
  if(!review.update)return "Model ready.";
  const {added_states,changed_observables}=review.update;
  if(!added_states&&!changed_observables.length)return "The existing model was retained; no additional changes were applied.";
  const added=added_states?`Added ${added_states} model ${added_states===1?"state":"states"}. `:"";
  return added+(changed_observables.length?`Updated ${changed_observables.length} existing ${changed_observables.length===1?"observable":"observables"}.`:"Existing whole-body results are unchanged.");
}
