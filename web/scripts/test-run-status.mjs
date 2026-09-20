import test from "node:test";
import assert from "node:assert/strict";
import {progressMessage,reviewMessage} from "../lib/run-status.ts";

test("detailed backend phases are displayed instead of one stage label",()=>{
  assert.equal(progressMessage({stage:2,message:"Checking primary research."}),"Checking primary research.");
  assert.equal(progressMessage({stage:2,message:"Preparing English descriptions."}),"Preparing English descriptions.");
});
test("no change, additional states and failed generation are explicit",()=>{
  assert.match(reviewMessage({generation:{planning_status:"completed",update:{added_states:0,changed_observables:[]}}}),/no additional changes/);
  assert.match(reviewMessage({generation:{planning_status:"completed",update:{added_states:1,changed_observables:[]}}}),/Existing whole-body results are unchanged/);
  assert.match(reviewMessage({generation:{planning_status:"failed",failure_reason:"output_limit"}}),/output limit/);
  assert.match(reviewMessage({generation:{planning_status:"failed",failure_reason:"timeout"}}),/time limit/);
});

test("native completion has no additional-review notice",()=>{
  assert.equal(reviewMessage({generation:{planning_status:"not_needed",complete:true}}),null);
  assert.equal(reviewMessage({generation:{planning_status:"disabled"}}),null);
});
