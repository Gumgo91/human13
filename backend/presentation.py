METHOD_LABELS={"source_model":"Existing model equations","assumption":"Mechanistic equations and assumed coefficients","derived":"Unit and definition calculation","llm_hypothesis":"unconnected pathway","llm_surrogate":"LLM approximation, computational reflection","hybrid":"Existing formula + LLM connection","knowledge":"Retrieved evidence","unknown":"Unmodeled"}


def make_node(id,label,group,stage,method,description,**extra):
    return {"id":id,"label":label,"group":group,"stage":stage,"method":method,"method_label":METHOD_LABELS[method],"description":description,"state_key":None,"unit":None,"cell_types":[],"equations":[],"parameters":[],"sources":[],"limitations":[],**extra}
