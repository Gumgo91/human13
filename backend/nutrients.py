"""Chemical identity, never a table of guessed food serving compositions."""

# mg/mmol (or mg per mmol of anhydrohexose for starch).
SPECIES={
    "sucrose":{"mw":342.2965,"hexoses":2,"products":{"glucose":1,"fructose":1},"enzyme":"sucrase","label":"sucrose"},
    "lactose":{"mw":342.2965,"hexoses":2,"products":{"glucose":1,"galactose":1},"enzyme":"lactase","label":"lactose"},
    "maltose":{"mw":342.2965,"hexoses":2,"products":{"glucose":2},"enzyme":"maltase","label":"maltose"},
    "starch":{"mw":162.1406,"hexoses":1,"products":{"glucose":1},"enzyme":"amylase","label":"starch unit"},
    "glucose":{"mw":180.156,"hexoses":1,"products":{},"label":"glucose"},
    "fructose":{"mw":180.156,"hexoses":1,"products":{},"label":"fructose"},
    "galactose":{"mw":180.156,"hexoses":1,"products":{},"label":"galactose"},
    "carbohydrate":{"mw":180.156,"hexoses":1,"products":{"glucose":1},"enzyme":"equivalent","label":"carbohydrate equivalent"},
}
# Canonical nutrient identities. A lookup table only — it maps an entity string
# the interpreter emitted to a canonical species; it never infers composition
# from food names or label text.
ALIASES={"Sugar":"sucrose","자당":"sucrose","table sugar":"sucrose","sugar":"sucrose","sucrose":"sucrose","saccharose":"sucrose","포도당":"glucose","glucose":"glucose","d-glucose":"glucose","dextrose":"glucose","과당":"fructose","fructose":"fructose","d-fructose":"fructose","유당":"lactose","lactose":"lactose","맥아당":"maltose","maltose":"maltose","갈락토스":"galactose","galactose":"galactose","전분":"starch","녹말":"starch","starch":"starch","탄수화물":"carbohydrate","carbohydrates":"carbohydrate","carbohydrate":"carbohydrate","carbs":"carbohydrate"}


def nutrient_identity(entity):
    return ALIASES.get((entity or "").strip().casefold())


# Qualifiers that veto a sugar reference scenario. This is a conservative
# guard against interpreter error: it can only ever remove a fabricated
# reference, never create one.
SUGAR_FREE = ("sugar-free", "sugar free", "sugarfree", "sugarless", "제로",
              "무설탕", "무당", "xylitol", "sorbitol", "erythritol", "stevia")


# Foods whose sugar species is presumed for the conditional reference
# scenario only. This is not entity canonicalization - it exists so an
# unidentified candy still explores the sucrose pathway, explicitly flagged
# as a composition assumption.
PRESUMED_SPECIES = {"candy": "sucrose", "candies": "sucrose",
                    "sweet": "sucrose", "sweets": "sucrose"}


def reference_nutrients(plan, allow_assumptions):
    """Run an explicitly conditional 1 g substrate scenario, never infer a serving.

    The substrate must already be identified on the event's entity by the
    interpreter — composition is not guessed from food names. Reference mass
    is an input to the nonlinear solver, not a per-gram scaling law or a dose
    estimate.
    """
    modeled = plan.model_copy(deep=True)
    references = []
    if not allow_assumptions:
        return modeled, references
    for i, event in enumerate(modeled.interventions):
        if event.kind != "nutrition" or event.route not in {"oral", "unspecified"}:
            continue
        if event.quantity is not None and (event.unit or "").casefold() not in {"piece", "pieces", "serving", "servings"}:
            continue
        species = nutrient_identity(event.entity) or PRESUMED_SPECIES.get((event.entity or "").strip().casefold())
        if species == "sucrose" and any(q in (event.label or "").casefold() for q in SUGAR_FREE):
            continue
        if not species:
            continue
        note = (f"Reference scenario: 1 g {species}. Actual nutrient amount is unknown; "
                "this explores the selected substrate pathway and does not predict the consumed serving. ")
        if species == "sucrose":
            note += "Sucrose-containing food is assumed where composition is unspecified. "
        note += "Responses are nonlinear and must not be multiplied by a serving count."
        references.append({"event_index": i, "label": event.label, "entity": species,
                           "quantity": 1., "unit": "g", "basis": "reference", "description": note})
        event.entity = species
        event.quantity = 1.
        event.unit = "g"
    return modeled, references


def normalize_nutrients(plan, text=None):
    """Canonicalize declared nutrient identities; never recover amounts.

    Quantities come from the interpreter only. A food label is not evidence
    for an amount or a composition, so unidentified nutrition events keep a
    null quantity and an explicit missing marker.
    """
    for event in plan.interventions:
        if event.route=="oral" and (event.entity or "").casefold().strip() in {"sodium chloride","table salt","salt"}:
            event.kind="nutrition"
        identity=nutrient_identity(event.entity)
        if identity:
            event.kind="nutrition";event.entity=identity
        other_known=(event.entity or "").casefold().strip() in {"protein","dietary protein","fat","dietary fat","lipid","fiber","dietary fiber","salt","table salt","sodium chloride","potassium"}
        if event.kind=="nutrition" and event.quantity is not None and not identity and not other_known:
            event.quantity=None
            event.missing.append("composition-resolved nutrient mass")
            plan.notes.append(f"{event.label}: total food mass was not converted to carbohydrate because the composition could not be verified.")
    return plan
